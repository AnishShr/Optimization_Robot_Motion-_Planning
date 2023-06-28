import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit
import sys
import highway_env
from pprint import pprint
from time import sleep
import math
# %matplotlib inline

from vpsto.vpsto import VPSTO, VPSTOOptions, VPSTOSolution

import matplotlib.pyplot as plt

jax_interp = jit(jnp.interp)

config = {
    'initial_lane_id': 1,
    'duration': 20,
    'action': dict(
        type="ContinuousAction",
        longitudinal=True,
        lateral=True
    ),
    'observation': dict(
        type="Kinematics",
        vehicles_count=7,
        features=["x", "y", "vx", "vy", "heading"],
        features_range=dict(
            x=[-500, 500],
            y=[-4, 8],
            vx=[-10, 50],
            vy=[-10, 10]
            ),
        absolute=False,
        normalize=False
    ),
    'policy_frequency': 50,
    'simulation_frequency': 100,
    'vehicles_density': 2
    
}

def compute_path_parameters(x_path, y_path):

    Fx_dot = jnp.diff(x_path)
    Fy_dot = jnp.diff(y_path)

    Fx_dot = jnp.hstack(( Fx_dot[0], Fx_dot  ))

    Fy_dot = jnp.hstack(( Fy_dot[0], Fy_dot  ))
        
    Fx_ddot = jnp.diff(Fx_dot)
    Fy_ddot = jnp.diff(Fy_dot)

    Fx_ddot = jnp.hstack(( Fx_ddot[0], Fx_ddot  ))

    Fy_ddot = jnp.hstack(( Fy_ddot[0], Fy_ddot  ))
    
    arc = jnp.cumsum( jnp.sqrt( Fx_dot**2+Fy_dot**2 )   )
    arc_vec = jnp.hstack((0, arc[0:-1] ))
    # arc_vec = arc 

    arc_length = arc_vec[-1]

    kappa = (Fy_ddot*Fx_dot-Fx_ddot*Fy_dot)/((Fx_dot**2+Fy_dot**2)**(1.5))

    return Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length


def global_to_frenet(x_path, y_path, initial_state, arc_vec, Fx_dot, Fy_dot, kappa ):

    x_global_init, y_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init = initial_state
    idx_closest_point = jnp.argmin( jnp.sqrt((x_path-x_global_init)**2+(y_path-y_global_init)**2))
    closest_point_x, closest_point_y = x_path[idx_closest_point], y_path[idx_closest_point]

    x_init = arc_vec[idx_closest_point]

    kappa_interp = jax_interp(x_init, arc_vec, kappa)
    kappa_pert = jax_interp(x_init+0.001, arc_vec, kappa)

    kappa_prime = (kappa_pert-kappa_interp)/0.001

    Fx_dot_interp = jax_interp(x_init, arc_vec, Fx_dot)
    Fy_dot_interp = jax_interp(x_init, arc_vec, Fy_dot)

    normal_x = -Fy_dot_interp
    normal_y = Fx_dot_interp

    normal = jnp.hstack((normal_x, normal_y   ))
    vec = jnp.asarray([x_global_init-closest_point_x,y_global_init-closest_point_y ])
    y_init = (1/(jnp.linalg.norm(normal)))*jnp.dot(normal,vec)
    
    psi_init = psi_global_init-jnp.arctan2(Fy_dot_interp, Fx_dot_interp)
    psi_init = jnp.arctan2(jnp.sin(psi_init), jnp.cos(psi_init))
    
    vx_init = v_global_init*jnp.cos(psi_init)/(1-y_init*kappa_interp)
    vy_init = v_global_init*jnp.sin(psi_init)

    psidot_init = psidot_global_init-kappa_interp*vx_init

    ay_init = vdot_global_init*jnp.sin(psi_init)+v_global_init*jnp.cos(psi_init)*psidot_init
    
    ax_init_part_1 = vdot_global_init*jnp.cos(psi_init)-v_global_init*jnp.sin(psi_init)*psidot_init
    ax_init_part_2 = -vy_init*kappa_interp-y_init*kappa_prime*vx_init

    ax_init = (ax_init_part_1*(1-y_init*kappa_interp)-(v_global_init*jnp.cos(psi_init))*(ax_init_part_2) )/((1-y_init*kappa_interp)**2)
        
    psi_fin = 0.0

    return x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init


# def create_obs_lists(obs):
#     obs_x = []
#     obs_y = []
#     for i in range(obs.shape[0]):   
#         obs_x.append(obs[i][0])
#         obs_y.append(obs[i][1])

#     return np.array(obs_x), np.array(obs_y)


def compute_cumutative_collision_cost(x, y, obs_x, obs_y, obs_a, obs_b):
    obs_x = np.array(obs_x).reshape(len(obs_x), 1)
    obs_y = np.array(obs_y).reshape(len(obs_y), 1)
    
    cost_collision = np.maximum(np.zeros(x.shape), -((x - obs_x[:, np.newaxis])/((2*obs_a)))**2 - ((y - obs_y[:, np.newaxis])/(2*obs_b))**2 + 1) 
    cost_collision_combined = np.sum(cost_collision, axis=0)
    final_cost_collision = cost_collision_combined[0]    

    return final_cost_collision


# Computing various losses
def loss_limits(candidates):
    q = candidates['pos']
    d_min = np.maximum(np.zeros_like(q), - q + q_min)
    d_max = np.maximum(np.zeros_like(q), q - q_max)
    return np.sum(d_min, axis=(1,2)) + np.sum(d_max, axis=(1,2))

def collision_loss(candidates):
    costs = []
    for traj in candidates['pos']:
        cost = compute_cumutative_collision_cost(traj[:, 0],
                                            traj[:, 1],
                                            obs_x,
                                            obs_y,
                                            obs_a, 
                                            obs_b) 
    
        cost = np.sum(cost)
    
        costs.append(cost)
    
    costs = np.array(costs)
    return costs

def loss_curvature(candidates):
    dq = candidates['vel']
    ddq = candidates['acc']
    dq_sq = np.sum(dq**2, axis=-1)
    ddq_sq = np.sum(ddq**2, axis=-1)
    dq_ddq = np.sum(dq*ddq, axis=-1) 
    return np.mean((dq_sq * ddq_sq - dq_ddq**2) / (dq_sq**3 + 1e-6), axis=-1)

def loss_path_offset(candidates):
    costs = []
    for traj in candidates['pos']:
        cost = traj[:, 1]**2
        cost = np.sum(cost)

        costs.append(cost)
    costs = np.array(costs)
    return costs
        

def loss(candidates):
    cost_curvature = loss_curvature(candidates)
    cost_collision = collision_loss(candidates)
    cost_limits = loss_limits(candidates)
    cost_path_offset = loss_path_offset(candidates)
    # return candidates['T'] + 1e-2 * cost_curvature + 4*1e4 * cost_collision + 1e6 * cost_limits
    return candidates['T'] + 1e-2 * cost_curvature + 1e3 * cost_collision + 1e3 * cost_limits + 80 * cost_path_offset 
    # return candidates['T'] + 1e-3 * cost_curvature + 1e4 * cost_collision + 1e3 * cost_limits + 1e2 * cost_path_offset 


env = gym.make('highway-v0', config=config)
# pprint(env.config)
# sys.exit(0)
np.random.seed(33)
env.reset()
print(type(env.vehicle.position))

ego_vel_lim = np.array([50.0, 5.0])
ego_acc_lim = np.array([20, 5.0])
num_via = 2
num_eval = 100
num_pop_size = 80
ndof = 2

# num_obs = config['observation']['vehicles_count']
# print(f"num_obs: {num_obs}")

obs_a = 6.0
obs_b = 3.2

ego_x, ego_y = env.vehicle.position

q_min = np.array([0, -4])
q_max = np.array([10000, 8])

path_x = np.linspace(env.vehicle.position[0], env.vehicle.position[0] + 1000, 2000)
path_y = np.array([env.vehicle.position[1] for x in path_x])

frenet_path = []
Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length = compute_path_parameters(path_x, path_y)
for i in range(len(path_x)):
    
    x_global_init = path_x[i]
    y_global_init = path_y[i]

    # v_global_init = 10
    v_global_init = 50
    vdot_global_init = 0

    psi_global_init = np.arctan2(Fy_dot[i], Fx_dot[i])
    psidot_global_init = 0

    initial_state =  x_global_init, y_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init

    x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init = global_to_frenet(path_x, path_y, initial_state, arc_vec, Fx_dot, Fy_dot, kappa)
    frenet_path.append([x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init])

frenet_path = np.array(frenet_path)

frenet_x = frenet_path[:, 0]
frenet_y = frenet_path[:, 1]

# print(f"Frenet X: {frenet_x}")
# print(f"Frenet Y: {frenet_y}")


action = [0.0, 0.0]
wheel_base = 2.5

final_y = env.vehicle.position[1]

# while True:
for _ in range(1000):    

    # break
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    print(obs[0])
    # break

    x_ego_global_init = obs[0][0]
    y_ego_global_init = obs[0][1]

    # psi_ego_global_init = np.arctan2(Fy_dot[i+1]-Fy_dot[i], Fx_dot[i+1]-Fx_dot[i])
    psi_ego_global_init = 0
    # psi_ego_global_init = np.arctan2(Fy_dot[0], Fx_dot[0])

    obstacles_x = obs[:, 0][1:]
    obstacles_y = obs[:, 1][1:]

    ego_initial_state =  x_ego_global_init, y_ego_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init
    x_ego_init, y_ego_init, vx_ego_init, vy_ego_init, ax_ego_init, ay_ego_init, psi_ego_init, psi_ego_fin, psidot_ego_init = global_to_frenet(path_x, path_y, ego_initial_state, arc_vec, Fx_dot, Fy_dot, kappa)

    print("ego vehicle in frenet coordinates:")
    print(x_ego_init, y_ego_init) 

    obs_x = x_ego_init + obstacles_x
    obs_y = y_ego_init + obstacles_y   

    # print(obs_x)
    # print(obs_y)

    x_ego_final = x_ego_init + 100
    y_ego_final = frenet_y[0]
    

    print(f"Init: {x_ego_init, y_ego_init}")
    print(f"Final: { x_ego_final, y_ego_final}")

    opt = VPSTOOptions(ndof=ndof)
    opt.N_via = num_via
    opt.N_eval = num_eval
    opt.pop_size = num_pop_size
    opt.vel_lim = ego_vel_lim
    opt.acc_lim = ego_acc_lim

    traj_opt = VPSTO(opt)
    traj_opt.ndof = ndof

    q0 = np.array([x_ego_init, y_ego_init])
    qT = np.array([x_ego_final, y_ego_final])

    # dq0 = np.array([10.0, 0.0])
    # dqT = np.array([10.0, 0.0])

    dq0 = np.array([obs[0][2], obs[0][3]])
    dqT = np.array([obs[0][2], obs[0][3]])

    sol = traj_opt.minimize(loss,
                            q0=q0,
                            qT=qT,
                            dq0=dq0,
                            dqT=dqT)
    
    t_traj = np.linspace(0, sol.T_best, 500)
    pos, vel, acc = sol.get_posvelacc(t_traj)

    # print(f"Position: \n {pos}")
    # print(f"Velocity: \n     {vel}")
    # print(f"Acceleration: \n {acc}")

    # break

    pos_x = pos[:, 0]
    pos_y = pos[:, 1]

    vel_x = vel[:, 0]
    vel_y = vel[:, 1]

    acc_x = acc[:, 0]
    acc_y = acc[:, 1]

    # print(f"Vel_X: {vel_x}")
    # print(f"Vel_Y: {vel_y}")

    curvature = (acc_y * vel_x - vel_y * acc_x)/(((vel_x**2) + (vel_y**2) + 0.001)**1.5)
    
    steer = jnp.arctan(curvature * wheel_base)
    # print(f"Steer: {steer}")

    velocity = ((vel_x**2) + (vel_y**2))**0.5
    acceleration = jnp.diff(velocity)
    # print(f"Acceleration: {acceleration}")

    mean_steer = jnp.mean(steer[:10])
    mean_acc = jnp.mean(acceleration[:10])

    action = [mean_acc, mean_steer]
    print(f"Action executed: {action}")
    print("-------------------------------")