import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random

# import time
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

import scipy
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline

import cvxpy as cp

import gymnasium as gym
import numpy as np
import sys
import highway_env
from pprint import pprint
from time import sleep

jax_interp = jit(jnp.interp)

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

def pol_matrix_comp(t):
    
    delt = np.abs(t[1]-t[0])[0]

    num = len(t)

    Ad = np.array([[1, delt, 0.5*delt**2],[0, 1, delt],[0, 0, 1]])
    Bd = np.array([[1/6*delt**3, 0.5*delt**2, delt]]).T


    P = np.zeros((num-1, num-1))
        
    Pdot = np.zeros((num-1, num-1))
        
        
    Pddot = np.zeros((num-1, num-1))

    Pint =  np.zeros((num, 3))
    Pdotint =  np.zeros((num, 3))
    Pddotint = np.zeros((num, 3))        

    for i in range(0,num-1):
        for j in range(0,i):
            temp = np.dot(np.linalg.matrix_power(Ad, (i-j)), Bd)
            
            P[i][j] = temp[0]
            Pdot[i][j] = temp[1]
            Pddot[i][j] = temp[2]

    for i in range(0, num):
        temp = np.linalg.matrix_power(Ad,i)
        
        Pint[i] = temp[0]
        Pdotint[i] = temp[1]
        Pddotint[i] = temp[2]

    P = np.vstack((np.zeros((1,num-1)), P))
        
    Pdot = np.vstack((np.zeros((1,num-1)), Pdot))

    Pddot = np.vstack((np.zeros((1,num-1)), Pddot))


    P = np.hstack((Pint, P))
    Pdot = np.hstack((Pdotint, Pdot))
    Pddot = np.hstack((Pddotint, Pddot))

    return P, Pdot, Pddot



# def create_points(x_frenet, y_frenet, k):

#     points_x = []
#     points_y = []

#     for i in range(k):
#         points_x.append((x_frenet+10) + i*10)
#         points_y.append(0)

#     for i in range(k):
#         for j in range(k):
#             # points_x.append(30 + j*10)
#             # points_y.append(-40 + i*10)
#             # points_x.append((x_frenet+10) + j*10)
#             # points_y.append((y_frenet-30) + i*10)
#             points_x.append((x_frenet+10) + j*10)
#             points_y.append((y_frenet-15) + i*3)


#     return points_x, points_y

def create_points(x_frenet, y_frenet):

    points_x = []
    points_y = []

    for i in range(10):
        points_x.append((x_frenet+5) + i*5)
        points_y.append(0)

    for i in range(-4, 13):
        for j in range(10):            
            points_x.append((x_frenet+5) + j*5)
            # points_y.append((y_frenet-8) + i)
            points_y.append(i)

    
    return points_x, points_y


def compute_cumutative_collision_cost(x, y, obs_x, obs_y, obs_a, obs_b):
    obs_x = np.array(obs_x).reshape(len(obs_x), 1)
    obs_y = np.array(obs_y).reshape(len(obs_y), 1)

    # cost_collision = -((x - obs_x[:, np.newaxis])/((2*obs_a)))**2 - ((y - obs_y[:, np.newaxis])/(obs_b))**2 + 1
    cost_collision = np.maximum(np.zeros(x.shape), -((x - obs_x[:, np.newaxis])/((2*obs_a)))**2 - ((y - obs_y[:, np.newaxis])/(2*obs_b))**2 + 1) 

    # print(cost_collision.shape)
    cost_collision_combined = np.zeros((len(obs_x), x.shape[0]))
    # print(cost_collision_combined.shape)

    for i in range(obs_x.shape[0]):
        cost_collision_combined[i] = np.sum(cost_collision[i], axis=1)
    
    final_cost_collision = np.sum(cost_collision_combined, axis=0)

    return final_cost_collision

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
        vehicles_count=5,
        features=["x", "y", "vx", "vy", "heading"],
        features_range=dict(
            x=[-500, 500],
            y=[-4, 12],
            vx=[-10, 30],
            vy=[-5, 5]
            ),
        absolute=False,
        normalize=False
    )
    
}
env = gym.make('highway-v0', config=config)
# pprint(env.config)
# sys.exit(0)

env.reset()
print(type(env.vehicle.position))

path_x = np.linspace(env.vehicle.position[0], env.vehicle.position[0] + 1000, 500)
path_y = np.array([env.vehicle.position[1] for i in path_x])



# print(f"path_x: {path_x}")
# print(f"path_y: {path_y}")

# print(type(path_x))
# print(type(path_y))

obs_a = 5.0
obs_b = 1.5

frenet_path = []
Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length = compute_path_parameters(path_x, path_y)
for i in range(len(path_x)):
    
    x_global_init = path_x[i]
    y_global_init = path_y[i]

    v_global_init = 10
    vdot_global_init = 0

    psi_global_init = np.arctan2(Fy_dot[i], Fx_dot[i])
    psidot_global_init = 0

    initial_state =  x_global_init, y_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init

    x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init = global_to_frenet(path_x, path_y, initial_state, arc_vec, Fx_dot, Fy_dot, kappa)
    frenet_path.append([x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init])

frenet_path = np.array(frenet_path)
# print(frenet_path)

frenet_x = frenet_path[:, 0]
frenet_y = frenet_path[:, 1]

# print(path_x[0], path_y[0])
# print(x_global_init, y_global_init)
action = [0.0, 0.0] # env.action_space.sample()
wheel_base = 0.5

for _ in range(100):
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    x_ego_global_init = env.vehicle.position[0]
    y_ego_global_init = env.vehicle.position[1]

    psi_ego_global_init = np.arctan2(Fy_dot[0], Fx_dot[0])

    # x_ego_global_init = 0.0
    # y_ego_global_init = 0.0

    obstacles_x = obs[:, 0][1:]
    obstacles_y = obs[:, 1][1:]

    ego_initial_state =  x_ego_global_init, y_ego_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init

    x_ego_init, y_ego_init, vx_ego_init, vy_ego_init, ax_ego_init, ay_ego_init, psi_ego_init, psi_ego_fin, psidot_ego_init = global_to_frenet(path_x, path_y, ego_initial_state, arc_vec, Fx_dot, Fy_dot, kappa)

    # x_ego_init = 0.0
    # y_ego_init = 0.0

    print("ego vehicle in frenet coordinates:")
    print(x_ego_init, y_ego_init)
    
    # print(obstacles_x)
    # print(obstacles_y)

    obs_x = x_ego_init + obstacles_x
    obs_y = y_ego_init + obstacles_y

    # obs_x = obstacles_x
    # obs_y = obstacles_y

    print("obstacles in frenet coordinates:")
    print(obs_x)
    print(obs_y)

    print()

    num = 20
    t = np.linspace(0, 1, num).reshape(num,1)

    P, P_v, P_a = pol_matrix_comp(t)
    nvar = np.shape(P)[1]

    x_0 = x_ego_init
    xdot_0 = frenet_path[0][2]
    xddot_0 = frenet_path[0][4]

    y_0 = y_ego_init
    ydot_0 = frenet_path[0][3]
    yddot_0 = frenet_path[0][5]

    xdot_f = 10.0
    ydot_f = 0.0

    xddot_f = 0.0
    yddot_f = 0.0

    x_f, y_f = create_points(x_ego_init, y_ego_init)
    x_f = np.array(x_f)
    y_f = np.array(y_f)

    # plt.figure()
    # plt.scatter(x_f, y_f)
    # plt.show()

    # print(type(x_f))
    # print(type(y_f))

    # print(x_f)
    # print(y_f)

    w_1 = 2.0
    # w_2 = 20.0
    w_2 = 1.0

    v_max = 10.0
    v_min = -10.0

    a_max = 10.0
    a_min = -10.0

    fig, ax = plt.subplots(figsize=(10, 6))

    x_matrix = np.zeros((len(x_f), t.shape[0]))
    vx_matrix = np.zeros((len(x_f), t.shape[0]))
    ax_matrix = np.zeros((len(x_f), t.shape[0]))
    jx_matrix = np.zeros((len(x_f), t.shape[0]-1))

    y_matrix = np.zeros((len(y_f), t.shape[0]))
    vy_matrix = np.zeros((len(y_f), t.shape[0]))
    ay_matrix = np.zeros((len(y_f), t.shape[0]))
    jy_matrix = np.zeros((len(y_f), t.shape[0]-1))

    ##### CVXPY block goes here
    # start = time.time()
    for i in range(len(x_f)):
        xi = cp.Variable(2*nvar)
        xi_x = xi[0:nvar]
        xi_y = xi[nvar:2*nvar]

        x = P@xi_x
        y = P@xi_y

        xdot = P_v @ xi_x
        ydot = P_v @ xi_y

        xddot = P_a @ xi_x
        yddot = P_a @ xi_y

        cost = w_2*(cp.sum_squares(x[-1]-x_f[i])+cp.sum_squares(y[-1]-y_f[i])+cp.sum_squares(xdot[-1]-xdot_f)+cp.sum_squares(ydot[-1]-ydot_f)+cp.sum_squares(xddot[-1]-xddot_f)+cp.sum_squares(yddot[-1]-yddot_f))
        constraints = [x[0]==x_0, xdot[0]==xdot_0, xddot[0]==xddot_0, y[0]==y_0, ydot[0]==ydot_0, yddot[0]==yddot_0, xdot[1:]<=v_max, -xdot[1:]<=-v_min, ydot[1:]<=v_max, -ydot[1:]<=-v_min, xddot[1:]<=a_max, -xddot[1:]<=-a_min, yddot[1:]<=a_max, -yddot[1:]<=-a_min]

        prob = cp.Problem(cp.Minimize(cost), constraints)

        prob.solve(solver='ECOS')

        sol = xi.value

        xi_x = sol[0:nvar]
        xi_y = sol[nvar:2*nvar]

        x = P@xi_x
        y = P@xi_y

        xdot = P_v@xi_x
        ydot = P_v@xi_y

        xddot = P_a @ xi_x
        yddot = P_a @ xi_y  

        x_matrix[i] = x
        vx_matrix[i] = xdot
        ax_matrix[i] = xddot
        jx_matrix[i] = xi_x[3:]**2

        y_matrix[i] = y
        vy_matrix[i] = ydot
        ay_matrix[i] = yddot
        jy_matrix[i] = xi_y[3:]**2

    #   ax.plot(x, y, color="black", linewidth=2)

    # plt.grid()
    # plt.axis('equal')

    # plt.scatter(x_f, y_f, linewidth=5, color="black")
    # plt.scatter(x_0, y_0)

    # t = np.linspace(0, 2*np.pi, 100)
    # for i in range(len(obs_x)):
    #   plt.plot(obs_x[i]+obs_a*np.cos(t) , obs_y[i]+obs_b*np.sin(t) , "--", color="red")

    cost_collision = compute_cumutative_collision_cost(x_matrix, y_matrix, obs_x, obs_y, obs_a, obs_b)
    # print(cost_collision)
    # print(cost_collision.shape)

    jx = np.sum(jx_matrix, axis=1)
    jy = np.sum(jy_matrix, axis=1)

    cost_smoothness = jx**2 + jy**2
    # print(cost_smoothness)
    # print(cost_smoothness.shape)

    collision_free_indices = []
    for id, cost_coll in enumerate(cost_collision):
      if cost_coll == 0.0:
        collision_free_indices.append(id)
    # print(collision_free_indices)

    collision_free_x = []
    collision_free_y = []

    best_smoothness = cost_smoothness[0]
    best_goal_index = 0

    for i in collision_free_indices:

      collision_free_x.append(x_matrix[i])
      collision_free_y.append(y_matrix[i])

    #   cost_smooth = cost_smoothness[i]
    #   if cost_smooth < best_smoothness:
    #     best_cost_smoothness = cost_smooth
    #     best_goal_index = i

    collision_free_y_sum = []
    for index in collision_free_indices:
      sum_y = np.sum(np.abs(y_matrix[index] - np.zeros(num)))
      collision_free_y_sum.append((index, sum_y))

    sorted_collision_free_y_sum = sorted(collision_free_y_sum, key=lambda x:x[1])

    # print(sorted_collision_free_y_sum)

    sorted_collision_free_y_sum = sorted_collision_free_y_sum[:10]
    # # print(sorted_collision_free_y_sum)
    # # print(collision_free_y)

    best_10_indices = [t[0] for t in sorted_collision_free_y_sum]
    # # print(best_10_indices)

    best_10_x = np.array([x_matrix[i] for i in best_10_indices])
    best_10_y = np.array([y_matrix[i] for i in best_10_indices])

    # # print(best_10_x)
    # # print(best_10_y)

    best_cost_index = None
    best_cost_smoothness = cost_smoothness[sorted_collision_free_y_sum[0][0]]
    # best_cost_smoothness = cost_smoothness[0]
    # best_cost_smoothness = cost_smoothness[sorted_collision_free_y_sum[0]]
    for idx, _ in sorted_collision_free_y_sum:
    # for idx, _ in collision_free_x:
        if cost_smoothness[idx] < best_cost_smoothness:
            best_cost_smoothness = cost_smoothness[idx]
            best_cost_index = idx

    # print(best_cost_index)


    # ax.scatter(collision_free_x, collision_free_y, color="blue", linewidth=0.5)

    # best_cost_index = best_goal_index

    best_x = None
    best_y = None

    best_x = x_matrix[best_cost_index]
    best_y = y_matrix[best_cost_index]

    # print(f"best_x: {best_x}")
    # print(f"best_y: {best_y}")

    best_vx = vx_matrix[best_cost_index]
    best_vy = vy_matrix[best_cost_index]

    # print(f"best_vx: {best_vx}")
    # print(f"best_vy: {best_vy}")

    best_ax = ax_matrix[best_cost_index]
    best_ay = ay_matrix[best_cost_index]

    # print(f"best_ax: {best_ax}")
    # print(f"best_ay: {best_ay}")

    # curvature = (best_ay * best_vx - best_vy * best_ax)/(((best_vx)**2 - (best_vy)**2)**1.5)
    curvature = (best_ay * best_vx - best_vy * best_ax)/(((best_vx)**2 + (best_vy)**2 + 0.0001)**1.5)
    steer = jnp.arctan(curvature * wheel_base)
    vel = ((best_vx)**2 + (best_vy)**2)**0.5
    acc = jnp.diff(vel)

    mean_steer = jnp.mean(steer[:10])
    mean_acc = jnp.mean(acc[:10])

    action = [mean_acc, mean_steer]
    # print(action)
    print("------------------------")
    # ax.plot(best_x, best_y, color="yellow", linewidth=2)

    # plt.show()
    

    # break
    # sleep(1000)