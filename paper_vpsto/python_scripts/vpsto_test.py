import numpy as np
from vpsto.vpsto import VPSTO, VPSTOOptions, VPSTOSolution
import matplotlib.pyplot as plt

obs_x = [40, 60]
obs_y = [0, -2]

obs_a = 3
obs_b = 0.5

x_init, y_init = [20, 0]
x_final, y_final = [100, 0]

x_lim = [0, 150]
y_lim = [-8, 8]

plt.figure(figsize=(12, 8))

t = np.linspace(0, 2*np.pi, 100)
for i in range(len(obs_x)):
  plt.plot(obs_x[i]+obs_a*np.cos(t) , obs_y[i]+obs_b*np.sin(t) , "--", color="black")

plt.plot(x_init+obs_a*np.cos(t) , y_init+obs_b*np.sin(t) , "--", color="red")
plt.scatter(x_final, y_final)

plt.xlim([-5, 110])
plt.ylim([-10, 10])

plt.grid()
plt.show()

def compute_cumutative_collision_cost(x, y, obs_x, obs_y, obs_a, obs_b):
    obs_x = np.array(obs_x).reshape(len(obs_x), 1)
    obs_y = np.array(obs_y).reshape(len(obs_y), 1)

    # cost_collision = -((x - obs_x[:, np.newaxis])/((2*obs_a)))**2 - ((y - obs_y[:, np.newaxis])/(obs_b))**2 + 1
    cost_collision = np.maximum(np.zeros(x.shape), -((x - obs_x[:, np.newaxis])/((2*obs_a)))**2 - ((y - obs_y[:, np.newaxis])/(2*obs_b))**2 + 1) 
    

    # print(cost_collision.shape)
    cost_collision_combined = np.sum(cost_collision, axis=0)
    # print(cost_collision_combined.shape)
    
    final_cost_collision = cost_collision_combined[0]    

    return final_cost_collision

# q_min = 0.0*np.ones(2)
# q_max = 0.5*np.ones(2)
q_min = np.array([0, -10])
q_max = np.array([10000, 10])

def loss_limits(candidates):
    q = candidates['pos']
    d_min = np.maximum(np.zeros_like(q), - q + q_min)
    d_max = np.maximum(np.zeros_like(q), q - q_max)
#     return np.sum(d_min > 0.0, axis=(1,2)) + np.sum(d_max > 0.0, axis=(1,2))
    return np.sum(d_min, axis=(1,2)) + np.sum(d_max, axis=(1,2)) 

# env = CollisionEnvironment()
# def loss_collision(candidates): 
#     costs = []
#     for traj in candidates['pos']:
#         costs.append(env.getTrajDist(traj))
#     costs = np.array(costs)
#     costs += costs > 0.0
#     return costs

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

def loss_path_offset(candidates):
    costs = []
    for traj in candidates['pos']:
        cost = np.sum(traj[:, 1]**2)
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

def loss(candidates):
    cost_curvature = loss_curvature(candidates)
    print(cost_curvature.shape)
#     cost_collision = loss_collision(candidates)
    cost_collision = collision_loss(candidates)
    print(cost_collision.shape)
    cost_limits = loss_limits(candidates)
    print(cost_limits.shape)
    cost_path_offset = loss_path_offset(candidates)
    print(cost_path_offset)
    return candidates['T'] + \
           1e-2 * cost_curvature + \
           1e3 * cost_collision + \
           1e3 * cost_limits
        #    1e-20 * cost_path_offset

opt = VPSTOOptions(ndof=2)
opt.N_via = 3
opt.N_eval = 100
opt.pop_size = 100
opt.log = True
opt.vel_lim = np.array([40.0, 5.0])
opt.acc_lim = np.array([20.0, 5.0])

traj_opt = VPSTO(opt)
traj_opt.ndof = 2

x_init = 20.0
y_init = 0.0

x_final = 100.0
y_final = 0.0

q0 = np.array([x_init, y_init])
qd = np.array([x_final, y_final])

dq0 = np.array([10.0, 0.0])
dqT = np.array([10.0, 0.0])

sol = traj_opt.minimize(loss, q0=q0, qT=qd, dq0=dq0, dqT=dqT)

t_traj = np.linspace(0, sol.T_best, 1000)
# pos, vel, acc = sol.get_trajectory(t_traj)
pos, vel, acc = sol.get_posvelacc(t_traj)

print('Movement duration: ', sol.T_best)

plt.figure(figsize=(12,8))

# plt.xlim([q_min[0], q_max[0]])
# plt.ylim([q_min[1], q_max[1]])
# ax = plt.axes()

plt.scatter([q0[0], qd[0]],[q0[1], qd[1]])
# for pol in env.poly_list:
#     ax.add_patch(patches.Polygon(pol, facecolor = 'gray'))
for i in range(len(obs_x)):
  plt.plot(obs_x[i]+obs_a*np.cos(t) , obs_y[i]+obs_b*np.sin(t) , "--", color="black")

plt.plot(x_init+obs_a*np.cos(t) , y_init+obs_b*np.sin(t) , "--", color="red")
plt.scatter(x_final, y_final)

plt.plot(pos[:,0], pos[:,1])

plt.xlim([-5, 110])
plt.ylim([-10, 10])
plt.grid()
plt.show()
# plt.tight_layout()