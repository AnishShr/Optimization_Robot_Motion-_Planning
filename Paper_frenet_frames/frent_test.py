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

def create_points(x_frenet, y_frenet, k):

    points_x = []
    points_y = []

    for i in range(k):
      points_x.append((x_frenet+10) + i*10)
      points_y.append(0)
    
    for i in range(k):
        for j in range(k):
            # points_x.append(30 + j*10)
            # points_y.append(-40 + i*10)
            points_x.append((x_frenet+10) + j*10)
            points_y.append((y_frenet-30) + i*10)
    
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

def frenet_to_global(y_frenet, ref_x, ref_y, dx_by_ds, dy_by_ds):
  
    normal_x = -1*dy_by_ds
    normal_y = dx_by_ds

    norm_vec = np.sqrt(normal_x**2 + normal_y**2)
    normal_unit_x = (1/norm_vec)*normal_x
    normal_unit_y = (1/norm_vec)*normal_y

    global_x = ref_x + y_frenet*normal_unit_x
    global_y = ref_y + y_frenet*normal_unit_y

    psi_global = np.unwrap(np.arctan2(np.diff(global_y),np.diff(global_x)))

    return global_x, global_y, psi_global

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


a = 80
b = 60
arc_angles = np.linspace(np.pi*0.9, np.pi/8, 100)
x = a * np.cos(arc_angles)
y = b * np.sin(arc_angles)

obstacle_points = np.array([[-10, 40], [-50, 48]])

plt.figure(figsize=(12, 8))
plt.plot(x, y)

plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], color="red")
plt.grid()
plt.gca().set_aspect('equal')
plt.ylim([0, 80])
plt.show()

Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length = compute_path_parameters(x, y)

x_path = x
y_path = y

frenet_points = []

for i in range(len(x)):

  x_global_init = x[i]
  y_global_init = y[i]

  v_global_init = 10
  vdot_global_init = 0

  psi_global_init = np.arctan2(Fy_dot[i], Fx_dot[i])
  psidot_global_init = 0

  initial_state =  x_global_init, y_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init

  x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init = global_to_frenet(x_path, y_path, initial_state, arc_vec, Fx_dot, Fy_dot, kappa)

  frenet_points.append([x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init])

# x_global_init = x[0]
# y_global_init = x[1]

# initial_state =  x_global_init, y_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init

# x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init = global_to_frenet(x_path, y_path, initial_state, arc_vec, Fx_dot, Fy_dot, kappa)

# print(x_init, y_init)

frenet_points = np.array(frenet_points)
print(frenet_points)
print(frenet_points.shape)

frenet_x = frenet_points[:, 0]
frenet_y = frenet_points[:, 1]

# plt.figure()
# plt.plot(frenet_x, frenet_y)
# plt.show()

curve_points = []
for i in range(len(x)):
  curve_points.append([x[i], y[i]])

curve_points = np.array(curve_points)

obs_frenet = []
# Calculate the distances between the external point and each curve point
for obs in obstacle_points:
  min_distance = cdist([obs], curve_points)

# Find the index of the curve point with the minimum distance
  closest_index = np.argmin(min_distance)

# Retrieve the closest point on the curve
  closest_point = curve_points[closest_index]

  if obs[1] < curve_points[closest_index][1]:
    obs_frenet.append([frenet_points[closest_index][0], frenet_points[closest_index][1]-np.min(min_distance)])
  else:
    obs_frenet.append([frenet_points[closest_index][0], frenet_points[closest_index][1]+np.min(min_distance)])
  

obs_frenet = np.array(obs_frenet)

obs_x = obs_frenet[:, 0]
obs_y = obs_frenet[:, 1]
obs_a = 6
obs_b = 2.5

print(frenet_points)

plt.figure()
plt.plot(frenet_points[:, 0], frenet_points[:, 1])
plt.scatter(obs_frenet[:, 0], obs_frenet[:, 1], color="red")
plt.scatter(x_global_init, y_global_init)

plt.ylim([-20, 5])
plt.show()


num = 100
t = np.linspace(0, 10, num).reshape(num,1)

P, P_v, P_a = pol_matrix_comp(t)
nvar = np.shape(P)[1]

x_0 = frenet_points[0][0]
xdot_0 = frenet_points[0][2]
xddot_0 = frenet_points[0][4]

y_0 = frenet_points[0][1]
ydot_0 = frenet_points[0][3]
yddot_0 = frenet_points[0][5]

xdot_f = 10.0
ydot_f = 0.0

xddot_f = 0.0
yddot_f = 0.0


# plt.figure()
# plt.scatter(x_0, y_0)
# plt.show()

x_f, y_f = create_points(x_0, y_0, 7)

w_1 = 2.0
# w_2 = 20.0
w_2 = 1.0

v_max = 10.0
v_min = -10.0

a_max = 10.0
a_min = -10.0

obs_x = obs_frenet[:, 0]
obs_y = obs_frenet[:, 1]
obs_a = 6
obs_b = 2.5

fig, ax = plt.subplots(figsize=(10, 6))

x_matrix = np.zeros((len(x_f), t.shape[0]))
jx_matrix = np.zeros((len(x_f), t.shape[0]-1))

y_matrix = np.zeros((len(y_f), t.shape[0]))
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
  jx_matrix[i] = xi_x[3:]**2

  y_matrix[i] = y
  jy_matrix[i] = xi_y[3:]**2

  ax.plot(x, y, color="black", linewidth=2)

plt.grid()
plt.axis('equal')

plt.scatter(x_f, y_f, linewidth=5, color="black")
plt.scatter(x_0, y_0)

t = np.linspace(0, 2*np.pi, 100)
for i in range(len(obs_x)):
  plt.plot(obs_x[i]+obs_a*np.cos(t) , obs_y[i]+obs_b*np.sin(t) , "--", color="red")

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

  # cost_smooth = cost_smoothness[i]
  # if cost_smooth < best_smoothness:
  #   best_cost_smoothness = cost_smooth
  #   best_goal_index = i

collision_free_y_sum = []
for index in collision_free_indices:
  sum_y = np.sum(np.abs(y_matrix[index] - np.zeros(100)))
  collision_free_y_sum.append((index, sum_y))

sorted_collision_free_y_sum = sorted(collision_free_y_sum, key=lambda x:x[1])

# print(sorted_collision_free_y_sum)

sorted_collision_free_y_sum = sorted_collision_free_y_sum[:10]
# print(sorted_collision_free_y_sum)
# print(collision_free_y)

best_10_indices = [t[0] for t in sorted_collision_free_y_sum]
# print(best_10_indices)

best_10_x = np.array([x_matrix[i] for i in best_10_indices])
best_10_y = np.array([y_matrix[i] for i in best_10_indices])

# print(best_10_x)
# print(best_10_y)

best_cost_index = None
best_cost_smoothness = cost_smoothness[sorted_collision_free_y_sum[0][0]]

for idx, _ in sorted_collision_free_y_sum:
  if cost_smoothness[idx] < best_cost_smoothness:
    best_cost_smoothness = cost_smoothness[idx]
    best_cost_index = idx

# print(best_cost_index)


ax.scatter(collision_free_x, collision_free_y, color="blue", linewidth=0.5)


best_x = None
best_y = None

best_x = x_matrix[best_cost_index]
best_y = y_matrix[best_cost_index]
ax.plot(best_x, best_y, color="yellow", linewidth=2)

plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_path, y_path)

global_10_best_x = []
global_10_best_y = []

for i in best_10_indices:

    ref_x = jax_interp(x_matrix[i], arc_vec, x_path)
    ref_y = jax_interp(x_matrix[i], arc_vec, y_path)
    dx_by_ds = jax_interp(x_matrix[i], arc_vec, Fx_dot)
    dy_by_ds = jax_interp(x_matrix[i], arc_vec, Fy_dot)

    global_x, global_y, psi_global	= frenet_to_global(y_matrix[i], ref_x, ref_y, dx_by_ds, dy_by_ds)
    global_10_best_x.append(global_x)
    global_10_best_y.append(global_y)


global_10_best_x = np.array(global_10_best_x)
global_10_best_y = np.array(global_10_best_y)

for x, y in zip(global_10_best_x, global_10_best_y):
    # Plot the path
    plt.plot(x, y, color="blue")

ref_x = jax_interp(best_x, arc_vec, x_path)
ref_y = jax_interp(best_x, arc_vec, y_path)
dx_by_ds = jax_interp(best_x, arc_vec, Fx_dot)
dy_by_ds = jax_interp(best_x, arc_vec, Fy_dot)

global_x, global_y, psi_global	= frenet_to_global(best_y, ref_x, ref_y, dx_by_ds, dy_by_ds)

print(global_x[50])
print(global_y[50])

ax.plot(global_x, global_y, color="green", linewidth=3)
ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], color="red")
ax.scatter(global_x[50], global_y[50], color="black")


point, = ax.plot([], [], 'ro')

def init():
   point.set_data([], [])
   return point, 

def update(frame):
   x_points = global_x[frame]
   y_points = global_y[frame]
   point.set_data(x_points, y_points)
   return point, 

animate = animation.FuncAnimation(fig, update, frames=len(global_x), interval=100, init_func=init, blit=True)

ax.set_ylim([0, 100])
ax.grid()
plt.show()




new_init_x = global_x[60]
new_init_y = global_y[60]

initial_state =  new_init_x, new_init_y, v_global_init, vdot_global_init, psi_global_init, psidot_global_init

x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init = global_to_frenet(x_path, y_path, initial_state, arc_vec, Fx_dot, Fy_dot, kappa)

print(x_init, y_init)



num = 100
t = np.linspace(0, 10, num).reshape(num,1)

P, P_v, P_a = pol_matrix_comp(t)
nvar = np.shape(P)[1]

x_0 = x_init
xdot_0 = frenet_points[0][2]
xddot_0 = frenet_points[0][4]

y_0 = y_init
ydot_0 = frenet_points[0][3]
yddot_0 = frenet_points[0][5]

xdot_f = 10.0
ydot_f = 0.0

xddot_f = 0.0
yddot_f = 0.0


# plt.figure()
# plt.scatter(x_0, y_0)
# plt.show()

x_f, y_f = create_points(x_0, y_0, 7)

w_1 = 2.0
# w_2 = 20.0
w_2 = 1.0

v_max = 10.0
v_min = -10.0

a_max = 10.0
a_min = -10.0

obs_x = obs_frenet[:, 0]
obs_y = obs_frenet[:, 1]
obs_a = 6
obs_b = 2.5

fig, ax = plt.subplots(figsize=(10, 6))

x_matrix = np.zeros((len(x_f), t.shape[0]))
jx_matrix = np.zeros((len(x_f), t.shape[0]-1))

y_matrix = np.zeros((len(y_f), t.shape[0]))
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
  jx_matrix[i] = xi_x[3:]**2

  y_matrix[i] = y
  jy_matrix[i] = xi_y[3:]**2

  ax.plot(x, y, color="black", linewidth=2)

plt.grid()
plt.axis('equal')

plt.scatter(x_f, y_f, linewidth=5, color="black")
plt.scatter(x_0, y_0)

t = np.linspace(0, 2*np.pi, 100)
for i in range(len(obs_x)):
  plt.plot(obs_x[i]+obs_a*np.cos(t) , obs_y[i]+obs_b*np.sin(t) , "--", color="red")

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

  # cost_smooth = cost_smoothness[i]
  # if cost_smooth < best_smoothness:
  #   best_cost_smoothness = cost_smooth
  #   best_goal_index = i

collision_free_y_sum = []
for index in collision_free_indices:
  sum_y = np.sum(np.abs(y_matrix[index] - np.zeros(100)))
  collision_free_y_sum.append((index, sum_y))

sorted_collision_free_y_sum = sorted(collision_free_y_sum, key=lambda x:x[1])

# print(sorted_collision_free_y_sum)

sorted_collision_free_y_sum = sorted_collision_free_y_sum[:10]
# print(sorted_collision_free_y_sum)
# print(collision_free_y)

best_10_indices = [t[0] for t in sorted_collision_free_y_sum]
# print(best_10_indices)

best_10_x = np.array([x_matrix[i] for i in best_10_indices])
best_10_y = np.array([y_matrix[i] for i in best_10_indices])

# print(best_10_x)
# print(best_10_y)

best_cost_index = None
best_cost_smoothness = cost_smoothness[sorted_collision_free_y_sum[0][0]]

for idx, _ in sorted_collision_free_y_sum:
  if cost_smoothness[idx] < best_cost_smoothness:
    best_cost_smoothness = cost_smoothness[idx]
    best_cost_index = idx

# print(best_cost_index)


ax.scatter(collision_free_x, collision_free_y, color="blue", linewidth=0.5)


best_x = None
best_y = None

best_x = x_matrix[best_cost_index]
best_y = y_matrix[best_cost_index]
ax.plot(best_x, best_y, color="yellow", linewidth=2)

plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_path, y_path)

global_10_best_x = []
global_10_best_y = []

for i in best_10_indices:

    ref_x = jax_interp(x_matrix[i], arc_vec, x_path)
    ref_y = jax_interp(x_matrix[i], arc_vec, y_path)
    dx_by_ds = jax_interp(x_matrix[i], arc_vec, Fx_dot)
    dy_by_ds = jax_interp(x_matrix[i], arc_vec, Fy_dot)

    global_x, global_y, psi_global	= frenet_to_global(y_matrix[i], ref_x, ref_y, dx_by_ds, dy_by_ds)
    global_10_best_x.append(global_x)
    global_10_best_y.append(global_y)


global_10_best_x = np.array(global_10_best_x)
global_10_best_y = np.array(global_10_best_y)

for x, y in zip(global_10_best_x, global_10_best_y):
    # Plot the path
    plt.plot(x, y, color="blue")

ref_x = jax_interp(best_x, arc_vec, x_path)
ref_y = jax_interp(best_x, arc_vec, y_path)
dx_by_ds = jax_interp(best_x, arc_vec, Fx_dot)
dy_by_ds = jax_interp(best_x, arc_vec, Fy_dot)

global_x, global_y, psi_global	= frenet_to_global(best_y, ref_x, ref_y, dx_by_ds, dy_by_ds)

print(global_x[50])
print(global_y[50])

ax.plot(global_x, global_y, color="green", linewidth=3)
ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], color="red")
ax.scatter(global_x[50], global_y[50], color="black")


point, = ax.plot([], [], 'ro')

def init():
   point.set_data([], [])
   return point, 

def update(frame):
   x_points = global_x[frame]
   y_points = global_y[frame]
   point.set_data(x_points, y_points)
   return point, 

animate = animation.FuncAnimation(fig, update, frames=len(global_x), interval=100, init_func=init, blit=True)

ax.set_ylim([0, 100])
ax.grid()
plt.show()
