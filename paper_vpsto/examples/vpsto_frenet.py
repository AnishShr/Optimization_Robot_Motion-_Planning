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

from vpsto.vpsto import VPSTO, VPSTOOptions, VPSTOSolution


# Creating multiple goal positions
def create_points(x_0, y_0, k):
    points_x = []
    points_y = []

    for i in range(k):
        points_x.append((x_0+10) + i*10)
        points_y.append(0)

    for i in range(k):
        for j in range(k):
            points_x.append((x_0+10) + j*10)
            points_y.append((y_0-30) + i*10)
    
    return points_x, points_y

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

num = 100
t = np.linspace(0, 10, num).reshape(num, 1)

P, P_v, P_a = pol_matrix_comp(t)
nvar = np.shape(P)[1]

x_0 = 0.0
xdot_0 = 10.0
xddot_0 = 0.0

y_0 = 0.0
ydot_0 = 0.0
yddot_0 = 0.0

xdot_f = 10.0
ydot_f = 0.0

xddot_f = 0.0
yddot_f = 0.0

x_f, y_f = create_points(x_0, y_0, 8)

plt.figure()
plt.scatter(x_0, y_0, color="red")
plt.scatter(x_f, y_f)
plt.xlim([-20, 100])
plt.grid()
plt.show()




