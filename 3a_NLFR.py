import numpy as np
import matplotlib.pyplot as plt
from hbm_pseudo_arclength import hbm_pseudo_arclength, plot_nlfr

M = np.array([[1, 0],
              [0, 1]])

C = np.array([[3, -1],
              [-1, 3]])

K = np.array([[2e4, -1e4],
              [-1e4, 2e4]])


F0 = 100.0
k4 = 6.7e5
k3 = 3.3e6

"""
k4 = -5.2e7
k3 = -2.6e6
c4 = 1e5
c3 = 1e5
"""

def f_ext(omega, t):
    F = np.zeros((len(t), 2))
    F[:, 0] = F0 * np.cos(omega * t)
    return F

def f_nl(q, q_dot):
    F = np.zeros_like(q)
    F[:, 0] = k4 * (q[:, 0])**4 
    F[:, 1] = k3 * (q[:, 1])**3

    #F[:, 0] = (k4 * (q[:, 0]-q[:,1])**4 + k3 * (q[:, 0]-q[:,1])**3 
    #          +c3 * (q[:, 0]-q[:,1])**3 + c4 * (q[:, 0]-q[:,1])**4)
    #F[:, 1] = - F[:,0]
    return F

result = hbm_pseudo_arclength(
    M, C, K,
    f_ext=f_ext,
    f_nl=f_nl,
    omega0=30.0,
    omegamax = 200,
    n_harmonics=5,
    dw_initial=0.5,
    n_steps=4000,
    ds=0.02,
    ds_min=1e-4,
    ds_max=0.06)

plot_nlfr(result, dof=0)
plt.show()