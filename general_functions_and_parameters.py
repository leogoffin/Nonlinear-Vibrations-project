import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import eigh,expm
from itertools import combinations_with_replacement
from pathlib import Path
import h5py
import os


M = np.array([[1, 0], [0, 1]])
C = np.array([[3, -1], [-1, 3]])
K = np.array([[2e4, -1e4], [-1e4, 2e4]])
k4 = 6.7e5
k3 = 3.3e6

def f_nl(q, q_dot):
    F = np.zeros_like(q)
    F[:, 0] = k4 * q[:, 0]**4
    F[:, 1] = k3 * q[:, 1]**3
    return F

def loadmat(filename):

    with h5py.File(filename, "r") as file:
        t = np.array(file["data/t"]).flatten()
        q = np.array(file["data/x"])
        q_dot = np.array(file["data/xd"])
        q_ddot = np.array(file["data/xdd"])
        pex = np.array(file["data/pex"]).flatten()

    #applied to node 1
    f_ext = np.column_stack((pex, np.zeros_like(pex)))

    return t, q, q_dot, q_ddot, f_ext

def print_signal(t, q, q_dot, q_ddot, f_ext):

    print("Time:", t.shape)
    print("Displacement:", q.shape)
    print("Velocity:", q_dot.shape)
    print("Acceleration:", q_ddot.shape)
    print("External force:", f_ext.shape)

dof_colors = ["blue","red"]

def set_plot_params(font_size):
    plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": font_size})

def setfig(linewidth = 0.7,AR = 1.618):    
    plt.figure(figsize = (8*linewidth/0.7,8/AR*linewidth/0.7))

def savefig(name,figure_dir,format = 'pdf'):
    plt.tight_layout()  
    if format == "png" : plt.savefig(os.path.join(figure_dir, name + ".png"), dpi=180,bbox_inches="tight")
    elif format == "pdf" : plt.savefig(os.path.join(figure_dir, name),format = "pdf",bbox_inches="tight")
    plt.close()

def plot_signal(t, q, q_dot, q_ddot, f_ext, figure_dir): 
   
    plt.figure(figsize=(9, 5))
    plt.plot(t, q[:, 0], label="DOF 1",color = 'blue')
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Displacement [m]", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    savefig("signal_DOF1.pdf",figure_dir)

    plt.figure(figsize=(9, 5))
    plt.plot(t, q[:, 1], label="DOF 2",color = 'red')
    
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Displacement [m]", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18)
    savefig("signal_DOF2.pdf",figure_dir)
     
    return 


def rms(values):

    values = np.asarray(values)     
    squared_values = values**2     
    mean_squared = np.mean(squared_values)
    rms = np.sqrt(mean_squared)      
    rms = float(rms)     

    return rms
