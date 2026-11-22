import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import eigh
from itertools import combinations_with_replacement
from pathlib import Path
import h5py
import os

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



