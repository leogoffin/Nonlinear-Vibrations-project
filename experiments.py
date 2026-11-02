import numpy as np
import os
from scipy.io import savemat

def sinesweep_experiment(F0,f_start,f_end,sweep_rate,dt = None,filename = "exp"):

    if dt == None : 
        dt = 1/200/max(f_start,f_end)
    print(dt)
    tmax = abs(f_end - f_start) / sweep_rate 
    t = np.arange(0,tmax,dt)

    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * tmax))
    F = F0 * np.sin(phase)


    os.makedirs("mats", exist_ok=True)

    savemat("mats/" + filename + ".mat", {"time": t,"force": F})

    print(f"Succesfully saved the {filename}.mat file with {len(t)} components\n")
    return t,F

"""
Experiment 1 : 
Dof 1 excited 
Linear sweep rate : 6 Hz/min = 0.1 Hz/sec

F = 250 N
Signal 1 : sinesweep  6 Hz -> 40 Hz
Signal 2 : sinesweep 40 Hz ->  6 Hz

F = 20 N
Signal 3 : sinesweep  6 Hz -> 40 Hz
"""
sinesweep_experiment(F0 = 250,f_start=6 ,f_end=40,sweep_rate=0.1,filename="exp1_signal1")
sinesweep_experiment(F0 = 250,f_start=40,f_end=6 ,sweep_rate=0.1,filename="exp1_signal2")
sinesweep_experiment(F0 = 20 ,f_start=6 ,f_end=40,sweep_rate=0.1,filename="exp1_signal3")