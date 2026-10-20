

import numpy as np

#Experiment 1 pour le projet de non linear vibration
#Type: sine sweep,forward puis backward
#forced DOF: 1
#ampli : 250
#freq range: 5 à 40 Hz
#phase: 0
#sweep rate: 5 Hz/min
#sweep style: linear 

#NEWMARK
#time step: 1/6000s

def experiment1():
    F0 = 250
    f_min = 5
    f_max = 40
    sweep_rate = 5 / 60       
    dt = 1 / 6000          
    T_sweep = (f_max - f_min) / sweep_rate
    T_total = 2 * T_sweep #forward puis backward

    t = np.arange(round(T_total / dt) + 1) * dt
    F = np.zeros(len(t))

    #Forward sweep: 5 -> 40 Hz
    forward = np.where(t <= T_sweep)
    tau = t[forward]

    phase = 2 * np.pi * (f_min * tau + sweep_rate * tau**2 / 2)
    F[forward] = F0 * np.sin(phase)

    #Backward sweep: 40 -> 5 Hz
    backward = np.where(t > T_sweep)
    tau = t[backward] - T_sweep

    phase = 2 * np.pi * (f_max * tau - sweep_rate * tau**2 / 2)
    F[backward] = F0 * np.sin(phase)

    data = np.column_stack((t, F))
    np.savetxt("C:/Users/emman/OneDrive/Bureau/ULg/MASTER_2/Nonlinear_vibration/experiment1.csv", data, delimiter=",",header="time_s,force_N", comments="")

    return t, F


t, F = experiment1()
