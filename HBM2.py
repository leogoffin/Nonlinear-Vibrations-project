import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

#==============
# EoM matrices
#--------------
M = np.array([[1, 0],
              [0, 1]])

C = np.array([[3, -1],
              [-1, 3]])

K = np.array([[2e4, -1e4],
              [-1e4, 2e4]])

#==============

# NonLinear part of the system - to be found
def f_nl(q):
    alpha = 1e6
    order = 3
    return np.array([alpha * q[0]**order, 0.0])

class FourierCoefs:
    def __init__(self, H):

        self.H = H
        self.mean = np.zeros(2)  # [Q01, Q02]
        self.cos = np.zeros((2, H))  # [[a11,a12,...,a1H],[a21,a22,...,a2H]]
        self.sin = np.zeros((2, H))  # [[b11,b12,...,b1H],[b21,b22,...,b2H]]

    def flatten(self):
        X_flat = [self.mean]

        for i in range(self.H):
            X_flat.append(self.cos[:, i])
            X_flat.append(self.sin[:, i])

        # [Q01,Q02,a11,a21,b11,b21,a12,a22,b12,b22,...,a1H,a2H,b1H,b2H]] #format que Manu utilise depuis le bébut
        return np.concatenate(X_flat) # 2 * (1+2*H)

def unflatten(x_flat, H):
    #full chat j'ai confiance
    X = FourierCoefs(H)
    X.mean = x_flat[0:2]

    for i in range(H):
        start = 2 + 4 * i
        X.cos[:, i] = x_flat[start : start + 2]
        X.sin[:, i] = x_flat[start + 2 : start + 4]

    return X

def kinematics(X,t,omega):

    q = X.mean.copy() # copy of [Q01, Q02]
    qdot = np.zeros(2) # [0,0]
    qdotdot = np.zeros(2) # [0,0]

    for i in range(X.H):
        h = i+1 #harmonic

        a = X.cos[:,i] # [a1i,a2i]
        b = X.sin[:,i] # [b1i,b2i]

        q += a * np.cos(h*omega*t) + b * np.sin(h*omega*t) #displacement (2x1)
        qdot += h*omega*(-a * np.sin(h*omega*t) + b * np.cos(h*omega*t)) #velocity (2x1)
        qdotdot += -(h*omega)**2 * (a * np.cos(h*omega*t) + b * np.sin(h*omega*t)) #acceleration (2x1)

    return q, qdot , qdotdot


def hbm_residuals(X,t, omega, F_ext):
    """
        Demo tablette de Manu 
        - j'utilise direct qdot et qdotdot -> mathematiquement la même
    """

    q,qdot,qdotdot = kinematics(X,t,omega)  # linear kinematics
    F_nl = f_nl(q)
    R_t = M @ qdotdot + C @ qdot + K @ q + F_nl - F_ext # residual at time t (2 x 1)

    return R_t

def aft_residual(x_flat, omega, F,H, N):

    X = unflatten(x_flat, H)
    H = X.H
    T = 2 * np.pi / omega
    t_vector = np.linspace(0, T, N, endpoint=False)
    dt = t_vector[1] - t_vector[0]


    Rt = np.zeros((2, N)) #residuals (2 x #timesteps)
    for j, t in enumerate(t_vector):
        F_ext = np.array([F * np.cos(omega * t), 0.0])
        Rt[:, j] = hbm_residuals(X,t,omega,F_ext) # residual at time t (2 x 1)

    # Alternating Frequency-Time method
    R = FourierCoefs(H)
    R.mean = np.mean(Rt, axis=1)
    for i in range(H):
        h = i + 1
        # https://hal.science/hal-01333697v1/document source avec cette formule mais pas sûr du tout, aussi pas de 2 au début
        R.cos[:, i] = (2.0 / N) * np.sum(Rt * np.cos(h * omega * t_vector)[None, :], axis=1)
        R.sin[:, i] = (2.0 / N) * np.sum(Rt * np.sin(h * omega * t_vector)[None, :], axis=1)

    return R.flatten()

def run_sequential_continuation(omega_span, F, H, N):

    x_current = np.zeros(2*(2*H+1))
    omega_sol = []
    amp1 = []
    amp2 = []

    for omega in omega_span:

        sol = root(aft_residual,x_current,args=(omega, F, H, N),options={"eps": 1e-8},method = "lm")
        if not sol.success or np.max(np.abs(sol.fun)) > 1e-3:
            print(f"Convergence failed at omega = {omega:.2f} rad/s")
            break

        x_current = sol.x.copy() # next initial guess is last solution -> we follow one branch
        X = unflatten(x_current, H)

        omega_sol.append(omega) # different from omega_span -> there is not the omegas without solutions

        amp1.append(np.sqrt((X.cos[0, 0] + X.cos[1, 0])**2 + (X.sin[0, 0] + X.sin[1, 0])**2))
        amp2.append(np.sqrt((X.cos[0, 0] - X.cos[1, 0])**2 + (X.sin[0, 0] - X.sin[1, 0])**2))

    return np.array(omega_sol),amp1,amp2

def full_sweep_routine(omega_span, F, H, N):

    omega_sol, amp1, amp2 = run_sequential_continuation(omega_span, F, H, N)
    fail_id1 = len(omega_sol)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(np.array(omega_sol)/(2*np.pi), amp1, label="Initial sweep")
    ax2.plot(np.array(omega_sol)/(2*np.pi), amp2, label="Initial sweep")

    # First backward sweep
    omega_left1 = omega_sol[::-1]
    omega_solL1, amp1L1, amp2L1 = run_sequential_continuation(omega_left1, F, H, N)

    ax1.plot(np.array(omega_solL1)/(2*np.pi), amp1L1, label="Backward 1")
    ax2.plot(np.array(omega_solL1)/(2*np.pi), amp2L1, label="Backward 1")

    # First restart
    if 0 < fail_id1 < len(omega_span):

        omega_right1 = omega_span[fail_id1 - 1:]

        omega_solR1, amp1R1, amp2R1 = run_sequential_continuation(omega_right1, F, H, N)

        ax1.plot(np.array(omega_solR1)/(2*np.pi), amp1R1, label="Restart 1")
        ax2.plot(np.array(omega_solR1)/(2*np.pi), amp2R1, label="Restart 1")

        # Second break
        fail_id2 = len(omega_solR1)

        if 0 < fail_id2 < len(omega_right1):

            # Second backward sweep
            omega_left2 = omega_solR1[::-1]

            omega_solL2, amp1L2, amp2L2 = run_sequential_continuation(omega_left2, F, H, N)

            ax1.plot(np.array(omega_solL2)/(2*np.pi), amp1L2,label="Backward 2")
            ax2.plot(np.array(omega_solL2)/(2*np.pi), amp2L2,label="Backward 2")

            # Second restart
            omega_right2 = omega_right1[fail_id2 - 1:]
            omega_solR2, amp1R2, amp2R2 = run_sequential_continuation(omega_right2, F, H, N)

            ax1.plot(np.array(omega_solR2)/(2*np.pi), amp1R2,label="Restart 2")
            ax2.plot(np.array(omega_solR2)/(2*np.pi), amp2R2,label="Restart 2")

    for ax, title in zip((ax1, ax2), ("Mode 1", "Mode 2")):
        ax.set_xlabel("Excitation frequency (Hz)")
        ax.set_ylabel("Fundamental amplitude")
        ax.set_title(title)
        ax.set_xlim(omega_span[0]/(2*np.pi),omega_span[-1]/(2*np.pi))
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"figures/NLFR_F{F}_H{H}_N{N}.pdf", bbox_inches="tight")
    plt.show()



H = 3
N = 50

omega_span = np.linspace(50,350, 400)
F = 400.0
full_sweep_routine(omega_span,F,H,N)
