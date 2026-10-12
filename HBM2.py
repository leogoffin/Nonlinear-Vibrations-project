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
        R.cos[:, i] = ( (2.0 / T) * np.trapezoid( Rt * np.cos(h * omega * t_vector), dx=dt, axis=1 ) )
        R.sin[:, i] = ( (2.0 / T) * np.trapezoid( Rt * np.sin(h * omega * t_vector), dx=dt, axis=1 ) ) 

    return R.flatten()

def run_sequential_continuation(omega_span, F, H, N):

    x_current = np.zeros(2*(2*H+1))
    omega_sol = []
    amp1 = []
    amp2 = []

    for omega in omega_span:

        sol = root(aft_residual,x_current,args=(omega, F, H, N),method="lm")
        if not sol.success:
            print(f"Convergence failed at omega = {omega:.2f} rad/s")
            break

        x_current = sol.x.copy() # next initial guess is last solution -> we follow one branch
        X = unflatten(x_current, H)

        omega_sol.append(omega) # different from omega_span -> there is not the omegas without solutions
        amp1.append(np.sqrt(X.cos[0, 0]**2 + X.sin[0, 0]**2))
        amp2.append(np.sqrt(X.cos[1, 0]**2 + X.sin[1, 0]**2))

    return omega_sol,amp1,amp2

H = 1
N = 20

omega_span = np.linspace(50,250, 500)
F = 100.0

omega_sol_up,amp1_up,amp2_up = run_sequential_continuation(omega_span, F, H, N)
omega_sol_down,amp1_down,amp2_down = run_sequential_continuation(omega_span[::-1], F, H, N)

plt.plot(omega_sol_up, amp1_up, label="DOF 1 — upward")
plt.plot(omega_sol_down, amp1_down, "--", label="DOF 1 — downward")

plt.xlabel("Excitation frequency (rad/s)")
plt.ylabel("Fundamental amplitude")
plt.legend()
plt.savefig(f"figures/NLFR_dof1.pdf", bbox_inches="tight")
plt.show()

plt.plot(omega_sol_up, amp2_up, label="DOF 2 — upward")
plt.plot(omega_sol_down, amp2_down, "--", label="DOF 2 — downward")

plt.xlabel("Excitation frequency (rad/s)")
plt.ylabel("Fundamental amplitude")
plt.legend()
plt.savefig(f"figures/NLFR_dof2.pdf", format='pdf', bbox_inches="tight")
plt.show()