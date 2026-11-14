import numpy as np
import matplotlib.pyplot as plt
from hbm_pseudo_arclength import hbm_pseudo_arclength, plot_nlfr
from scipy.optimize import least_squares

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

#--------------------------------
# NNM computation (3b)
#--------------------------------
def compute_nnm_backbone(mode, A_values, NH=5, n_samples=128):
    theta = 2*np.pi*np.arange(n_samples)/n_samples

    Phi = np.zeros((n_samples, 2*NH + 1))
    Phi[:, 0] = 0.5
    for k in range(1, NH + 1):
        Phi[:, 2*k - 1] = np.sin(k*theta)
        Phi[:, 2*k]     = np.cos(k*theta)

    Phi_pinv = np.linalg.pinv(Phi)

    def f_nl_nnm(q):
        F = np.zeros_like(q)
        F[:, 0] = k4 * q[:, 0]**4
        F[:, 1] = k3 * q[:, 1]**3
        return F

    def nnm_residual(y, A, omega_linear):
        Z = y[:-1].reshape(2*NH + 1, 2) * A
        omega = y[-1] * omega_linear

        D2 = np.zeros((2*NH + 1, 2*NH + 1))
        for k in range(1, NH + 1):
            D2[2*k - 1, 2*k - 1] = -(k*omega)**2
            D2[2*k,     2*k    ] = -(k*omega)**2

        q = Phi @ Z
        q_ddot = Phi @ D2 @ Z

        R = q_ddot @ M.T + q @ K.T + f_nl_nnm(q)
        hbm = (Phi_pinv @ R).ravel() / (2e4 * A)

        # phase condition + amplitude condition on DOF 1
        phase = Z[1, 0] / A
        amplitude = Z[2, 0] / A - 1.0

        return np.concatenate((hbm, [phase, amplitude]))

    eigenvalues, eigenvectors = np.linalg.eigh(K)

    omega_linear = np.sqrt(eigenvalues[mode])
    shape = eigenvectors[:, mode] / eigenvectors[0, mode]

    previous = None
    frequencies = []
    amplitudes = []

    for i in range(len(A_values)):
        A = A_values[i]

        if previous is None:
            Z_guess = np.zeros((2*NH + 1, 2))
            Z_guess[2] = A * shape
            omega_guess = omega_linear
        else:
            Z_guess = previous[0]
            omega_guess = previous[1]

        y_guess = np.concatenate((Z_guess.ravel()/A,
                                  [omega_guess/omega_linear]))

        solution = least_squares(
            nnm_residual,
            y_guess,
            args=(A, omega_linear),
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            max_nfev=5000
        )

        res = nnm_residual(solution.x, A, omega_linear)
        error = np.max(np.abs(res))

        if not np.isfinite(error) or error > 1e-6:
            print("Mode", mode+1, "stopped at A =", A, "residual =", error)
            break

        Z = solution.x[:-1].reshape(2*NH + 1, 2) * A
        omega = solution.x[-1] * omega_linear
        previous = (Z, omega)

        q = Phi @ Z

        amp = (np.max(q[:, 0]) - np.min(q[:, 0])) / 2.0
        freq = omega / (2*np.pi)

        amplitudes.append(amp)
        frequencies.append(freq)

    return np.array(frequencies), np.array(amplitudes)

# Choose amplitude range for NNM
A_values = np.linspace(0.0005, 0.225, 200)

freq_nnm1, amp_nnm1 = compute_nnm_backbone(0, A_values, NH=5, n_samples=128)
freq_nnm2, amp_nnm2 = compute_nnm_backbone(1, A_values, NH=5, n_samples=128)

#--------------------------------
# Superposition plot (3c)
#--------------------------------
plt.figure()

plt.plot(result["frequency_hz"], result["amplitude"][:, 0], label="NLFR", linewidth=2)
plt.plot(freq_nnm1, amp_nnm1, "--", label="NNM 1")
plt.plot(freq_nnm2, amp_nnm2, "--", label="NNM 2")

plt.xlabel("Frequency (Hz)")
plt.ylabel("DOF 1 displacement amplitude")
plt.xlim(min(result["frequency_hz"]), max(result["frequency_hz"]))
plt.legend()
plt.show()