import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

M = np.eye(2)
K = np.array([[2e4, -1e4], [-1e4, 2e4]])

k4 = 6.7e5
k3 = 3.3e6

NH = 5
n_samples = 128
theta = 2 * np.pi * np.arange(n_samples) / n_samples
Phi = np.zeros((n_samples, 2 * NH + 1))
Phi[:, 0] = 0.5
for k in range(1, NH + 1):
    Phi[:, 2*k - 1] = np.sin(k * theta)
    Phi[:, 2*k] = np.cos(k * theta)
Phi_pinv = np.linalg.pinv(Phi)


def f_nl(q):
    F = np.zeros_like(q)
    F[:, 0] = k4 * q[:, 0]**4
    F[:, 1] = k3 * q[:, 1]**3
    return F


def nnm_residual(y, A, omega_linear):
    Z = y[:-1].reshape(2 * NH + 1, 2) * A
    omega = y[-1] * omega_linear
    D2 = np.zeros((2 * NH + 1, 2 * NH + 1))
    for k in range(1, NH + 1):
        D2[2*k - 1, 2*k - 1] = -(k * omega)**2
        D2[2*k, 2*k] = -(k * omega)**2

    q = Phi @ Z
    q_ddot = Phi @ D2 @ Z
    R = q_ddot @ M.T + q @ K.T + f_nl(q)
    hbm = (Phi_pinv @ R).ravel() / (2e4 * A)

    phase = Z[1, 0] / A
    amplitude = Z[2, 0] / A - 1
    return np.concatenate((hbm, [phase, amplitude]))


eigenvalues, eigenvectors = np.linalg.eigh(K)
A_values = np.linspace(0.0005, 0.3, 50)

plt.figure()
for mode in range(2):
    omega_linear = np.sqrt(eigenvalues[mode])
    shape = eigenvectors[:, mode] / eigenvectors[0, mode]
    previous = None
    frequencies = []
    amplitudes = []

    for A in A_values:
        if previous is None:
            Z_guess = np.zeros((2 * NH + 1, 2))
            Z_guess[2] = A * shape
            omega_guess = omega_linear
        else:
            Z_guess, omega_guess = previous

        y_guess = np.concatenate((Z_guess.ravel() / A,
                                  [omega_guess / omega_linear]))
        solution = least_squares(nnm_residual, y_guess,
                                 args=(A, omega_linear),
                                 xtol=1e-10, ftol=1e-10, gtol=1e-10)
        error = np.max(np.abs(nnm_residual(solution.x, A, omega_linear)))
        if not np.isfinite(error) or error > 1e-7:
            print(f"Mode {mode + 1}: stopped at A={A:.5f}, residual={error:.2e}")
            break

        Z = solution.x[:-1].reshape(2 * NH + 1, 2) * A
        omega = solution.x[-1] * omega_linear
        previous = (Z, omega)
        q = Phi @ Z
        amplitudes.append((np.max(q[:, 0]) - np.min(q[:, 0])) / 2)
        frequencies.append(omega / (2 * np.pi))

    plt.plot(frequencies, amplitudes, label=f"NNM {mode + 1}")
    print(f"Mode {mode + 1}: {len(frequencies)} converged points")

plt.xlabel("Natural frequency (Hz)")
plt.ylabel("DOF 1 displacement amplitude")
plt.legend()
plt.show()
