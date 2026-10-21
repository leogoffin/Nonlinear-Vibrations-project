"""Harmonic balance + pseudo-arclength continuation for periodic responses.

The equation is written as
    M q_ddot + C q_dot + K q + f_nl(q, q_dot) = f_ext(omega, t)

Fourier coefficients follow the convention
    q(t) = c0/2 + sum_k [s_k sin(k omega t) + c_k cos(k omega t)]
and are stored as [c0, s1, c1, ..., s_NH, c_NH], with a DOF vector
for each coefficient block.
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def _fourier_basis(theta, n_harmonics):
    """Return Phi for q = Phi @ coefficients, using c0/2 as the first term."""
    n_samples = len(theta)
    n_coeff = 2 * n_harmonics + 1
    Phi = np.zeros((n_samples, n_coeff))
    Phi[:, 0] = 0.5

    for k in range(1, n_harmonics + 1):
        Phi[:, 2 * k - 1] = np.sin(k * theta)
        Phi[:, 2 * k] = np.cos(k * theta)

    return Phi


def _derivative_matrix(omega, n_harmonics):
    """Coefficient-space matrix D such that q_dot = Phi @ D @ Z."""
    n_coeff = 2 * n_harmonics + 1
    D = np.zeros((n_coeff, n_coeff))

    for k in range(1, n_harmonics + 1):
        i = 2 * k - 1
        D[i:i + 2, i:i + 2] = np.array([[0.0, -k * omega],
                                         [k * omega, 0.0]])

    return D


def _as_time_matrix(values, n_samples, n_dof, name):
    """Convert scalar, constant DOF vector, or sampled force to (n_samples, n_dof)."""
    values = np.asarray(values, dtype=float)

    if values.ndim == 0:
        return np.full((n_samples, n_dof), float(values))

    if values.ndim == 1:
        if n_dof == 1 and values.size == n_samples:
            return values.reshape(n_samples, 1)
        if values.size == n_dof:
            return np.tile(values.reshape(1, n_dof), (n_samples, 1))

    if values.shape == (n_samples, n_dof):
        return values

    raise ValueError(
        f"{name} must be a scalar, a constant vector of length {n_dof}, "
        f"or an array with shape ({n_samples}, {n_dof}); got {values.shape}."
    )


def hbm_pseudo_arclength(
    M,
    C,
    K,
    f_ext,
    omega0,
    f_nl=None,
    z0=None,
    n_harmonics=3,
    n_samples=None,
    dw_initial=None,
    n_steps=200,
    ds=0.02,
    ds_min=1e-4,
    ds_max=0.06,
    z_scale=None,
    omega_scale=None,
    force_scale=None,
    tol=1e-9,
    max_retries=8,
    adaptive_step=True,
):
    """Trace a nonlinear frequency-response branch with HBM + pseudo-arclength.

    Parameters
    ----------
    M, C, K : (n_dof, n_dof) arrays
        Mass, damping, and linear stiffness matrices.
    f_ext : callable
        External force callback: f_ext(omega, t_samples). For each frequency,
        return shape (n_samples, n_dof), or a compatible scalar/vector.
        The time samples span one period: t_samples = theta / omega.
    omega0 : float
        Positive starting excitation frequency in rad/s.
    f_nl : callable, optional
        Internal nonlinear force callback f_nl(q, q_dot), with q and q_dot
        shaped (n_samples, n_dof). It must return the same shape, or a
        compatible scalar/vector. Use positive internal restoring forces on
        the left-hand side of the equation of motion. Defaults to zero.
    z0 : array, optional
        Initial guess for the Fourier coefficients. Shape (2*n_harmonics+1,
        n_dof) or a flattened vector. If omitted, a linear HBM solution is used
        as the first guess. The first two branch points are solved at fixed
        frequencies to initialise the secant tangent.
    n_harmonics : int
        Number of retained harmonics.
    n_samples : int, optional
        Number of AFT time samples. Defaults to a power of two with at least
        about eight samples per Fourier coefficient.
    dw_initial : float, optional
        Frequency increment for the second initial point, in rad/s. Defaults
        to 0.5% of omega0 (at least 1e-4 rad/s).
    n_steps : int
        Number of pseudo-arclength points to compute after the two seed points.
    ds : float
        Initial arclength step in scaled coordinates. Tune this for resolution.
    ds_min, ds_max : float
        Minimum and maximum adaptive arclength step.
    z_scale : float or array, optional
        Characteristic scale for Fourier coefficients. If omitted, inferred
        from the initial linear solution (or z0). A custom array may be flat
        or shaped like the coefficient matrix.
    omega_scale : float, optional
        Characteristic frequency scale for continuation. Defaults to
        max(abs(omega0), 1).
    force_scale : float, optional
        Residual scale. If omitted, inferred from the external-force Fourier
        coefficients at omega0.
    tol : float
        Nonlinear solver tolerance.
    max_retries : int
        Number of step halvings allowed after a failed corrector.
    adaptive_step : bool
        Increase/decrease the step based on corrector effort.

    Returns
    -------
    result : dict
        Keys: 'omega', 'frequency_hz', 'coefficients', 'amplitude',
        'scaled_step', and 'corrector_nfev'. Coefficients have shape
        (n_points, 2*n_harmonics+1, n_dof). 'amplitude' is the first-harmonic
        amplitude sqrt(s1**2 + c1**2), shape (n_points, n_dof).
    """
    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")
    n_dof = M.shape[0]
    if C.shape != M.shape or K.shape != M.shape:
        raise ValueError("M, C, and K must have the same square shape.")
    if omega0 <= 0:
        raise ValueError("omega0 must be positive (rad/s).")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1.")
    if ds <= 0 or ds_min <= 0 or ds_max < ds_min:
        raise ValueError("Require ds > 0 and 0 < ds_min <= ds_max.")

    n_coeff = 2 * n_harmonics + 1
    n_z = n_coeff * n_dof

    if n_samples is None:
        n_required = max(64, 8 * n_coeff)
        n_samples = 2 ** int(np.ceil(np.log2(n_required)))
    if n_samples <= 2 * n_harmonics:
        raise ValueError("n_samples must be greater than 2*n_harmonics.")

    theta = 2.0 * np.pi * np.arange(n_samples) / n_samples
    Phi = _fourier_basis(theta, n_harmonics)
    Phi_pinv = np.linalg.pinv(Phi)
    identity_coeff = np.eye(n_coeff)

    if f_nl is None:
        def f_nl(q, q_dot):
            return np.zeros_like(q)

    def external_samples(omega):
        # Avoid division by zero during a Newton trial near omega = 0.
        omega_for_time = omega if abs(omega) > 1e-10 else np.copysign(1e-10, omega if omega != 0 else 1.0)
        t_samples = theta / omega_for_time
        return _as_time_matrix(f_ext(omega, t_samples), n_samples, n_dof, "f_ext")

    def nonlinear_samples(q, q_dot):
        return _as_time_matrix(f_nl(q, q_dot), n_samples, n_dof, "f_nl")

    def hbm_residual(z, omega):
        Z = np.asarray(z, dtype=float).reshape(n_coeff, n_dof)
        D = _derivative_matrix(omega, n_harmonics)
        D2 = D @ D

        q = Phi @ Z
        q_dot = Phi @ D @ Z
        q_ddot = Phi @ D2 @ Z

        R = (q_ddot @ M.T + q_dot @ C.T + q @ K.T
             + nonlinear_samples(q, q_dot) - external_samples(omega))
        return (Phi_pinv @ R).ravel()

    def linear_initial_guess(omega):
        D = _derivative_matrix(omega, n_harmonics)
        D2 = D @ D
        A = (np.kron(D2, M) + np.kron(D, C)
             + np.kron(identity_coeff, K))
        b = (Phi_pinv @ external_samples(omega)).ravel()
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def unpack_z(values):
        values = np.asarray(values, dtype=float)
        if values.shape == (n_coeff, n_dof):
            return values.ravel()
        if values.size == n_z:
            return values.reshape(n_z)
        raise ValueError(
            f"z0 must contain {n_z} coefficients, arranged as "
            f"({n_coeff}, {n_dof}) or a flat vector."
        )

    z_linear0 = linear_initial_guess(omega0)
    z_guess0 = z_linear0 if z0 is None else unpack_z(z0)

    if z_scale is None:
        z_characteristic = max(float(np.max(np.abs(z_guess0))), 1e-8)
        z_scale_vector = np.full(n_z, z_characteristic)
    else:
        z_scale_array = np.asarray(z_scale, dtype=float)
        if z_scale_array.ndim == 0:
            z_scale_vector = np.full(n_z, float(z_scale_array))
        elif z_scale_array.size == n_z:
            z_scale_vector = z_scale_array.reshape(n_z)
        else:
            raise ValueError(f"z_scale must be scalar or contain {n_z} values.")
        if np.any(z_scale_vector <= 0):
            raise ValueError("All z_scale values must be positive.")

    if omega_scale is None:
        omega_scale = max(abs(omega0), 1.0)
    if omega_scale <= 0:
        raise ValueError("omega_scale must be positive.")

    if force_scale is None:
        b0 = (Phi_pinv @ external_samples(omega0)).ravel()
        force_scale = max(float(np.max(np.abs(b0))), 1e-12)
    if force_scale <= 0:
        raise ValueError("force_scale must be positive.")

    def solve_fixed_frequency(omega, z_guess):
        def residual_scaled(y):
            return hbm_residual(y * z_scale_vector, omega) / force_scale

        solution = least_squares(
            residual_scaled, z_guess / z_scale_vector,
            xtol=tol, ftol=tol, gtol=tol, x_scale="jac", max_nfev=5000,
        )
        residual_norm = np.linalg.norm(residual_scaled(solution.x), ord=np.inf)
        if (not solution.success) and residual_norm > max(100.0 * tol, 1e-8):
            raise RuntimeError(
                f"HBM solve failed at omega={omega:.8g} rad/s: "
                f"{solution.message}; scaled residual={residual_norm:.3e}."
            )
        if not np.isfinite(residual_norm) or residual_norm > max(100.0 * tol, 1e-8):
            raise RuntimeError(
                f"HBM residual too large at omega={omega:.8g} rad/s: "
                f"{residual_norm:.3e}."
            )
        return solution.x * z_scale_vector

    if dw_initial is None:
        dw_initial = max(0.005 * omega0, 1e-4)
    omega1 = omega0 + dw_initial
    if omega1 <= 0:
        raise ValueError("omega0 + dw_initial must remain positive.")

    z_first = solve_fixed_frequency(omega0, z_guess0)
    z_second_guess = z_first
    z_second = solve_fixed_frequency(omega1, z_second_guess)

    # Continue in scaled coordinates y = [z / z_scale, omega / omega_scale].
    scale = np.concatenate((z_scale_vector, [float(omega_scale)]))
    y_previous = np.concatenate((z_first, [omega0])) / scale
    y_current = np.concatenate((z_second, [omega1])) / scale

    y_points = [y_previous.copy(), y_current.copy()]
    corrector_nfev = [0, 0]
    used_steps = [0.0, 0.0]
    step = float(np.clip(ds, ds_min, ds_max))

    for _ in range(n_steps):
        tangent = y_current - y_previous
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            raise RuntimeError("Continuation tangent vanished; the two seed points coincide.")
        tangent = tangent / tangent_norm

        accepted = False
        trial_step = step

        for _retry in range(max_retries + 1):
            y_predictor = y_current + trial_step * tangent

            def augmented_residual(y):
                z = y[:-1] * scale[:-1]
                omega = y[-1] * scale[-1]
                h = hbm_residual(z, omega) / force_scale
                arc_constraint = np.dot(tangent, y - y_predictor)
                return np.concatenate((h, [arc_constraint]))

            solution = least_squares(
                augmented_residual, y_predictor,
                xtol=tol, ftol=tol, gtol=tol, x_scale="jac", max_nfev=5000,
            )
            residual_norm = np.linalg.norm(augmented_residual(solution.x), ord=np.inf)
            omega_new = solution.x[-1] * scale[-1]
            converged = (solution.success or residual_norm <= max(100.0 * tol, 1e-8))
            converged = converged and np.isfinite(residual_norm)
            converged = converged and residual_norm <= max(100.0 * tol, 1e-8)
            converged = converged and omega_new > 0.0

            if converged:
                accepted = True
                break

            trial_step *= 0.5
            if trial_step < ds_min:
                break

        if not accepted:
            raise RuntimeError(
                "Pseudo-arclength corrector failed after step-size reductions. "
                f"Last attempted scaled step={trial_step:.3e}. Try a smaller ds, "
                "larger n_samples, or a better initial point."
            )

        y_new = solution.x
        y_points.append(y_new.copy())
        corrector_nfev.append(int(solution.nfev))
        used_steps.append(float(trial_step))

        y_previous = y_current
        y_current = y_new

        if adaptive_step:
            if solution.nfev <= 7:
                step = min(1.2 * trial_step, ds_max)
            elif solution.nfev >= 15:
                step = max(0.7 * trial_step, ds_min)
            else:
                step = trial_step
        else:
            step = float(np.clip(ds, ds_min, ds_max))

    y_points = np.asarray(y_points)
    x_points = y_points * scale
    z_points = x_points[:, :-1]
    omega_points = x_points[:, -1]
    coefficient_points = z_points.reshape(-1, n_coeff, n_dof)
    first_harmonic_amplitude = np.hypot(
        coefficient_points[:, 1, :], coefficient_points[:, 2, :]
    )

    return {
        "omega": omega_points,
        "frequency_hz": omega_points / (2.0 * np.pi),
        "coefficients": coefficient_points,
        "amplitude": first_harmonic_amplitude,
        "scaled_step": np.asarray(used_steps),
        "corrector_nfev": np.asarray(corrector_nfev),
    }


def plot_nlfr(result, dof=0):
    """Plot first-harmonic displacement amplitude versus excitation frequency."""
    plt.plot(result["frequency_hz"], result["amplitude"][:, dof])
    plt.xlabel("Excitation frequency (Hz)")
    plt.ylabel(f"First-harmonic amplitude, DOF {dof} (displacement units)")
    plt.xlim(min(result["frequency_hz"]),max(result["frequency_hz"]))
    plt.grid(True)


if __name__ == "__main__":
    # Demonstration: forced Duffing oscillator. Replace these functions and
    # matrices with the model of interest.
    M = np.array([[1.0]])
    C = np.array([[0.05]])
    K = np.array([[1.0]])
    force_amplitude = 0.5
    cubic_stiffness = 1.0

    def f_ext(omega, t):
        return force_amplitude * np.cos(omega * t)

    def f_nl(q, q_dot):
        return cubic_stiffness * q**3

    result = hbm_pseudo_arclength(
        M, C, K, f_ext, omega0=0.6, f_nl=f_nl,
        n_harmonics=5, dw_initial=0.002, n_steps=250,
        ds=0.015, ds_min=1e-4, ds_max=0.03,
    )

    plot_nlfr(result)
    plt.tight_layout()
    plt.show()
