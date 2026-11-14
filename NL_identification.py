"""Nonlinear vibrations project — Exercise 2a/2b/2c.

Workflow:
  1. Load the experimental q, q_dot, q_ddot and external forces.
  2. Characterize DOF 1, DOF 2 and the relative coordinate (q1-q2).
  3. MANUALLY choose physically plausible nonlinear terms in SELECTED_TERMS.
  4. Estimate one shared coupling force and two optional local forces.
  5. Check fitted force traces, restoring-force surfaces and validation errors.

Model convention:
  M*q_ddot + C*q_dot + K*q + F_nl = F_ext
  F_nl,1 = local_1(q1,qd1) + coupling(q1-q2,qd1-qd2)
  F_nl,2 = local_2(q2,qd2) - coupling(q1-q2,qd1-qd2)

Run once with RUN_ESTIMATION=False to inspect the plots. Then set
SELECTED_TERMS based on their appearance and run the estimation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# ============================================================================
# SETTINGS — edit these
# ============================================================================
DATA_DIR = r"C:\Users\leogo\OneDrive\Bureau\Master 2\Nonlinear Vibrations\signals"
FILE_TYPE = "mat"                      # "csv" or "mat"
MAT_FILENAME = "G7_exp1_3.mat"
FIGURE_DIR = os.path.join(DATA_DIR, "figures_characterization")

STEP = 50                             # Decimate AFTER loading; 1 = keep all samples
MAX_PLOT_POINTS = 12000               # Scatter-plot limit, independent of fitting
SPECTROGRAM_ENABLED = True            # Only meaningful for sine sweeps
SPECTROGRAM_WINDOW_SECONDS = 0.5      # Change to resolve relevant frequencies
RUN_ESTIMATION = True
TRAIN_FRACTION = 0.75                 # First 75% train, last 25% validate

# "spring" / "damper" exponents 2 through 7, choose terms after inspecting plots.
# Locations: "mass1", "mass2", "between".
# Kinds: "q", "v", "q_abs", "v_abs".
# q_abs means q*abs(q)**(n-1), useful for sign-symmetric odd restoring forces
# at even n; v_abs similarly for dissipative drag-like models.
# Plain q/v powers include even powers for asymmetric nonlinearities.
SELECTED_TERMS = [
    ("mass2", "q", 3),
    ("mass1", "q", 4)
]

M = np.eye(2)
C = np.array([[3.0, -1.0], [-1.0, 3.0]])
K = np.array([[2e4, -1e4], [-1e4, 2e4]])


# ============================================================================
# DATA LOADING
# ============================================================================
def csv_signal(filename, column):
    path = os.path.join(DATA_DIR, filename + ".csv")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    if column not in data.columns:
        raise ValueError(f"{path}: missing '{column}'; found {list(data.columns)}")
    return data[column].to_numpy(dtype=float)


def load_data():
    if FILE_TYPE == "csv":
        t = csv_signal("dof1_q", "Time (s)")
        q1 = csv_signal("dof1_q", "(m)")
        q2 = csv_signal("dof2_q", "(m)")
        v1 = csv_signal("dof1_qdot", "(m/s)")
        v2 = csv_signal("dof2_qdot", "(m/s)")
        a1 = csv_signal("dof1_qdotdot", "(m/s2)")
        a2 = csv_signal("dof2_qdotdot", "(m/s2)")
        f1 = csv_signal("dof1_force", "Force (N)")
        f2 = np.zeros_like(f1)  # Change if DOF 2 is also externally excited
        q = np.column_stack((q1, q2))
        v = np.column_stack((v1, v2))
        a = np.column_stack((a1, a2))
        force = np.column_stack((f1, f2))
    elif FILE_TYPE == "mat":
        # Uses the user's existing reader for the simulated experiments.
        from lookmat import loadmat
        t, q, v, a, force = loadmat(os.path.join(DATA_DIR, MAT_FILENAME))
        t = np.asarray(t, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        a = np.asarray(a, dtype=float)
        force = np.asarray(force, dtype=float)
        if force.ndim == 1 or (force.ndim == 2 and 1 in force.shape):
            f1 = force.reshape(-1)
            force = np.column_stack((f1, np.zeros_like(f1)))
    else:
        raise ValueError("FILE_TYPE must be csv or mat")

    t = np.asarray(t, dtype=float).reshape(-1)
    if q.shape != (len(t), 2) or v.shape != q.shape or a.shape != q.shape:
        raise ValueError("q, q_dot, q_ddot must each have shape (N, 2)")
    if force.shape != q.shape:
        raise ValueError("external forces must have shape (N, 2)")
    if any(not np.all(np.isfinite(arr)) for arr in (t, q, v, a, force)):
        raise ValueError("Input data contain NaN or infinite samples")
    dt = np.diff(t)
    if not np.allclose(dt, np.median(dt), rtol=1e-3, atol=1e-10):
        print("WARNING: timestamps are not uniformly spaced; spectrogram may be invalid")
    return t, q, v, a, force


def nonlinear_force(q, v, a, force):
    return force - a @ M.T - v @ C.T - q @ K.T


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, name + ".png"), dpi=180)
    plt.close()


def plot_indices(n):
    return np.arange(0, n, max(1, int(np.ceil(n / MAX_PLOT_POINTS))))


# ============================================================================
# 2a / 2b : CHARACTERIZATION
# ============================================================================
def characterize(t, q, v, a, force, fnl):
    print("\n=== 2a: NONLINEAR FORCE LEVELS ===")
    for i in range(2):
        print(f"DOF {i+1}: RMS nonlinear force = {rms(fnl[:, i]):.5g} N")
    print("RMS(Fnl1 + Fnl2) =", rms(fnl[:, 0] + fnl[:, 1]), "N")
    print("RMS(Fnl1 - Fnl2) =", rms(fnl[:, 0] - fnl[:, 1]), "N")
    print("A small Fnl1+Fnl2 compared with Fnl1-Fnl2 supports internal coupling.")

    qr = q[:, 0] - q[:, 1]
    vr = v[:, 0] - v[:, 1]
    ar = a[:, 0] - a[:, 1]
    # This is the antisymmetric component of nonlinear force, not a
    # guarantee that it is purely the coupling force if local terms exist.
    fr = 0.5 * (fnl[:, 0] - fnl[:, 1])

    xs = [q[:, 0], q[:, 1], qr]
    vs = [v[:, 0], v[:, 1], vr]
    accs = [a[:, 0], a[:, 1], ar]
    fs = [fnl[:, 0], fnl[:, 1], fr]
    names = ["DOF1", "DOF2", "relative"]
    pi = plot_indices(len(t))

    for j in range(3):
        x, vel, acc, f = xs[j], vs[j], accs[j], fs[j]
        name = names[j]
        plt.figure(figsize=(9, 4))
        plt.plot(t, x)
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        savefig("time_displacement_" + name)

        plt.figure(figsize=(9, 4))
        plt.plot(t, f)
        plt.xlabel("Time (s)")
        plt.ylabel("Nonlinear force (N)")
        savefig("time_nonlinear_force_" + name)

        plt.figure()
        plt.scatter(x[pi], f[pi], s=2)
        plt.xlabel("Displacement (m)")
        plt.ylabel("Nonlinear force (N)")
        savefig("force_vs_displacement_" + name)

        plt.figure()
        plt.scatter(vel[pi], f[pi], s=2)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Nonlinear force (N)")
        savefig("force_vs_velocity_" + name)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[pi], vel[pi], acc[pi], s=2)
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_zlabel("Acceleration (m/s²)")
        savefig("ASM_" + name)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[pi], vel[pi], f[pi], s=2)
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_zlabel("Nonlinear force (N)")
        savefig("RFS_" + name)

    if SPECTROGRAM_ENABLED:
        dt = np.median(np.diff(t))
        fsample = 1.0 / dt
        nperseg = min(len(t), max(64, int(round(SPECTROGRAM_WINDOW_SECONDS * fsample))))
        if len(t) >= 64 and nperseg >= 64:
            noverlap = min(nperseg - 1, int(0.85*nperseg))
            for name, signal in zip(names, xs):
                f, times, mag = spectrogram(signal, fs=fsample,
                    nperseg=nperseg, noverlap=noverlap,
                    detrend="constant", scaling="spectrum", mode="magnitude")
                db = 20*np.log10(np.maximum(mag, 1e-15))
                plt.figure(figsize=(9, 5))
                plt.pcolormesh(t[0] + times, f, db, shading="auto", cmap="viridis")
                plt.ylim(0, min(100, fsample/2))
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.colorbar(label="Magnitude (dB, relative to 1 m)")
                savefig("spectrogram_" + name)
        else:
            print("Not enough samples for a spectrogram")

    # Useful signature of an equal-and-opposite nonlinear element.
    plt.figure(figsize=(6, 5))
    plt.scatter(fnl[pi, 0], fnl[pi, 1], s=2)
    lo = float(np.min(fnl[pi]))
    hi = float(np.max(fnl[pi]))
    plt.plot([lo, hi], [-lo, -hi], "k--", label="Fnl2 = -Fnl1")
    plt.xlabel("Nonlinear force DOF 1 (N)")
    plt.ylabel("Nonlinear force DOF 2 (N)")
    plt.legend()
    savefig("force_coupling_check")


# ============================================================================
# 2c : PHYSICALLY SELECTED BASIS — a single global least-squares fit
# ============================================================================
def candidate_signal(q, v, location, kind, degree):
    if location == "mass1":
        x, vel = q[:, 0], v[:, 0]
    elif location == "mass2":
        x, vel = q[:, 1], v[:, 1]
    elif location == "between":
        x, vel = q[:, 0]-q[:, 1], v[:, 0]-v[:, 1]
    else:
        raise ValueError("Unknown location: " + str(location))
    if kind == "q":
        return x**degree
    if kind == "v":
        return vel**degree
    if kind == "q_abs":
        return x * np.abs(x)**(degree-1)
    if kind == "v_abs":
        return vel * np.abs(vel)**(degree-1)
    raise ValueError("Unknown kind: " + str(kind))


def build_joint_matrix(q, v, terms):
    """Stack DOF 1 and DOF 2; enforce Newton's third law on 'between'."""
    n = len(q)
    X = np.zeros((2*n, len(terms)))
    for j, (location, kind, degree) in enumerate(terms):
        #if not isinstance(degree, (int, np.integer)) or degree < 2 or degree > 7:
            #raise ValueError("Degree must be an integer between 2 and 7")
        signal = candidate_signal(q, v, location, kind, degree)
        if location == "mass1":
            X[:n, j] = signal
        elif location == "mass2":
            X[n:, j] = signal
        else:
            X[:n, j] = signal
            X[n:, j] = -signal
    return X


def fit_model(q, v, fnl, terms, train_fraction=0.75):
    if not terms:
        print("No terms selected. Set SELECTED_TERMS after inspecting characterization.")
        return None
    if len(set(terms)) != len(terms):
        raise ValueError("Duplicate terms in SELECTED_TERMS")
    n = len(q)
    n_train = int(n*train_fraction)
    if n_train < 10 or n-n_train < 10:
        raise ValueError("Insufficient training or validation samples")

    X = build_joint_matrix(q, v, terms)
    target = np.concatenate((fnl[:, 0], fnl[:, 1]))
    ix_train = np.concatenate((np.arange(n_train), n+np.arange(n_train)))
    ix_test = np.concatenate((np.arange(n_train, n), n+np.arange(n_train, n)))

    # Scale each column to avoid poor conditioning simply because powers
    # of small displacements / velocities have wildly different units.
    X_train = X[ix_train]
    y_train = target[ix_train]
    scales = np.sqrt(np.mean(X_train**2, axis=0))
    if np.any(scales <= 1e-30):
        raise ValueError("A selected term vanishes on training data")
    normalized = X_train / scales
    beta, _, rank, singular = np.linalg.lstsq(normalized, y_train, rcond=None)
    coefficients = beta / scales
    prediction = X @ coefficients
    pred_force = np.column_stack((prediction[:n], prediction[n:]))

    print("\n=== 2c: COEFFICIENT ESTIMATION ===")
    for term, coeff in zip(terms, coefficients):
        print(f"{term!s:32s}: {coeff:+.3e} (SI units)")
    print(f"Selected terms: {len(terms)}; numerical rank: {rank}")
    if singular[-1] > 0:
        print(f"Normalized matrix condition number: {singular[0]/singular[-1]:.3e}")
    for label, ix in (("TRAIN", ix_train), ("VALIDATION", ix_test)):
        error = target[ix] - prediction[ix]
        rel = 100 * rms(error) / max(rms(target[ix]), 1e-30)
        print(f"{label}: RMSE={rms(error):.6e} N, relative RMSE={rel:.5f}%")
    for dof in range(2):
        for label, indices in (("train", slice(None, n_train)),
                               ("validation", slice(n_train, None))):
            e = fnl[indices, dof]-pred_force[indices, dof]
            denom = max(rms(fnl[indices, dof]), 1e-30)
            print(f"DOF {dof+1} {label}: RMSE={rms(e):.5e} N ",
                  f"({100*rms(e)/denom:.5f}%)")
    return pred_force, coefficients


def plot_validation(t, q, v, a, force, fnl, pred_force):
    n = len(t)
    names = ["DOF1", "DOF2", "relative"]
    xs = [q[:, 0], q[:, 1], q[:, 0]-q[:, 1]]
    vs = [v[:, 0], v[:, 1], v[:, 0]-v[:, 1]]
    actual = [fnl[:, 0], fnl[:, 1], 0.5*(fnl[:, 0]-fnl[:, 1])]
    predicted = [pred_force[:, 0], pred_force[:, 1],
                 0.5*(pred_force[:, 0]-pred_force[:, 1])]
    pi = plot_indices(n)

    # Full restoring force, as recommended by the exercise:
    # measured = F_ext - M*q_ddot
    # model = C*q_dot + K*q + identified F_nl
    total_measured = force - a @ M.T
    total_model = v @ C.T + q @ K.T + pred_force
    for dof in range(2):
        fig = plt.figure(figsize=(11, 5))
        left = fig.add_subplot(121, projection="3d")
        right = fig.add_subplot(122, projection="3d")
        for ax, z, title in ((left, total_measured[:, dof], "Experimental restoring force"),
                             (right, total_model[:, dof], "Model restoring force")):
            ax.scatter(q[pi, dof], v[pi, dof], z[pi], s=2)
            ax.set_xlabel("Displacement (m)")
            ax.set_ylabel("Velocity (m/s)")
            ax.set_zlabel("Total restoring force (N)")
            ax.set_title(title)
        savefig(f"total_restoring_surface_DOF{dof+1}")

    for name, x, vel, measured, fitted in zip(names, xs, vs, actual, predicted):
        plt.figure(figsize=(10, 4))
        plt.plot(t, measured, label="Measured nonlinear force", linewidth=1)
        plt.plot(t, fitted, "--", label="Identified model", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Nonlinear force (N)")
        plt.legend()
        savefig("fit_time_" + name)

        plt.figure(figsize=(7, 5))
        plt.scatter(measured[pi], fitted[pi], s=3)
        lo = float(min(np.min(measured[pi]), np.min(fitted[pi])))
        hi = float(max(np.max(measured[pi]), np.max(fitted[pi])))
        plt.plot([lo, hi], [lo, hi], "k--")
        plt.xlabel("Measured nonlinear force (N)")
        plt.ylabel("Predicted nonlinear force (N)")
        savefig("fit_parity_" + name)

        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        for ax, f, title in ((ax1, measured, "Experiment"),
                              (ax2, fitted, "Model")):
            ax.scatter(x[pi], vel[pi], f[pi], s=2)
            ax.set_xlabel("Displacement (m)")
            ax.set_ylabel("Velocity (m/s)")
            ax.set_zlabel("Nonlinear force (N)")
            ax.set_title(title)
        savefig("RFS_experiment_vs_model_" + name)


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    t_full, q_full, v_full, a_full, force_full = load_data()
    print(f"Original samples: {len(t_full)}, dt={np.median(np.diff(t_full)):.6g} s")
    step = int(STEP)
    if step < 1:
        raise ValueError("STEP must be >= 1")
    t, q, v, a, force = (arr[::step] for arr in
                         (t_full, q_full, v_full, a_full, force_full))
    print(f"Samples after decimation: {len(t)}")
    fs_downsampled = 1.0 / np.median(np.diff(t))
    print(f"Downsampled Nyquist frequency: {fs_downsampled/2:.2f} Hz")
    if fs_downsampled/2 < 100:
        print("WARNING: decimation may hide higher-frequency harmonics in spectrograms")
    print("NOTE: supplied accelerations are used directly, not differentiated again")
    fnl = nonlinear_force(q, v, a, force)
    qr = q[:, 0] - q[:, 1]
    vr = v[:, 0] - v[:, 1]

    F_rel = (fnl[:, 0] - fnl[:, 1]) / 2

    F_remaining = F_rel 

    plt.figure()
    plt.scatter(vr, F_remaining, s=4)
    plt.xlabel("Relative velocity (m/s)")
    plt.ylabel("Remaining nonlinear force (N)")
    plt.show()
    characterize(t, q, v, a, force, fnl)
    if RUN_ESTIMATION:
        fitted = fit_model(q, v, fnl, SELECTED_TERMS, TRAIN_FRACTION)
        if fitted is not None:
            pred_force, coefficients = fitted
            plot_validation(t, q, v, a, force, fnl, pred_force)
    print("Figures written to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
