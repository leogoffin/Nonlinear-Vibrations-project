import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from mpl_toolkits.mplot3d import Axes3D  
import pandas as pd
import os
from lookmat import loadmat




# q: (Ns, 2), déplacements
# q_dot: (Ns, 2), vitesses
# q_ddot: (Ns, 2), accélérations
# F_ext : (Ns, 2), forces
# t : temps
# ======================================================= PARAMETERS AND MATRICES  ============================================================

M = np.array([[1, 0],
              [0, 1]])

C = np.array([[3, -1],
              [-1, 3]])

K = np.array([[2e4, -1e4],
              [-1e4, 2e4]])

#data_dir = ("C:/Users/emman/OneDrive/Bureau/ULg/MASTER_2/""Nonlinear_vibration/NI2D_test/")
#data_dir = ("C:/Users/leogo/OneDrive/Bureau/Master 2/Nonlinear Vibrations/")
#data_dir = ("C:/Users/emman/OneDrive/Bureau/ULg/MASTER_2/Nonlinear_vibration/NI2D_q1q2")


figure_dir = os.path.join(data_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

# ===========================================================LOAD CSV FILES ============================================================

def load_csv(filename, key):
    path = os.path.join(data_dir, filename + ".csv")
    df = pd.read_csv(path)

    
    df.columns = df.columns.str.strip()

    if key not in df.columns:
        raise ValueError(
            f"Column '{key}' not found in {filename}.csv. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df[key].to_numpy(dtype=np.float64)


t = load_csv("dof1_q", "Time (s)")
pex = load_csv("dof1_force", "Force (N)")
F2 = np.zeros_like(pex)

q1 = load_csv("dof1_q", "(m)")
q2 = load_csv("dof2_q", "(m)")

qd1 = load_csv("dof1_qdot", "(m/s)")
qd2 = load_csv("dof2_qdot", "(m/s)")

qdd1 = load_csv("dof1_qdotdot", "(m/s2)")
qdd2 = load_csv("dof2_qdotdot", "(m/s2)")
"""

file_path = os.path.join(data_dir, "G7_exp1_3.mat")
t, q, qd, qdd, pex = loadmat(file_path)
q1,q2 = q[:,0],q[:,1]
qd1,qd2 = qd[:,0],qd[:,1]
qdd1,qdd2 = qdd[:,0],qdd[:,1]
"""
#one row per time sample
q = np.column_stack((q1, q2))
q_dot = np.column_stack((qd1, qd2))
q_ddot = np.column_stack((qdd1, qdd2))
F_ext = np.column_stack((pex, np.zeros_like(pex)))

signals = {
    "q1": q1,
    "q2": q2,
    "qdot1": qd1,
    "qdot2": qd2,
    "qddot1": qdd1,
    "qddot2": qdd2,
    "F1": pex,
    "t": t
}

for name, signal in signals.items():
    invalid = ~np.isfinite(signal)
    if np.any(invalid):
        print(f"{name}: {invalid.sum()} invalid values")
        print("First invalid indices:", np.where(invalid)[0][:10])

DOF_COLORS = {
    0: "blue",
    1: "red"}

# =====================================DOWNSAMPLING ============================================================

#keep 1 sample every 50
step = 50

t = t[::step]
q = q[::step]
q_dot = q_dot[::step]
q_ddot = q_ddot[::step]
F_ext = F_ext[::step]

print("Number of samples used:", len(t))

# calculate restoring force

def calculate_restoring_force(q_ddot, F_ext, M, C, K, q, q_dot):

    return F_ext - q_ddot @ M.T - q_dot @ C.T - q @ K.T

# Check the individual contributions to the restoring force


F_RFS = calculate_restoring_force(
    q_ddot, F_ext, M, C, K, q, q_dot
)

terms = {
    "External force": F_ext,
    "Inertial force": q_ddot @ M.T,
    "Damping force": q_dot @ C.T,
    "Linear stiffness force": q @ K.T,
    "Calculated RFS": F_RFS
}

print("\n--- Force contribution check ---")

for name, values in terms.items():
    print(f"\n{name}:")
    for dof in range(2):
        rms = np.sqrt(np.mean(values[:, dof]**2))
        minimum = np.min(values[:, dof])
        maximum = np.max(values[:, dof])

        print(
            f"  DOF {dof+1}: "
            f"RMS = {rms:.6e} N, "
            f"min = {minimum:.6e}, "
            f"max = {maximum:.6e}"
        )
# check action-reaction principle

force_sum = F_RFS[:, 0] + F_RFS[:, 1]

rms_force_1 = np.sqrt(np.mean(F_RFS[:, 0]**2))
rms_force_2 = np.sqrt(np.mean(F_RFS[:, 1]**2))
rms_force_sum = np.sqrt(np.mean(force_sum**2))

relative_imbalance = rms_force_sum / np.sqrt(
    rms_force_1**2 + rms_force_2**2
)

print("\n--- Action-reaction check ---")
print(f"RMS force DOF 1: {rms_force_1:.6e} N")
print(f"RMS force DOF 2: {rms_force_2:.6e} N")
print(f"RMS force sum  : {rms_force_sum:.6e} N")
print(f"Relative imbalance: {100*relative_imbalance:.3f} %")

# relative displacement and velocity

delta = q[:, 1] - q[:, 0]
delta_dot = q_dot[:, 1] - q_dot[:, 0]

# extract the internal nonlinear force

F_relative = 0.5 * (F_RFS[:, 1] - F_RFS[:, 0])


def build_relative_basis(delta, delta_dot, max_degree=7):

    variables = np.column_stack((delta, delta_dot))

    variable_names = ["delta", "delta_dot"]

    # normalize variables

    variable_scales = np.sqrt(np.mean(variables**2, axis=0))

    if np.any(variable_scales == 0):
        raise ValueError("A relative variable is identically zero.")

    z = variables / variable_scales

    columns = []
    names = []
    exponents = []

    # generate polynomial terms

    for degree in range(1, max_degree + 1):

        for indices in combinations_with_replacement(range(2), degree):

            powers = np.bincount(indices, minlength=2)

            columns.append(np.prod(z**powers, axis=1))
            exponents.append(powers)

            factors = []

            for name, power in zip(variable_names, powers):

                if power == 1:
                    factors.append(name)

                elif power > 1:
                    factors.append(f"{name}**{power}")

            names.append("*".join(factors))

    X = np.column_stack(columns)

    return X, names, np.asarray(exponents), variable_scales


def identify_relative_rfs(delta, delta_dot, F_relative, max_degree=7):

    X, names, exponents, variable_scales = build_relative_basis(
        delta, delta_dot, max_degree
    )

    y = F_relative

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("NaN or Inf detected in the input data.")

    # normalize candidate columns

    column_scales = np.sqrt(np.mean(X**2, axis=0))

    valid = column_scales > 1e-14

    X = X[:, valid]
    column_scales = column_scales[valid]
    names = [name for name, keep in zip(names, valid) if keep]
    exponents = exponents[valid]

    Xn = X / column_scales

    n, p = Xn.shape

    # full least-squares estimate

    beta_full = np.linalg.lstsq(Xn, y, rcond=None)[0]

    coefficient_scale = np.max(np.abs(beta_full))

    if coefficient_scale == 0:
        raise ValueError("All identified coefficients are zero.")

    thresholds = coefficient_scale * np.concatenate((
        [0.0],
        np.logspace(-5, -0.5, 60)
    ))

    # BIC selection

    best_bic = np.inf
    best_active = None
    best_beta = None
    best_prediction = None

    for threshold in thresholds:

        beta = beta_full.copy()
        active = np.abs(beta) >= threshold

        # iteratively remove weak terms and refit

        for _ in range(20):

            if not np.any(active):
                break

            beta_new = np.zeros(p)

            beta_new[active] = np.linalg.lstsq(
                Xn[:, active], y, rcond=None
            )[0]

            new_active = np.abs(beta_new) >= threshold

            if not np.any(new_active):
                new_active[np.argmax(np.abs(beta_new))] = True

            if np.array_equal(new_active, active):
                beta = beta_new
                break

            active = new_active
            beta = beta_new

        if not np.any(active):
            continue

        # final fit

        beta = np.zeros(p)

        beta[active] = np.linalg.lstsq(
            Xn[:, active], y, rcond=None
        )[0]

        prediction = Xn @ beta
        residual = y - prediction

        rss = np.sum(residual**2)
        n_selected = np.count_nonzero(active)

        bic = (
            n * np.log(max(rss / n, np.finfo(float).tiny))
            + n_selected * np.log(n)
        )

        if bic < best_bic:

            best_bic = bic
            best_active = active.copy()
            best_beta = beta.copy()
            best_prediction = prediction.copy()

    # selected terms

    selected_names = [
        name for name, keep in zip(names, best_active) if keep
    ]

    selected_exponents = exponents[best_active]
    selected_beta = best_beta[best_active]
    selected_scales = column_scales[best_active]

    # convert coefficients to SI units

    coefficients_normalized = selected_beta / selected_scales

    coefficients_SI = np.array([
        coefficient / np.prod(variable_scales**powers)
        for coefficient, powers in zip(
            coefficients_normalized, selected_exponents
        )
    ])

    residual = y - best_prediction

    rmse = np.sqrt(np.mean(residual**2))
    force_rms = np.sqrt(np.mean(y**2))

    return {
        "names": selected_names,
        "exponents": selected_exponents,
        "coefficients": coefficients_SI,
        "prediction": best_prediction,
        "residual": residual,
        "RMSE": rmse,
        "force_RMS": force_rms,
        "BIC": best_bic,
        "n_selected": len(selected_names),
        "n_candidates": Xn.shape[1],
        "rank": np.linalg.matrix_rank(Xn[:, best_active])
    }


def print_relative_results(results):

    print("\n" + "=" * 60)
    print("RELATIVE NONLINEAR FORCE IDENTIFICATION")
    print("=" * 60)

    print(f"Candidate terms : {results['n_candidates']}")
    print(f"Selected terms  : {results['n_selected']}")
    print(f"Selected rank   : {results['rank']}")
    print(f"BIC             : {results['BIC']:.6e}")
    print(f"RMSE            : {results['RMSE']:.6e} N")
    print(f"Force RMS       : {results['force_RMS']:.6e} N")

    if results["force_RMS"] > 0:

        print(
            f"Relative RMSE   : "
            f"{100*results['RMSE']/results['force_RMS']:.4f} %"
        )

    print("\nIdentified terms:")

    for name, coefficient in zip(
        results["names"], results["coefficients"]
    ):

        print(f"{coefficient:+.6e} * {name}")

def plot_relative_results(results, time):

    y = results["prediction"] + results["residual"]
    y_pred = results["prediction"]
    residual = results["residual"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # force reconstruction

    axes[0].plot(time, y, label="Calculated force")
    axes[0].plot(time, y_pred, "--", label="Reconstructed force")

    axes[0].set_ylabel("Force [N]")
    axes[0].legend()
    axes[0].grid(False)

    # residual

    axes[1].plot(time, residual)

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Residual [N]")
    axes[1].grid(False)

    plt.tight_layout()

    save_path = os.path.join(
        figure_dir, "Relative_force_reconstruction.pdf"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
   

def plot_relative_contribution(results, delta, delta_dot):

    exponents = results["exponents"]
    coefficients = results["coefficients"]

    max_delta = int(np.max(exponents[:, 0]))
    max_velocity = int(np.max(exponents[:, 1]))

    contributions = np.zeros((max_velocity + 1, max_delta + 1))

    # calculate RMS contribution of each selected term

    for powers, coefficient in zip(exponents, coefficients):

        term = coefficient * delta**powers[0] * delta_dot**powers[1]

        contribution = np.sqrt(np.mean(term**2))

        contributions[powers[1], powers[0]] = contribution

    # convert to relative percentages

    total_contribution = np.sum(contributions)

    if total_contribution > 0:

        contributions = 100 * contributions / total_contribution

    # plot

    fig, ax = plt.subplots(figsize=(8, 6))

    masked = np.ma.masked_where(contributions == 0, contributions)

    im = ax.imshow(
        masked,
        origin="lower",
        cmap="viridis",
        aspect="equal",
        vmin=0
    )

    # annotate squares

    for i in range(max_velocity + 1):

        for j in range(max_delta + 1):

            if contributions[i, j] > 0:

                ax.text(
                    j, i,
                    f"{contributions[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9
                )

    ax.set_xlabel(r"Exponent of $\delta$")
    ax.set_ylabel(r"Exponent of $\dot{\delta}$")

    ax.set_title("Relative RMS contribution of identified terms")

    fig.colorbar(
        im,
        ax=ax,
        label="Normalized RMS contribution [%]"
    )

    plt.tight_layout()

    save_path = os.path.join(
        figure_dir, "Relative_contribution_matrix.pdf"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
   


# ============================================================
# RELATIVE NONLINEAR FORCE IDENTIFICATION
# ============================================================

results = identify_relative_rfs(
    delta,
    delta_dot,
    F_relative,
    max_degree=7
)

# print identification results

print_relative_results(results)

# plot force reconstruction

plot_relative_results(results, t)

# plot contribution matrix

plot_relative_contribution(
    results,
    delta,
    delta_dot
)

# reconstruct forces on both masses

F_RFS_reconstructed = np.column_stack((
    -results["prediction"],
    results["prediction"]
))

force_error = F_RFS - F_RFS_reconstructed

print("\n--- Full force reconstruction ---")

for dof in range(2):

    rmse = np.sqrt(np.mean(force_error[:, dof]**2))
    force_rms = np.sqrt(np.mean(F_RFS[:, dof]**2))

    print(f"DOF {dof+1}:")
    print(f"  RMSE: {rmse:.6e} N")
    print(f"  Relative RMSE: {100*rmse/force_rms:.4f} %")

plt.show()