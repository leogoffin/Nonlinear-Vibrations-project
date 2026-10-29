import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import pandas as pd
import os



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

data_dir = ("C:/Users/emman/OneDrive/Bureau/ULg/MASTER_2/""Nonlinear_vibration/NI2D_test/")

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
F1 = load_csv("dof1_force", "Force (N)")
F2 = np.zeros_like(F1)

q1 = load_csv("dof1_q", "(m)")
q2 = load_csv("dof2_q", "(m)")

qdot1 = load_csv("dof1_qdot", "(m/s)")
qdot2 = load_csv("dof2_qdot", "(m/s)")

qddot1 = load_csv("dof1_qdotdot", "(m/s2)")
qddot2 = load_csv("dof2_qdotdot", "(m/s2)")


#one row per time sample
q = np.column_stack((q1, q2))
q_dot = np.column_stack((qdot1, qdot2))
q_ddot = np.column_stack((qddot1, qddot2))
F_ext = np.column_stack((F1, F2))

signals = {
    "q1": q1,
    "q2": q2,
    "qdot1": qdot1,
    "qdot2": qdot2,
    "qddot1": qddot1,
    "qddot2": qddot2,
    "F1": F1,
    "t": t
}

for name, signal in signals.items():
    invalid = ~np.isfinite(signal)
    if np.any(invalid):
        print(f"{name}: {invalid.sum()} invalid values")
        print("First invalid indices:", np.where(invalid)[0][:10])

# ============================================================DOWNSAMPLING ============================================================

#keep 1 sample every 50
step = 50

t = t[::step]
q = q[::step]
q_dot = q_dot[::step]
q_ddot = q_ddot[::step]
F_ext = F_ext[::step]

print("Number of samples used:", len(t))


# ==========================================================Q2a AND Q2b: RESIDUAL FORCES ============================================================

def calculate_residual_forces(q, q_dot, q_ddot, F_ext, M, C, K):
    
    F_nl = (F_ext- q_ddot @ M.T- q_dot @ C.T- q @ K.T)

    return F_nl


def calculate_residual_rms(F_nl):
   
    rms_1 = np.sqrt(np.mean(F_nl[:, 0]**2))
    rms_2 = np.sqrt(np.mean(F_nl[:, 1]**2))

    print("\nResidual force RMS")
    print(f"DOF 1: {rms_1:.6e} N")
    print(f"DOF 2: {rms_2:.6e} N")

    return rms_1, rms_2


def plot_residual_forces(t, F_nl):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax[0].plot(t, F_nl[:, 0])
    ax[0].set_ylabel("Residual force [N]")
    ax[0].set_title("DOF 1")
    ax[0].grid(True)

    ax[1].plot(t, F_nl[:, 1])
    ax[1].set_ylabel("Residual force [N]")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_title("DOF 2")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "Residual_forces.pdf"),format="pdf")
    return 


def plot_asm_surface(q, q_dot, F_nl, dof=0):
    #Acceleration Surface Method (ASM) :residual force versus displacement and velocity.

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(q[:, dof], q_dot[:, dof], F_nl[:, dof],s=8)

    ax.set_xlabel(f"q{dof + 1} [m]")
    ax.set_ylabel(f"q{dof + 1} dot [m/s]")
    ax.set_zlabel("Residual force [N]")
    ax.set_title(f"ASM - DOF {dof + 1}")

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"ASM_DOF{dof + 1}.pdf"),format="pdf")
    return 


def plot_residual_vs_displacement(q, F_nl, dof=0):

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(q[:, dof], F_nl[:, dof], s=8)
    ax.set_xlabel(f"q{dof + 1} [m]")
    ax.set_ylabel("Residual force [N]")
    ax.set_title(f"Residual force versus displacement - DOF {dof + 1}")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join( figure_dir, f"Residual_vs_displacement_DOF{dof + 1}.pdf"),format="pdf")
    return 


F_nl = calculate_residual_forces(q, q_dot, q_ddot, F_ext, M, C, K) 

F_nl = calculate_residual_forces(q, q_dot, q_ddot, F_ext, M, C, K)

# Check the input data and residual force
signals = {
    "t": t,
    "q": q,
    "q_dot": q_dot,
    "q_ddot": q_ddot,
    "F_ext": F_ext,
    "F_nl": F_nl
}

rms_1, rms_2 = calculate_residual_rms(F_nl)
plot_residual_forces(t, F_nl)

#DOF 1
plot_asm_surface(q, q_dot, F_nl, dof=0)
plot_residual_vs_displacement(q, F_nl, dof=0)

#DOF 2
plot_asm_surface(q, q_dot, F_nl, dof=1)
plot_residual_vs_displacement(q, F_nl, dof=1)



def calculate_restoring_force(q_ddot, F_ext, M, C, K, q, q_dot):
   
    return (F_ext- q_ddot @ M.T- q_dot @ C.T - q @ K.T)


def build_rfs_basis(q, q_dot, max_degree=5):

    variables = np.column_stack((
        q[:, 0],
        q[:, 1],
        q_dot[:, 0],
        q_dot[:, 1]
    ))

    variable_names = ["q1", "q2", "q1_dot", "q2_dot"]

    variable_scales = np.sqrt(np.mean(variables**2, axis=0)) #RMS

    if np.any(variable_scales == 0):
        raise ValueError("A displacement or velocity is identically zero.")

    z = variables / variable_scales

    columns = [] #prend les valeurs num
    names = [] #noms (q1, q2,...)
    exponents = [] #exposants des termes

    for degree in range(1, max_degree + 1):
        for indices in combinations_with_replacement(range(4), degree): #range(4) c'est q1,q2,q1dot,q2dot. Genère combinaison en autorisant les repet

            powers = np.bincount(indices, minlength=4) #convertir indices en exposants. compte combien de fois chaque indice apparaît.

            columns.append(np.prod(z**powers, axis=1))
            exponents.append(powers) #renvoie les exposants

            factors = []
            for name, power in zip(variable_names, powers):
                if power == 1:
                    factors.append(name) #ne participent pas à la multiplication
                elif power > 1:
                    factors.append(f"{name}**{power}")

            names.append("*".join(factors)) #assemble les éléments en séparant par une étoile

    X = np.column_stack(columns)

    return X, names, np.asarray(exponents), variable_scales


def identify_rfs(q, q_dot, F_RFS, dof=0, max_degree=5):

    X, names, exponents, variable_scales = build_rfs_basis(q, q_dot, max_degree) #X sont les candidats (q1, q1dot, q2,...)

    y = F_RFS[:, dof]

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("NaN or Inf detected in the input data.")

    #normalize candidate columns
    column_scales = np.sqrt(np.mean(X**2, axis=0)) #RMS
    valid = column_scales > 1e-14

    X = X[:, valid]
    column_scales = column_scales[valid]
    names = [name for name, keep in zip(names, valid) if keep]
    exponents = exponents[valid]

    Xn = X / column_scales

    n, p = Xn.shape #n = nb échantillons, p = nb candidats observés

    #full least-squares estimate
    beta_full = np.linalg.lstsq(Xn, y, rcond=None)[0] #moindres carrés, min l'erreur quadratique: avoir la force la plus proche de la cible

    #Pourquoi ne pas s'arrêter ici ? Parce que plusieurs termes peuvent être fortement corrélés. Par exemple, sur une plage de déplacement limitée, \(q_1^3\) et \(q_1^5\) peuvent présenter des évolutions suffisamment similaires pour que leurs coefficients soient difficiles à distinguer.
    coefficient_scale = np.max(np.abs(beta_full))

    if coefficient_scale == 0:
        raise ValueError("All identified coefficients are zero.")

    thresholds = coefficient_scale * np.concatenate(([0.0],np.logspace(-5, -0.5, 60))) #sueil de sélection

    #J'ai arrêté d'essayer de comprendre ici
    best_bic = np.inf
    best_active = None
    best_beta = None
    best_prediction = None

    for threshold in thresholds:

        beta = beta_full.copy()
        active = np.abs(beta) >= threshold

        # Iteratively remove weak terms and refit
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

        # Final fit for the selected terms
        beta = np.zeros(p)

        beta[active] = np.linalg.lstsq(
            Xn[:, active], y, rcond=None
        )[0]

        prediction = Xn @ beta
        residual = y - prediction

        rss = np.sum(residual**2)
        n_selected = np.count_nonzero(active)

        # Bayesian Information Criterion
        bic = (
            n * np.log(max(rss / n, np.finfo(float).tiny))
            + n_selected * np.log(n)
        )

        if bic < best_bic:
            best_bic = bic
            best_active = active.copy()
            best_beta = beta.copy()
            best_prediction = prediction.copy()

    # Convert selected coefficients to physical SI coefficients
    selected_names = [
        name for name, keep in zip(names, best_active) if keep
    ]

    selected_exponents = exponents[best_active]
    selected_beta = best_beta[best_active]
    selected_scales = column_scales[best_active]

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
        "dof": dof,
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


def print_rfs_results(results):

    print("\n" + "=" * 60)
    print(f"RFS IDENTIFICATION - DOF {results['dof'] + 1}")
    print("=" * 60)

    print(f"Candidate terms : {results['n_candidates']}")
    print(f"Selected terms  : {results['n_selected']}")
    print(f"Selected rank   : {results['rank']}")
    print(f"BIC             : {results['BIC']:.8e}")
    print(f"RMSE            : {results['RMSE']:.8e} N")
    print(f"RMS force       : {results['force_RMS']:.8e} N")

    if results["force_RMS"] > 0:
        print(
            f"Relative RMSE   : "
            f"{100 * results['RMSE'] / results['force_RMS']:.6f} %"
        )


def plot_rfs_results(results, time=None):

    y_pred = results["prediction"]
    residual = results["residual"]
    y = y_pred + residual

    if time is None:
        time = np.arange(len(y))
        xlabel = "Sample"
    else:
        xlabel = "Time [s]"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(time, y, label="Calculated RFS")
    axes[0].plot(time, y_pred, "--", label="Reconstructed RFS")
    axes[0].set_ylabel("Force [N]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, residual)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Residual [N]")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"RFS_DOF{results['dof'] + 1}.pdf",dpi=300,bbox_inches="tight")
    #plt.show()


def print_nonlinear_terms_for_ni2d(results):

    variable_names = ["q1", "q2", "q1_dot", "q2_dot"]
    dof = results["dof"]

    print(f"\nDOF {dof + 1} — NONLINEAR TERMS")
    print("-" * 60)

    terms = []

    for powers, coefficient in zip(results["exponents"],results["coefficients"]):

        #exclude linear terms; retain total degree >= 2
        if np.sum(powers) < 2:
            continue

        factors = []

        for variable, power in zip(variable_names, powers):
            if power == 1:
                factors.append(variable)
            elif power > 1:
                factors.append(f"{variable}**{power}")

        expression = "*".join(factors)
        terms.append((coefficient, expression))

    if not terms:
        print("No nonlinear terms selected.")
        print(f"F_nl[{dof}] = 0")
        return

    print("Nonlinear coefficients to examine:\n")

    for coefficient, expression in terms:
        print(f"{coefficient:+.6e} * {expression}")

    force_expression = " ".join(
        f"{coefficient:+.6e}*{expression}"
        for coefficient, expression in terms
    )

    print(f"\nF_nl[{dof}] = {force_expression}")


F_RFS = calculate_restoring_force(q_ddot, F_ext, M, C, K, q, q_dot)

results_1 = identify_rfs(q, q_dot, F_RFS, dof=0, max_degree=5)
results_2 = identify_rfs(q, q_dot, F_RFS, dof=1, max_degree=5)

print_rfs_results(results_1)
plot_rfs_results(results_1, time=t)
print_nonlinear_terms_for_ni2d(results_1)

print_rfs_results(results_2)
plot_rfs_results(results_2, time=t)
print_nonlinear_terms_for_ni2d(results_2)