import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# System matrices
# --------------------------------------------------

M = np.array([[1, 0],
              [0, 1]])

C = np.array([[3, -1],
              [-1, 3]])

K = np.array([[2e4, -1e4],
              [-1e4, 2e4]])


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_csv(filename, key):
    df = pd.read_csv("NI2D_test/" + filename + ".csv")
    return df[key].to_numpy()


t = load_csv("dof1_q", "Time (s)")

F1 = load_csv("dof1_force", " Force (N)")
F2 = np.zeros_like(F1)

q1 = load_csv("dof1_q", " (m)")
q2 = load_csv("dof2_q", " (m)")

qdot1 = load_csv("dof1_qdot", " (m/s)")
qdot2 = load_csv("dof2_qdot", " (m/s)")

qddot1 = load_csv("dof1_qdotdot", " (m/s2)")
qddot2 = load_csv("dof2_qdotdot", " (m/s2)")


# --------------------------------------------------
# FFT filtering
# --------------------------------------------------

def filter_signal(signal, t, w1=5, w2=500):
    signal_fft = np.fft.fft(signal)
    dt = t[1] - t[0]
    w = 2 * np.pi * np.fft.fftfreq(len(signal), d=dt)

    mask = (np.abs(w) > w1) & (np.abs(w) < w2)
    signal_fft[~mask] = 0

    return np.real(np.fft.ifft(signal_fft))


q = np.column_stack([filter_signal(q1, t),filter_signal(q2, t)])
qdot = np.column_stack([filter_signal(qdot1, t),filter_signal(qdot2, t)])
qddot = np.column_stack([filter_signal(qddot1, t),filter_signal(qddot2, t)])
F = np.column_stack([filter_signal(F1, t),filter_signal(F2, t)])


# --------------------------------------------------
# Calculate nonlinear force residual
# --------------------------------------------------

fnl = F - qddot @ M.T - qdot @ C.T - q @ K.T

print("Nonlinear force residual calculated.")


# --------------------------------------------------
# Build candidate polynomial terms
# --------------------------------------------------

max_order = 5
threshold = 1  # Minimum NRMSE improvement in percentage points

# Candidate monomials q1^a * q2^b.
# Degree 1 terms are excluded because the linear stiffness
# contribution is already accounted for by K.

terms = []

for degree in range(2, max_order + 1):
    for a in range(degree, -1, -1):
        b = degree - a
        terms.append((a, b))


def term_name(term):
    a, b = term

    if a == 0:
        return f"q2^{b}"
    if b == 0:
        return f"q1^{a}"

    return f"q1^{a} q2^{b}"


def evaluate_term(q, term):
    a, b = term
    return q[:, 0]**a * q[:, 1]**b


# Each feature is scaled before fitting to improve numerical conditioning.
features = np.column_stack([evaluate_term(q, term) for term in terms])

feature_scales = np.std(features, axis=0)
feature_scales[feature_scales < 1e-30] = 1.0

features_scaled = features / feature_scales


# --------------------------------------------------
# Sequential term selection, separately for each DOF
# --------------------------------------------------

accepted_terms = [[], []]
accepted_coeff = [None, None]
tested_deltas = [[], []]
current_nrmse = []

for dof in range(2):

    target = fnl[:, dof]
    std_target = np.std(target)

    current_nrmse.append(np.sqrt(np.mean(target**2)) / std_target * 100)

    print(f"\n========== DOF {dof + 1} ==========")

    for j, term in enumerate(terms):

        candidate_indices = accepted_terms[dof] + [j]
        X = features_scaled[:, candidate_indices]

        coeff_scaled, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        prediction = X @ coeff_scaled

        rmse = np.sqrt(np.mean((target - prediction)**2))
        nrmse = rmse / std_target * 100
        delta = current_nrmse[dof] - nrmse

        tested_deltas[dof].append(delta)

        print(f"{term_name(term):12s} " f"NRMSE = {nrmse:.2f}%  " f"delta = {delta:.2f}%")

        if delta >= threshold:
            accepted_terms[dof].append(j)
            accepted_coeff[dof] = coeff_scaled / feature_scales[candidate_indices]
            current_nrmse[dof] = nrmse


    print("\nAccepted terms:")
    for j in accepted_terms[dof]:
        print(" ", term_name(terms[j]))

    print(f"Final NRMSE: {current_nrmse[dof]:.4f}%")


# --------------------------------------------------
# Refit final models
# --------------------------------------------------

predicted_fnl = np.zeros_like(fnl)

for dof in range(2):

    indices = accepted_terms[dof]

    if len(indices) == 0:
        accepted_coeff[dof] = np.array([])
        continue

    X = features_scaled[:, indices]

    coeff_scaled, _, _, _ = np.linalg.lstsq(X, fnl[:, dof], rcond=None)

    accepted_coeff[dof] = coeff_scaled / feature_scales[indices]
    predicted_fnl[:, dof] = X @ coeff_scaled


# --------------------------------------------------
# Plot nonlinear force: measured residual vs fitted model
# --------------------------------------------------

step = max(1, len(t) // 5000)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for dof in range(2):
    axes[dof].plot(t[::step], fnl[::step, dof],label="Nonlinear force residual", alpha=0.7)
    axes[dof].plot(t[::step], predicted_fnl[::step, dof],label="Polynomial model", alpha=0.8)
    axes[dof].set_ylabel(f"DOF {dof + 1} force (N)")
    axes[dof].legend()
    axes[dof].grid(True)

axes[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Plot NRMSE improvement for every candidate
# --------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

for dof in range(2):
    axes[dof].bar(np.arange(len(terms)),tested_deltas[dof])
    axes[dof].axhline(threshold, color="red", linestyle="--",label="0.5 percentage-point threshold")
    axes[dof].set_xticks(np.arange(len(terms)),[term_name(term) for term in terms],rotation=90)
    axes[dof].set_ylabel("NRMSE improvement (percentage points)")
    axes[dof].set_title(f"DOF {dof + 1}")
    axes[dof].legend()
    axes[dof].grid(axis="y")

axes[1].set_xlabel("Candidate polynomial term")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Plot logarithm of absolute accepted coefficients
# --------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

for dof in range(2):
    indices = accepted_terms[dof]

    if len(indices) == 0:
        axes[dof].set_title(f"DOF {dof + 1}: no accepted terms")
        continue

    coeff = accepted_coeff[dof]

    axes[dof].bar([term_name(terms[j]) for j in indices],np.log10(np.maximum(np.abs(coeff), 1e-30)))
    axes[dof].set_ylabel(r"$\log_{10}(|c|)$")
    axes[dof].set_title(f"DOF {dof + 1}")
    axes[dof].grid(axis="y")

axes[1].set_xlabel("Accepted polynomial term")
plt.tight_layout()
plt.show()