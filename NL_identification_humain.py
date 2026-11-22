"""Nonlinear vibrations - Questions 2a, 2b and 2c.

First, inspect the nonlinear forces and their dependence on displacement
and velocity. Then choose the terms to include in the model and estimate
their coefficients using least squares.

Equation: M*q_ddot + C*q_dot + K*q + F_nl = F_ext
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from lookmat import loadmat



############################## SETTINGS #######################################
#-------------------------------------------------------------------------------

#data_dir = r"C:\Users\leogo\OneDrive\Bureau\Master 2\Nonlinear Vibrations\signals"
data_dir = r"C:/Users/emman/OneDrive/Bureau/ULg/MASTER_2/Nonlinear_vibration/NI2D_running/"
figure_dir = os.path.join(data_dir, "figures_characterization")

file_type = "mat"  # "mat" or "csv"
mat_filename = "G7_exp1_3.mat"

step = 50
max_plot_points = 12000
spectrogram_enabled = True #SUPPRIMER ???
spectrogram_window_seconds = 0.5

run_estimation = True
train_fraction = 0.75

#Each term is defined by (location, variable, degree).
#Locations: mass1, mass2, between (relative motion).
#Variables: q (displacement), v (velocity).
#q_abs and v_abs represent x*abs(x)**(degree-1).
#Only the terms are selected here; their coefficients are fitted later.
selected_terms = [
    ("mass2", "q", 3),
    ("mass1", "q", 4),
]

m = np.eye(2)
c = np.array([[3.0, -1.0], [-1.0, 3.0]])
k = np.array([[2e4, -1e4], [-1e4, 2e4]])


############################## LOAD EXPERIMENTAL DATA #######################################
#-------------------------------------------------------------------------------

def read_csv_column(filename, column):
    path = os.path.join(data_dir, filename + ".csv")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data[column].to_numpy(dtype=float)


def load_data():
    if file_type == "mat":
        
        path = os.path.join(data_dir, mat_filename)
        t, q, q_dot, q_ddot, f_ext = loadmat(path)

        t = np.asarray(t).reshape(-1)
        q = np.asarray(q)
        q_dot = np.asarray(q_dot)
        q_ddot = np.asarray(q_ddot)
        f_ext = np.asarray(f_ext)

        #some files only provide the external force on mass 1.
        if f_ext.ndim == 1 or 1 in f_ext.shape:
            f1 = f_ext.reshape(-1)
            f_ext = np.column_stack((f1, np.zeros_like(f1)))

    else:
        t = read_csv_column("dof1_q", "Time (s)")
        q1 = read_csv_column("dof1_q", "(m)")
        q2 = read_csv_column("dof2_q", "(m)")
        v1 = read_csv_column("dof1_qdot", "(m/s)")
        v2 = read_csv_column("dof2_qdot", "(m/s)")
        a1 = read_csv_column("dof1_qdotdot", "(m/s2)")
        a2 = read_csv_column("dof2_qdotdot", "(m/s2)")
        f1 = read_csv_column("dof1_force", "Force (N)")

        q = np.column_stack((q1, q2))
        q_dot = np.column_stack((v1, v2))
        q_ddot = np.column_stack((a1, a2))
        f_ext = np.column_stack((f1, np.zeros_like(f1)))

    return t, q, q_dot, q_ddot, f_ext


############################## HELPFUL FUNCTIONS #######################################
#-------------------------------------------------------------------------------

def rms(values):
    return np.sqrt(np.mean(np.asarray(values)**2))


def save_figure(name):
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, name + ".png"), dpi=180)
    plt.close()


def plot_indices(n_samples):
    #limit the number of displayed points without changing the fitting data.
    plot_step = max(1, int(np.ceil(n_samples / max_plot_points)))
    return np.arange(0, n_samples, plot_step)


def calculate_nonlinear_force(q, q_dot, q_ddot, f_ext):

    return f_ext - q_ddot @ m.T - q_dot @ c.T - q @ k.T


##############################  DETECTION AND CHARACTERIZATION (Q2a, Q2b) #######################################
#-------------------------------------------------------------------------------

def characterize(t, q, q_dot, q_ddot, f_nl):
    print("\n=== 2a: NONLINEAR FORCE LEVELS ===")
    for dof in range(2):
        print(f"DOF {dof + 1}: RMS nonlinear force = {rms(f_nl[:, dof]):.5g} N")

    force_sum = f_nl[:, 0] + f_nl[:, 1]
    force_difference = f_nl[:, 0] - f_nl[:, 1]
    print(f"RMS(Fnl1 + Fnl2) = {rms(force_sum):.5g} N")
    print(f"RMS(Fnl1 - Fnl2) = {rms(force_difference):.5g} N")
    print("A small force sum may indicate a nonlinear element between the masses.")

    relative_displacement = q[:, 0] - q[:, 1]
    relative_velocity = q_dot[:, 0] - q_dot[:, 1]
    relative_acceleration = q_ddot[:, 0] - q_ddot[:, 1]

    relative_force = 0.5 * force_difference

    names = ["DOF1", "DOF2", "relative"]
    displacements = [q[:, 0], q[:, 1], relative_displacement]
    velocities = [q_dot[:, 0], q_dot[:, 1], relative_velocity]
    accelerations = [q_ddot[:, 0], q_ddot[:, 1], relative_acceleration]
    nonlinear_forces = [f_nl[:, 0], f_nl[:, 1], relative_force]

    indices = plot_indices(len(t))

    for i in range(3):
        name = names[i]
        displacement = displacements[i]
        velocity = velocities[i]
        acceleration = accelerations[i]
        force = nonlinear_forces[i]

        plt.figure(figsize=(9, 4))
        plt.plot(t, displacement)
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        save_figure("time_displacement_" + name)

        plt.figure(figsize=(9, 4))
        plt.plot(t, force)
        plt.xlabel("Time (s)")
        plt.ylabel("Nonlinear force (N)")
        save_figure("time_nonlinear_force_" + name)

        plt.figure(figsize=(7, 5))
        plt.scatter(displacement[indices], force[indices], s=2)
        plt.xlabel("Displacement (m)")
        plt.ylabel("Nonlinear force (N)")
        save_figure("force_vs_displacement_" + name)

        plt.figure(figsize=(7, 5))
        plt.scatter(velocity[indices], force[indices], s=2)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Nonlinear force (N)")
        save_figure("force_vs_velocity_" + name)

        #ASM
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(displacement[indices], velocity[indices], acceleration[indices], s=2)
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_zlabel("Acceleration (m/s²)")
        save_figure("ASM_" + name)

        #RFS
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(displacement[indices], velocity[indices], force[indices], s=2)
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_zlabel("Nonlinear force (N)")
        save_figure("RFS_" + name)

    if spectrogram_enabled:
        dt = np.median(np.diff(t))
        sampling_frequency = 1 / dt
        window_points = int(round(spectrogram_window_seconds * sampling_frequency))
        window_points = min(len(t), max(64, window_points))

        if len(t) >= 64:
            overlap = int(0.85 * window_points)

            for i in range(3):
                frequencies, times, magnitude = spectrogram(
                    displacements[i],
                    fs=sampling_frequency,
                    nperseg=window_points,
                    noverlap=overlap,
                    detrend="constant",
                    scaling="spectrum",
                    mode="magnitude",
                )
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-15))

                plt.figure(figsize=(9, 5))
                plt.pcolormesh(t[0] + times, frequencies, magnitude_db,
                               shading="auto", cmap="viridis")
                plt.ylim(0, min(100, sampling_frequency / 2))
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.colorbar(label="Magnitude (dB, relative to 1 m)")
                save_figure("spectrogram_" + names[i])

    #an internal nonlinear force acts with opposite signs on the two masses.
    plt.figure(figsize=(6, 5))
    plt.scatter(f_nl[indices, 0], f_nl[indices, 1], s=2)
    force_min = np.min(f_nl[indices])
    force_max = np.max(f_nl[indices])
    plt.plot([force_min, force_max], [-force_min, -force_max],"k--", label="Fnl2 = -Fnl1")
    plt.xlabel("Nonlinear force DOF 1 (N)")
    plt.ylabel("Nonlinear force DOF 2 (N)")
    plt.legend()
    save_figure("force_coupling_check")

##############################  BUILD THE SELECTED NONLINEAR MODEL (Q2c) #######################################
#-------------------------------------------------------------------------------

def candidate_signal(q, q_dot, location, variable, degree):
    #select the motion associated with the nonlinear element.
    if location == "mass1":
        displacement = q[:, 0]
        velocity = q_dot[:, 0]
    elif location == "mass2":
        displacement = q[:, 1]
        velocity = q_dot[:, 1]
    else:  #between the two masses
        displacement = q[:, 0] - q[:, 1]
        velocity = q_dot[:, 0] - q_dot[:, 1]

    if variable == "q":
        return displacement**degree
    elif variable == "v":
        return velocity**degree
    elif variable == "q_abs":
        return displacement * np.abs(displacement)**(degree - 1)
    else:  #v_abs
        return velocity * np.abs(velocity)**(degree - 1)


def build_joint_matrix(q, q_dot, terms):
    #Rows 0:n describe DOF 1; rows n:2*n describe DOF 2.
    #Each column represents one coefficient to identify.
    n_samples = len(q)
    matrix = np.zeros((2 * n_samples, len(terms)))

    for j, term in enumerate(terms):
        location, variable, degree = term
        signal = candidate_signal(q, q_dot, location, variable, degree)

        if location == "mass1":
            matrix[:n_samples, j] = signal
        elif location == "mass2":
            matrix[n_samples:, j] = signal
        else:
            #same coupling coefficient, opposite forces (action-reaction).
            matrix[:n_samples, j] = signal
            matrix[n_samples:, j] = -signal

    return matrix


def fit_model(q, q_dot, f_nl, terms):
    if len(terms) == 0:
        print("Choose nonlinear terms in selected_terms before fitting.")
        return None

    n_samples = len(q)
    n_train = int(train_fraction * n_samples)

    matrix = build_joint_matrix(q, q_dot, terms)
    measured_force = np.concatenate((f_nl[:, 0], f_nl[:, 1]))

    #keep the same time interval for training both DOFs.
    train_indices = np.concatenate((np.arange(n_train),n_samples + np.arange(n_train)))
    validation_indices = np.concatenate(( np.arange(n_train, n_samples), n_samples + np.arange(n_train, n_samples)))

    train_matrix = matrix[train_indices]
    train_force = measured_force[train_indices]

    #normalize the columns before least squares, then recover SI coefficients.
    scales = np.sqrt(np.mean(train_matrix**2, axis=0))
    normalized_matrix = train_matrix / scales
    normalized_coefficients, _, rank, singular_values = np.linalg.lstsq(normalized_matrix, train_force, rcond=None)
    coefficients = normalized_coefficients / scales

    predicted_force = matrix @ coefficients
    predicted_force_dofs = np.column_stack((
        predicted_force[:n_samples],
        predicted_force[n_samples:],
    ))

    print("\n=== 2c: IDENTIFIED COEFFICIENTS ===")
    for term, coefficient in zip(terms, coefficients):
        print(f"{str(term):32s}: {coefficient:+.6e} (SI units)")

    print(f"Selected terms: {len(terms)}")
    print(f"Matrix rank: {rank}")
    if singular_values[-1] > 0:
        condition_number = singular_values[0] / singular_values[-1]
        print(f"Normalized matrix condition number: {condition_number:.3e}")

    for label, indices in [("TRAIN", train_indices), ("VALIDATION", validation_indices)]:
        error = measured_force[indices] - predicted_force[indices]
        relative_error = 100 * rms(error) / max(rms(measured_force[indices]), 1e-30)
        print(f"{label}: RMSE = {rms(error):.6e} N, "
              f"relative RMSE = {relative_error:.5f}%")

    for dof in range(2):
        for label, time_slice in [("train", slice(0, n_train)),
                                  ("validation", slice(n_train, None))]:
            error = f_nl[time_slice, dof] - predicted_force_dofs[time_slice, dof]
            reference = rms(f_nl[time_slice, dof])
            relative_error = 100 * rms(error) / max(reference, 1e-30)
            print(f"DOF {dof + 1} {label}: RMSE = {rms(error):.5e} N "
                  f"({relative_error:.5f}%)")

    return predicted_force_dofs, coefficients

##############################  COMPARE THE MODEL WITH THE EXPERIMENT (Q2c) #######################################
#-------------------------------------------------------------------------------

def plot_validation(t, q, q_dot, q_ddot, f_ext, f_nl, predicted_force):
    n_samples = len(t)
    indices = plot_indices(n_samples)

    names = ["DOF1", "DOF2", "relative"]
    displacements = [q[:, 0], q[:, 1], q[:, 0] - q[:, 1]]
    velocities = [q_dot[:, 0], q_dot[:, 1], q_dot[:, 0] - q_dot[:, 1]]
    measured_forces = [
        f_nl[:, 0], f_nl[:, 1], 0.5 * (f_nl[:, 0] - f_nl[:, 1])
    ]
    model_forces = [
        predicted_force[:, 0], predicted_force[:, 1],
        0.5 * (predicted_force[:, 0] - predicted_force[:, 1])
    ]

    #compare total restoring forces, including the known linear terms.
    measured_restoring = f_ext - q_ddot @ m.T
    model_restoring = q_dot @ c.T + q @ k.T + predicted_force

    for dof in range(2):
        fig = plt.figure(figsize=(11, 5))
        ax_measured = fig.add_subplot(121, projection="3d")
        ax_model = fig.add_subplot(122, projection="3d")

        for ax, values, title in [
            (ax_measured, measured_restoring[:, dof], "Experimental restoring force"),
            (ax_model, model_restoring[:, dof], "Model restoring force"),
        ]:
            ax.scatter(q[indices, dof], q_dot[indices, dof], values[indices], s=2)
            ax.set_xlabel("Displacement (m)")
            ax.set_ylabel("Velocity (m/s)")
            ax.set_zlabel("Total restoring force (N)")
            ax.set_title(title)

        save_figure(f"total_restoring_surface_DOF{dof + 1}")

    for i in range(3):
        name = names[i]
        displacement = displacements[i]
        velocity = velocities[i]
        measured = measured_forces[i]
        predicted = model_forces[i]

        plt.figure(figsize=(10, 4))
        plt.plot(t, measured, label="Measured nonlinear force", linewidth=1)
        plt.plot(t, predicted, "--", label="Identified model", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Nonlinear force (N)")
        plt.legend()
        save_figure("fit_time_" + name)

        plt.figure(figsize=(7, 5))
        plt.scatter(measured[indices], predicted[indices], s=3)
        force_min = min(np.min(measured[indices]), np.min(predicted[indices]))
        force_max = max(np.max(measured[indices]), np.max(predicted[indices]))
        plt.plot([force_min, force_max], [force_min, force_max], "k--")
        plt.xlabel("Measured nonlinear force (N)")
        plt.ylabel("Predicted nonlinear force (N)")
        save_figure("fit_parity_" + name)

        fig = plt.figure(figsize=(11, 5))
        ax_measured = fig.add_subplot(121, projection="3d")
        ax_model = fig.add_subplot(122, projection="3d")

        for ax, values, title in [
            (ax_measured, measured, "Experiment"),
            (ax_model, predicted, "Model"),
        ]:
            ax.scatter(displacement[indices], velocity[indices], values[indices], s=2)
            ax.set_xlabel("Displacement (m)")
            ax.set_ylabel("Velocity (m/s)")
            ax.set_zlabel("Nonlinear force (N)")
            ax.set_title(title)

        save_figure("RFS_experiment_vs_model_" + name)

#################################  MAIN #######################################
#-------------------------------------------------------------------------------


def main():
    os.makedirs(figure_dir, exist_ok=True)

    t, q, q_dot, q_ddot, f_ext = load_data()
    print(f"Original samples: {len(t)}")

    t = t[::step]
    q = q[::step]
    q_dot = q_dot[::step]
    q_ddot = q_ddot[::step]
    f_ext = f_ext[::step]

    print(f"Samples after decimation: {len(t)}")
    sampling_frequency = 1 / np.median(np.diff(t))
    print(f"Downsampled Nyquist frequency: {sampling_frequency / 2:.2f} Hz")

    f_nl = calculate_nonlinear_force(q, q_dot, q_ddot, f_ext)

    # 2a and 2b: detect and characterize the nonlinear behavior.
    characterize(t, q, q_dot, q_ddot, f_nl)

    # 2c: estimate the coefficients of the manually selected terms.
    if run_estimation:
        result = fit_model(q, q_dot, f_nl, selected_terms)
        if result is not None:
            predicted_force, coefficients = result
            plot_validation(t, q, q_dot, q_ddot, f_ext, f_nl, predicted_force)

    return

if __name__ == "__main__":
    main()
