#==================================================

# AERO0035-1 Nonlinear Vibrations of Aerospace Structures
# Part 2 : Identify
# 2026-2027 Project
# Authors: Leo Goffin, Emmanuelle Henrion, Anaelle Sohier

#==================================================
"""
Question 2:
- Q2a: detection of the nonlinearities
- Q2b: characterization using ASM
- Q2c: parameter estimation using RFS

Equations of motion : M*q_ddot + C*q_dot + K*q + F_nl = F_ext
"""

from general_functions import *

########################## SETTINGS #######################
#----------------------------------------------------------

#Dynamics of the local component:
M = np.array([[1, 0],
              [0, 1]])

C = np.array([[3, -1],
              [-1, 3]])

K = np.array([[2e4, -1e4],
              [-1e4, 2e4]])


dof_colors = ["blue", "red"]

#(location, variable, degree): 
#    location = mass1/mass2/between
#    variable = q (displacement), v (velocity)
#    degree = 1,2,...

max_plot_points = 20000

########################## USEFUL FUNCTIONS #######################
#-----------------------------------------------------------------

def rms(values):

    values = np.asarray(values)     
    squared_values = values**2     
    mean_squared = np.mean(squared_values)
    rms = np.sqrt(mean_squared)      
    rms = float(rms)     

    return rms


def plot_indices(n_samples):

    #maximum number of points to display
    ratio = n_samples / max_plot_points

    #number of samples between two displayed points
    step = int(np.ceil(ratio))
    step = max(1, step)

    #select the indices of the points to display
    indices = np.arange(0, n_samples, step)

    return indices

########################## Q2a: DETECTION #######################
#----------------------------------------------------------------

def calculate_residual_forces(q, q_dot, q_ddot, f_ext):

    residu = f_ext - q_ddot @ M.T - q_dot @ C.T - q @ K.T

    return residu


def detect_nonlinearity(t, f_nl, figure_dir):

    print("\n--- Q2a: NONLINEAR RESIDUALS ---")

    for dof in range(2):

        nonlinear_force = f_nl[:, dof]
        force_rms = rms(nonlinear_force)

        print(f"DOF {dof+1}: RMS(F_nl) = {force_rms:.4e} N")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(t, nonlinear_force, color=dof_colors[dof], lw=0.8, label= f"DOF {dof+1}")

        ax.set_xlabel("Time [s]", fontsize=20)
        ax.set_ylabel("Residual force [N]", fontsize=20)
        ax.tick_params(labelsize=18)
        ax.grid(False)
        ax.legend(fontsize=18)

        fig.tight_layout()
        file_path = Path(figure_dir) / f"Q2a_residual_DOF{dof+1}.pdf"
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)

    return 

########################## Q2b: CHARACTERIZATION #######################
#----------------------------------------------------------------------

def characterize_nonlinearity(t, q, q_dot, q_ddot, f_nl, figure_dir):

    idx = plot_indices(len(t))
    figure_dir = Path(figure_dir)
   
    for dof in range(2):

        displacement = q[idx, dof]
        velocity = q_dot[idx, dof]
        acceleration = q_ddot[idx, dof]
        nonlinear_force = f_nl[idx, dof]

        color = dof_colors[dof]

        #RESIDUAL FORCE VS MOTION
        fig, ax = plt.subplots(figsize=(7, 5))

        # Force versus displacement
        ax.scatter(displacement, nonlinear_force,s=3, color=color, alpha=0.6)

        ax.set_xlabel(rf"$q_{dof+1}$ [m]", fontsize=18)
        ax.set_ylabel(rf"$F_{{\mathrm{{res}},{dof+1}}}$ [N]",fontsize=18 )

        ax.tick_params(axis="both", labelsize=16, pad=6)
        ax.ticklabel_format(axis="both", style="sci",scilimits=(-3, 3), useMathText=True)
        ax.grid(False)

        fig.tight_layout()

        file_path = figure_dir / f"Q2b_force_vs_displacement_DOF{dof+1}.pdf"
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        
        
        # Force versus velocity
        fig, ax = plt.subplots(figsize=(7, 5))
        
        ax.scatter(velocity, nonlinear_force,s=3, color=color, alpha=0.6)
        ax.set_xlabel(rf"$\dot{{q}}_{dof+1}$ [m/s]",fontsize=18)
        ax.set_ylabel(rf"$F_{{\mathrm{{res}},{dof+1}}}$ [N]",fontsize=18)

       
        ax.tick_params(axis="both", labelsize=16, pad=6)
        ax.ticklabel_format(axis="both", style="sci",scilimits=(-3, 3), useMathText=True)
        ax.grid(False)

        fig.tight_layout()

        file_path = figure_dir / f"Q2b_force_vs_velocity_DOF{dof+1}.pdf"
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        

        #3D STATE-SPACE TRAJECTORY
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(displacement, velocity, acceleration,s=2, color=color, alpha=0.5,rasterized=True)

        ax.set_xlabel(rf"$q_{dof+1}$ [m]", fontsize=16, labelpad=12)
        ax.set_ylabel(rf"$\dot{{q}}_{dof+1}$ [m/s]",fontsize=16, labelpad=12)
        ax.set_zlabel(rf"$\ddot{{q}}_{dof+1}$ [m/s$^2$]",fontsize=16, labelpad=14)

        ax.tick_params(axis="both", labelsize=11, pad=4)
        ax.view_init(elev=25, azim=-55)
        ax.grid(False)

        
        fig.subplots_adjust(left=0.08, right=0.83,bottom=0.08, top=0.95 )

        file_path = figure_dir / f"Q2b_ASM_DOF{dof+1}.pdf"
        fig.savefig(file_path, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        
    return 



########################## Q2c: EXPLORATORY IDENTIFICATION #######################
#--------------------------------------------------------------------------------

def build_exploratory_basis(q, q_dot, degree):

    #combine displacements and velocities into one matrix
    variables = np.column_stack((q, q_dot))
    variable_names = ["q1", "q2", "v1", "v2"]

    #compute the RMS of each variable
    squared_variables = variables**2
    mean_squared = np.mean(squared_variables, axis=0)
    scales = np.sqrt(mean_squared)

    if np.any(scales <= 0):
        raise ValueError("At least one displacement/velocity has zero RMS.")

    #normalize variables to avoid large differences in magnitude
    normalized_variables = variables / scales

    powers_list = []
    labels = []
    columns = []

    #start at degree 2 because linear dynamics have already been removed
    for order in range(2, degree + 1):

        #generate all combinations of the four variables
        combinations = combinations_with_replacement(range(4), order)

        for indices in combinations:

            #count how many times each variable appears
            powers = np.bincount(indices, minlength=4)

            #construct a readable name for the candidate term
            label_parts = []

            for j in range(4):

                name = variable_names[j]
                power = powers[j]

                if power == 1:
                    label_parts.append(name)

                elif power > 1:
                    label_parts.append(f"{name}^{power}")

            label = "*".join(label_parts)

            #compute the candidate signal
            candidate = np.ones(len(q))

            for j in range(4):

                power = powers[j]
                variable = normalized_variables[:, j]

                candidate = candidate * variable**power

            powers_list.append(powers)
            labels.append(label)
            columns.append(candidate)

    basis_matrix = np.column_stack(columns)
    powers_array = np.array(powers_list)

    return basis_matrix, powers_array, labels


def exploratory_sparse_fit(matrix, measured_force):

    max_terms = 25 #arbitrary 
    min_improvement = 0.01 #arbitrary

    #TRAINING DATA 
    n_samples = len(measured_force)
    n_train = int(0.75 * n_samples)
    n_candidates = matrix.shape[1]

    train_matrix = matrix[:n_train, :]
    train_force = measured_force[:n_train]

    #NORMALIZATION
    #compute the RMS of each candidate using training data
    squared_matrix = train_matrix**2
    mean_squared = np.mean(squared_matrix, axis=0)
    scales = np.sqrt(mean_squared)

    normalized_matrix = np.zeros_like(matrix)
    valid_terms = []

    for j in range(n_candidates):

        if scales[j] > 1e-12:

            normalized_matrix[:, j] = matrix[:, j] / scales[j]
            valid_terms.append(j)


    #INITIALIZATION
    selected_terms = []

    #initially, the model predicts zero nonlinear force
    residual = train_force.copy()
    initial_rms = rms(train_force)


    #TERM SELECTION 
    for iteration in range(max_terms):

        best_term = None
        best_score = -1

        #find the candidate most correlated with the residual
        for j in valid_terms:

            if j in selected_terms:
                continue

            candidate = normalized_matrix[:n_train, j]

            correlation = np.dot(candidate, residual)
            score = abs(correlation) / n_train

            if score > best_score:
                best_score = score
                best_term = j

        #if no candidate is available
        if best_term is None:
            break

        #TEMPORARY MODEL 
        #add the best candidate temporarily
        trial_terms = selected_terms.copy()
        trial_terms.append(best_term)

        #build the temporary model matrix
        trial_matrix = normalized_matrix[:n_train, trial_terms]

        #identify coefficients using least squares
        trial_coefficients = np.linalg.lstsq(trial_matrix, train_force, rcond=None)[0]

        #predict the nonlinear force
        trial_prediction = trial_matrix @ trial_coefficients

        #compute the remaining error
        trial_residual = train_force - trial_prediction


        #IMPROVEMENT CHECK 
        previous_error = rms(residual)
        new_error = rms(trial_residual)

        error_reduction = previous_error - new_error
        improvement = error_reduction / max(previous_error, 1e-30)

        #stop if the improvement is too small
        if improvement < min_improvement:
            break

        #accept the new candidate
        selected_terms.append(best_term)
        residual = trial_residual


    #FINAL COEFFICIENTS

    coefficients = np.zeros(n_candidates)

    if len(selected_terms) > 0:

        #keep only the selected candidates
        selected_matrix = normalized_matrix[:n_train, selected_terms]

        #identify their coefficients
        normalized_coefficients = np.linalg.lstsq(selected_matrix, train_force, rcond=None )[0]

        #convert coefficients back to the original basis
        for j in range(len(selected_terms)):

            candidate_index = selected_terms[j]
            normalized_coefficient = normalized_coefficients[j]

            coefficients[candidate_index] = (normalized_coefficient / scales[candidate_index])


    #FORCE RECONSTRUCTION
    predicted_force = matrix @ coefficients


    #VALIDATION

    print(f"\nSelected terms: {len(selected_terms)} / {n_candidates}")

    for dataset in ["training", "validation"]:

        if dataset == "training":
            measured = measured_force[:n_train]
            predicted = predicted_force[:n_train]

        else:
            measured = measured_force[n_train:]
            predicted = predicted_force[n_train:]

        #compute the prediction error
        error = measured - predicted

        rmse = rms(error)
        measured_rms = rms(measured)

        relative_rmse = 100 * rmse / max(measured_rms, 1e-30)

        print(
            f"{dataset}: RMSE = {rmse:.4e} N, "
            f"relative RMSE = {relative_rmse:.3f}%"
        )

    return coefficients, predicted_force, selected_terms



def plot_exploratory_tables(matrix, powers, labels, coefficients, dof, exploration_degree, figure_dir):

    #RMS CONTRIBUTIONS

    n_terms = len(coefficients)
    contribution_rms = np.zeros(n_terms)

    #compute the RMS force contribution of each term
    for j in range(n_terms):

        force = coefficients[j] * matrix[:, j]
        contribution_rms[j] = rms(force)

    #compute relative contributions
    total_rms = np.sum(contribution_rms)

    shares = np.zeros(n_terms)

    if total_rms > 0:
        shares = 100 * contribution_rms / total_rms

    #common color scale for all plots
    max_share = max(1, np.max(shares))


    #DISPLACEMENT AND VELOCITY TERMS 

    for kind in ["displacement", "velocity"]:


        if kind == "displacement":

            variable_1 = 0
            variable_2 = 1

            xlabel = r"Exponent of $q_1$"
            ylabel = r"Exponent of $q_2$"

        else:

            variable_1 = 2
            variable_2 = 3

            xlabel = r"Exponent of $\dot{q}_1$"
            ylabel = r"Exponent of $\dot{q}_2$"

        #contribution table
        grid_size = exploration_degree + 1
        grid = np.zeros((grid_size, grid_size))
        for j in range(n_terms):

            exponents = powers[j]

            if kind == "displacement":
                is_relevant = (exponents[2] == 0 and exponents[3] == 0)

            else:
                is_relevant = (exponents[0] == 0 and exponents[1] == 0)

            if not is_relevant:
                continue

            #find the position of the term in the table
            column = int(exponents[variable_1])
            row = int(exponents[variable_2])

            grid[row, column] = shares[j]


        #PLOT TABLE 
        fig, ax = plt.subplots(figsize=(9, 7))

      
        image = ax.imshow(grid,origin="lower",cmap='plasma',vmin=0,vmax=max_share)
        threshold = 0.75 * max_share
       
        #display percentages inside the cells
        for row in range(grid_size):
            for column in range(grid_size):

                value = grid[row, column]
                if value >= threshold:
                    text_color = "black"
                else:
                    text_color = "white"
               
                ax.text(column, row,  f"{value:.2f}%" ,ha="center", va="center",fontsize=14, color=text_color)

    
        ticks = np.arange(grid_size)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(labelsize=18)
        
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(r"Relative RMS contribution [\%]", fontsize=18)
        colorbar.ax.tick_params(labelsize=14)

        fig.tight_layout()
        file_path = Path(figure_dir) / f"Q2c_exploration_{kind}_DOF{dof+1}.pdf"
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        


    #MIXED TERMS
    print(f"\nMixed terms:")

    min_share_to_print = 0.01  #minimum contribution in %

    mixed_terms = []

    for j in range(n_terms):

        exponents = powers[j]

        has_displacement = (exponents[0] > 0 or exponents[1] > 0)
        has_velocity = (exponents[2] > 0 or exponents[3] > 0)

        if has_displacement and has_velocity:
            if shares[j] >= min_share_to_print:
                mixed_terms.append(j)

    mixed_terms.sort(key=lambda j: shares[j], reverse=True)

    if len(mixed_terms) == 0:
        print("No significant mixed terms identified.")

    for j in mixed_terms:
        print(f"{labels[j]:25s}: {shares[j]:8.4f}%")

    
    #PRINT
    print("\nLeading exploratory terms:")

    min_share_to_print = 0.01

    sorted_terms = np.argsort(shares)[::-1]

    for j in sorted_terms:

        if shares[j] >= min_share_to_print:
            print(f"{labels[j]:22s}: {shares[j]:8.4f}%")

    return 



def explore_all_nonlinearities(q, q_dot, f_nl, exploration_degree, figure_dir):

    matrix, powers, labels = build_exploratory_basis(q, q_dot, exploration_degree)

    n_candidates = matrix.shape[1]

    print("\n--- Q2c: EXPLORATORY IDENTIFICATION ---")
    print(f"Number of candidate terms: {n_candidates}")
    print(f"Maximum polynomial degree: {exploration_degree}")

    for dof in range(2):

        print(f"\n=== DOF {dof + 1} ===")

        measured_force = f_nl[:, dof]

        #identify the most relevant polynomial terms
        coefficients, predicted_force, selected_terms = exploratory_sparse_fit(matrix, measured_force)

        #display the relative contributions of the identified terms
        plot_exploratory_tables(matrix, powers, labels, coefficients, dof, exploration_degree, figure_dir)

    return 




########################## Q2c: RFS MODEL #######################
#----------------------------------------------------------------

def candidate_signal(q, q_dot, location, variable, degree):
   
    if location == "mass1":
        displacement = q[:, 0]
        velocity = q_dot[:, 0]

    elif location == "mass2":
        displacement = q[:, 1]
        velocity = q_dot[:, 1]

    elif location == "between":
        displacement = q[:, 0] - q[:, 1]
        velocity = q_dot[:, 0] - q_dot[:, 1]

    else : 
        raise ValueError("Unknown variable")
        
    if variable == "q":
        signal= displacement**degree
    
    elif variable == "v":
        signal =  velocity**degree
    
    else: 
        raise ValueError("Unknown variable")

    return signal


def term_contribution(q, q_dot, term):
   
    location, variable, degree = term
    signal = candidate_signal(q, q_dot, location, variable, degree)
    contribution = np.zeros((len(q), 2))

    if location == "mass1":
        contribution[:, 0] = signal

    elif location == "mass2":
        contribution[:, 1] = signal

    elif location == "between":
        contribution[:, 0] = signal
        contribution[:, 1] = -signal #principle of action-reaction

    else: 
        raise ValueError("Unknown location")

    return contribution


def build_matrix(q, q_dot, terms):

    n_samples = len(q)
    n_terms = len(terms)

    matrix = np.zeros((2 * n_samples, n_terms)) #first n_samples rows: DOF 1; last n_samples rows: DOF 2.

    for j, term in enumerate(terms):

        contribution = term_contribution(q, q_dot, term)

        force_dof1 = contribution[:, 0]
        force_dof2 = contribution[:, 1]

        combined_force = np.concatenate((force_dof1, force_dof2))

        matrix[:, j] = combined_force

    return matrix


def fit_model(q, q_dot, f_nl, terms):

    
    n_samples = len(q)
    n_train = int(0.75 * n_samples) #75% of the samples are used for training, the remaining 25% for validation

    #build the identification matrix for both degrees of freedom
    #each column corresponds to one candidate nonlinear term
    matrix = build_matrix(q, q_dot, terms)

    #combine the measured nonlinear forces of both masses
    #the first half contains DOF 1, the second half contains DOF 2
    measured = np.concatenate((f_nl[:, 0], f_nl[:, 1]))

    #select the rows corresponding to the training data
    #both degrees of freedom must use the same time samples
    train_dof1 = np.arange(0, n_train)
    train_dof2 = np.arange(n_samples, n_samples + n_train)

    train_indices = np.concatenate((train_dof1, train_dof2))

    train_matrix = matrix[train_indices, :]
    train_forces = measured[train_indices]

    #normalize each column of the identification matrix
    scales = np.sqrt(np.mean(train_matrix**2, axis=0))
    normalized_matrix = train_matrix / scales

    #identify the nonlinear coefficients using least squares
    #the solution minimizes the difference between measured and predicted forces
    normalized_coefficients, _, _, _ = np.linalg.lstsq(normalized_matrix,train_forces,rcond=None)

    #convert the coefficients back to their original SI units
    coefficients = normalized_coefficients / scales

    #reconstruct the nonlinear forces 
    predicted_flat = matrix @ coefficients

    predicted = np.zeros((n_samples, 2))
    predicted[:, 0] = predicted_flat[:n_samples]
    predicted[:, 1] = predicted_flat[n_samples:]


    print("\n--- Q2c: IDENTIFIED COEFFICIENTS ---")

    for term_index in range(len(terms)):

        term = terms[term_index]
        coefficient = coefficients[term_index]

        print(f"{str(term):30s}: {coefficient:+.8e} [SI]")

    
    #evaluate the prediction error separately for each degree of freedom
    for dof in range(2):

        for dataset in ["training", "validation"]:

            if dataset == "training":
                measured_force = f_nl[:n_train, dof]
                predicted_force = predicted[:n_train, dof]

            elif dataset == "validation":
                measured_force = f_nl[n_train:, dof]
                predicted_force = predicted[n_train:, dof]

            else: 
                raise ValueError("Unknown dataset")


            #difference between measured and reconstructed nonlinear forces
            error = measured_force - predicted_force

            #absolute and relative root mean square errors
            rmse = rms(error)
            relative_rmse = 100 * rmse / max(rms(measured_force), 1e-30)

            print(
                f"DOF {dof + 1} - {dataset}: "
                f"RMSE = {rmse:.6e} N, "
                f"relative RMSE = {relative_rmse:.4f}%"
            )

    #calculate the individual force contribution of each identified term
    #these contributions will be used for the plots
    contributions = []

    for term_index in range(len(terms)):

        term = terms[term_index]
        coefficient = coefficients[term_index]

        force_contribution = coefficient * term_contribution(q, q_dot, term)
        contributions.append(force_contribution)

    return predicted, coefficients, contributions, n_train



########################## Q2c: VALIDATION PLOTS #######################
#-----------------------------------------------------------------------

def plot_rfs_validation(q, f_nl, predicted, n_train, figure_dir):

    for dof in range(2):

        displacement = q[n_train:, dof]
        experimental_force = f_nl[n_train:, dof]
        predicted_force = predicted[n_train:, dof]

        order = np.argsort(displacement)

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(displacement,experimental_force,s=6, alpha = 0.4, color=dof_colors[dof],label="Experimental nonlinear force")

        ax.plot(displacement[order],predicted_force[order],color = "black", linestyle="--",linewidth=1.5,label="Identified model")

        ax.set_xlabel(f"Displacement $q_{dof+1}$ [m]", fontsize=20)
        ax.set_ylabel("Nonlinear force [N]", fontsize=20)

        ax.tick_params(labelsize=20)
        ax.legend(fontsize=20)
        ax.grid(False)

        fig.tight_layout()

        file_path = Path(figure_dir) / f"Q2c_RFS_validation_DOF{dof+1}.pdf"
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
    return 
       

def plot_term_contributions(t, f_nl, predicted,  figure_dir):

    for dof in range(2):

        experimental_force = f_nl[:, dof]
        reconstructed_force = predicted[:, dof]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(t,experimental_force,color=dof_colors[dof],lw=1.5,label="Experimental nonlinear force",zorder=2 )

        ax.plot(t,reconstructed_force,linestyle="None",marker="o",markersize=1,color="black",label="Reconstructed nonlinear force",zorder=3)
        
        ax.set_xlabel("Time [s]", fontsize=20)
        ax.set_ylabel("Nonlinear force [N]", fontsize=20)

        ax.tick_params(labelsize=20)
        ax.legend(fontsize=20)
        ax.grid(False)

        fig.tight_layout()

        file_path = Path(figure_dir) / f"Q2c_force_reconstruction_DOF{dof+1}.pdf"
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        
    return 
