#AERO0035-1 Nonlinear Vibrations of Aerospace Structures
#2026-2027 Project
#Authors: Léo Goffin, Emmanuelle Henrion, Anaëlle Sohier

from python_codes.general_functions import *
import python_codes.Part2 as Q2
import python_codes.Part3 as Q3
import shutil
print(shutil.which("latex"))

project_dir = Path(__file__).resolve().parent
figure_dir =project_dir.parent / "Figs"
figure_dir_Q2 = figure_dir / "Q2"
figure_dir_Q3 = figure_dir / "Q3"
file_path = project_dir.parent / "Signals" / "G7_exp2_2.mat"
    
def main():
    plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 20})

    t, q, q_dot, q_ddot, f_ext= loadmat(file_path)
    print_signal(t, q, q_dot, q_ddot, f_ext)
    plot_signal(t, q, q_dot, q_ddot, f_ext, figure_dir_Q2)

    print(f"Original samples: {len(t)}")

    step = 50
    exploration_degree = 5
    
    t = t[::step]
    q = q[::step]
    q_dot = q_dot[::step]
    q_ddot = q_ddot[::step]
    f_ext = f_ext[::step]
   
    f_nl = Q2.calculate_residual_forces(q, q_dot, q_ddot, f_ext)

    #Q2a
    Q2.detect_nonlinearity(t, f_nl, figure_dir_Q2)

    #Q2b
    Q2.characterize_nonlinearity(t, q, q_dot, q_ddot, f_nl, figure_dir_Q2)

    #Q2c-I: explore all polynomial candidates
    Q2.explore_all_nonlinearities(q, q_dot, f_nl, exploration_degree, figure_dir_Q2)
    
    selected_terms = [("mass2", "q", 3),
                      ("mass2", "q", 5),
                      ("mass1", "q", 4)]
    

    #Q2c-II: fit physically selected terms
    predicted, coefficients, contributions, n_train = Q2.fit_model(q, q_dot, f_nl, selected_terms)

    #Q2c: validation
    Q2.plot_rfs_validation(q, f_nl, predicted, n_train, figure_dir_Q2)
    Q2.plot_term_contributions(t, f_nl, predicted, figure_dir_Q2)

    # Q3a : Non-Linear Frequency Response
    F0 = 100
    Q3.plot_NLFR(F0,figure_dir_Q3)

    # Q3b-Q3c : Non-linear Normal Modes
    Q3.plot_NNM(F0,figure_dir_Q3) 

    # Q3d : Superposition with sinesweep
    Q3.plot_comparison_sinesweep(F0,file_path,figure_dir_Q3,direction = "backward")

    # Q3e : Bassin of attraction
    # TODO 
    return 


if __name__ == "__main__":
    main()

