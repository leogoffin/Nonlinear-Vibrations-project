import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import pandas as pd

M = np.array([[1, 0],
              [0, 1]])

C = np.array([[3, -1],
              [-1, 3]])

K = np.array([[2e4, -1e4],
              [-1e4, 2e4]])

#on espère recevoir ça des test: (Ns c'est la lg de l'élément)
# q: (Ns, 2), déplacements
# q_dot: (Ns, 2), vitesses
# q_ddot: (Ns, 2), accélérations
# F_ext : (Ns, 2), forces
# t : temps

def calculate_residual_forces(q, q_dot, q_ddot, F_ext, M, C, K):

    F_nl = (F_ext- q_ddot @ M.T- q_dot @ C.T- q @ K.T) #force non linéaire résiduelle
    
    return F_nl


def calculate_residual_rms(F_nl):

    residu_1 = np.sqrt(np.mean(F_nl[:, 0]**2))
    residu_2 = np.sqrt(np.mean(F_nl[:, 1]**2))

    print("RMS force résiduelle DOF 1 :", residu_1)
    print("RMS force résiduelle DOF 2 :", residu_2)

    #residu non nul indiquerait une non linéarité
    #(pourrait aussi être du bruit i guess donc checker avec d'autres méthodes?)

    return residu_1, residu_2


def plot_residual_forces(t, F_nl):

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t, F_nl[:, 0])
    ax[0].set_ylabel("Residual force DOF 1")
    ax[0].grid(True)

    ax[1].plot(t, F_nl[:, 1])
    ax[1].set_ylabel("Residual force DOF 2")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("figures/Residual_forces.pdf",format = "pdf")
    plt.show()

    return 

def plot_asm_surface(q, q_dot, F_nl, dof=0):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(q[:, dof],q_dot[:, dof],F_nl[:, dof],s=8)

    ax.set_xlabel(f"q{dof + 1} [m]")
    ax.set_ylabel(f"q{dof + 1}_dot [m/s]")
    ax.set_zlabel("Residual force [N]")
    ax.set_title(f"ASM - DOF {dof + 1}")

    plt.tight_layout()
    plt.savefig("figures/asm_surface.pdf",format = "pdf")
    plt.show()
    return 

def plot_residual_vs_displacement(q, F_nl, dof=0):

    #Pourrait être utile?

    plt.figure()

    plt.scatter(q[:, dof], F_nl[:, dof], s=8)

    plt.xlabel(f"q{dof + 1} [m]")
    plt.ylabel("Residual force [N]")
    plt.title(f"Residual force versus displacement - DOF {dof + 1}")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("figures/res_force_vs_displ.pdf",format = "pdf")
    plt.show()

    return

def calculate_restoring_force(q_ddot, F_ext, M):

    F_RFS = F_ext - q_ddot @ M.T #formule annexe

    return F_RFS

def estimate_rfs_polynomial(q, q_dot, F_RFS, dof=0,m_order=3, n_order=3):

    x = q[:, dof]
    x_dot = q_dot[:, dof]
    f = F_RFS[:, dof]
    
    columns = []
    term_names = []

    for i in range(m_order + 1):
        for j in range(n_order + 1):
            columns.append((x**i) * (x_dot**j))
            term_names.append((i, j)) #chat?

    X = np.column_stack(columns)

    alpha, residuals, rank, singular_values = np.linalg.lstsq(X, f, rcond=None) #chat?
    f_predicted = X @ alpha

    rmse = np.sqrt(np.mean((f - f_predicted)**2))
    variance_f = np.var(f)

    mse_percent = 100 * np.mean((f - f_predicted)**2) / variance_f #formule annexe
    
    return alpha, f_predicted, term_names, rmse, mse_percent


def print_rfs_results(alpha, term_names, rmse, mse_percent):

    print("\nIdentification RFS")

    for coefficient, (i, j) in zip(alpha, term_names):
        print(f"alpha_({i},{j}) = {coefficient:.6e}")

    print(f"\nRMSE force = {rmse:.6e} N")
    print(f"MSE normalisée = {mse_percent:.3f} %")
    
def plot_rfs_surface(q, q_dot, F_RFS, alpha, term_names,dof=0):

    x = q[:, dof]
    x_dot = q_dot[:, dof]
    f = F_RFS[:, dof]

    x_grid = np.linspace(np.min(x), np.max(x), 40)
    v_grid = np.linspace(np.min(x_dot), np.max(x_dot), 40)

    X_grid, V_grid = np.meshgrid(x_grid, v_grid)
    F_grid = np.zeros_like(X_grid)

    for coefficient, (i, j) in zip(alpha, term_names):
        F_grid += coefficient * X_grid**i * V_grid**j

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, x_dot, f, s=8, label="RFS reconstruite")

    ax.plot_surface(X_grid, V_grid, F_grid,alpha=0.5)

    ax.set_xlabel(f"q{dof + 1} [m]")
    ax.set_ylabel(f"q{dof + 1}_dot [m/s]")
    ax.set_zlabel("Restoring force [N]")
    ax.set_title(f"RFS identifiée - DOF {dof + 1}")

    plt.tight_layout()
    plt.savefig("figures/rfs_surface.pdf",format = "pdf")
    plt.show()

##################### MAIN ########################""
#q, q_dot, q_ddot, F_ext et t doivent être récupérés des tests
#--- Q2.b ----#

def load_csv(filename,key):
    df = pd.read_csv("NI2D_test/"+ filename +".csv")
    return df[key].to_numpy()

t = load_csv("dof1_q","Time (s)")
F1 = load_csv("dof1_force","Force (N)")
F2 = np.zeros_like(F1)

q1 = load_csv("dof1_q","pos(m)")
q2 = load_csv("dof2_q","pos(m)")

qdot1 = load_csv("dof1_qdot","vel(m/s)")
qdot2 = load_csv("dof2_qdot","vel(m/s)")

qddot1 = load_csv("dof1_qdotdot","acc(m/s2)")
qddot2 = load_csv("dof2_qdotdot","acc(m/s2)")

q = np.column_stack((q1, q2))
q_dot = np.column_stack((qdot1, qdot2))
q_ddot = np.column_stack((qddot1, qddot2))
F_ext = np.column_stack((F1, F2))

q = q[::50]
q_dot = q_dot[::50]
q_ddot = q_ddot[::50]
F_ext = F_ext[::50]
t = t[::50]

F_nl = calculate_residual_forces(q, q_dot, q_ddot, F_ext, M, C, K)

residu_1, residu_2 = calculate_residual_rms(F_nl)

plot_residual_forces(t, F_nl)

#DOF 1
plot_asm_surface(q, q_dot, F_nl, dof=0)
plot_residual_vs_displacement(q, F_nl, dof=0)

#DOF 2
plot_asm_surface(q, q_dot, F_nl, dof=1)
plot_residual_vs_displacement(q, F_nl, dof=1)


#---- Q2.c ---- #
F_RFS = calculate_restoring_force(q_ddot, F_ext, M)

#exposants à tester, on verra bien ce qu'on observera à la Q2.b si j'ai bien compris
m_order = 3
n_order = 3


results_1 = estimate_rfs_polynomial(q, q_dot, F_RFS,dof=0,m_order=m_order,n_order=n_order)
alpha_1, f_pred_1, terms_1, rmse_1, mse_1 = results_1

print("\nDOF 1")
print_rfs_results(alpha_1, terms_1, rmse_1, mse_1)

plot_rfs_surface(q, q_dot, F_RFS,alpha_1, terms_1,dof=0)

results_2 = estimate_rfs_polynomial(q, q_dot, F_RFS,dof=1,m_order=m_order,n_order=n_order)
alpha_2, f_pred_2, terms_2, rmse_2, mse_2 = results_2

print("\nDOF 2")
print_rfs_results(alpha_2, terms_2, rmse_2, mse_2)

plot_rfs_surface(q, q_dot, F_RFS,alpha_2, terms_2,dof=1)

