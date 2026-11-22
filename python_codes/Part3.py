#==================================================

# AERO0035-1 Nonlinear Vibrations of Aerospace Structures
# Part 3 : Simulate
# 2026-2027 Project
# Authors: Leo Goffin, Emmanuelle Henrion, Anaelle Sohier

#==================================================

from python_codes.general_functions import *

########################## SETTINGS #######################
#----------------------------------------------------------

M = np.array([[1, 0], [0, 1]])
C = np.array([[3, -1], [-1, 3]])
K = np.array([[2e4, -1e4], [-1e4, 2e4]])
F0 = 20.0
k4 = 6.7e5
k3 = 3.3e6

def f_nl(q, q_dot):
    F = np.zeros_like(q)
    F[:, 0] = k4 * q[:, 0]**4
    F[:, 1] = k3 * q[:, 1]**3
    return F

#--------------------------------
# NLFR computation (3a)
#--------------------------------
def hbm_pseudo_arclength(F0,
                         omega0, omegamax,
                         n_harmonics=5,
                         n_samples=128,
                         n_steps=300, ds=0.02,
                         ds_min=1e-6, ds_max=0.2):

    def f_ext(omega, t, F0):
        F = np.zeros((len(t), 2))
        F[:, 0] = F0 * np.cos(omega * t)
        return F

    ndof = M.shape[0]
    nc = 2 * n_harmonics + 1
    theta = 2 * np.pi * np.arange(n_samples) / n_samples

    """
    Fourier decomposition : 

    """
    Phi = np.zeros((n_samples, nc))
    Phi[:, 0] = 0.5
    for k in range(1, n_harmonics + 1):
        Phi[:, 2*k - 1] = np.sin(k * theta)
        Phi[:, 2*k] = np.cos(k * theta)

    """
    Differentiation matrix : 

    """
    def D_matrix(omega):
        D = np.zeros((nc, nc))
        for k in range(1, n_harmonics + 1):
            D[2*k-1, 2*k] = -k * omega
            D[2*k, 2*k-1] = k * omega
        return D


    """
    HBM residual : 

    """
    def residual(z, omega):
        Z = z.reshape(nc, ndof)
        D = D_matrix(omega)
        q = Phi @ Z
        qdot = Phi @ D @ Z
        qddot = Phi @ D @ D @ Z
        r = qddot @ M.T + qdot @ C.T + q @ K.T
        r += f_nl(q, qdot) - f_ext(omega, theta / omega,F0)
        R = 2 / n_samples * Phi.T @ r
        R[0] *= 2
        return R.ravel()

    # Find the 2 first points of the branch by guessing
    z1 = least_squares(lambda z: residual(z, omega0), np.zeros(nc * ndof)).x # initial guess full of 0
    z2 = least_squares(lambda z: residual(z, omega0 + ds_min), z1).x         # next guess is the previous solution
    previous = np.r_[z1, omega0]
    current  = np.r_[z2, omega0 + ds]
    branch   = [previous.copy(), current.copy()]

    for _ in range(n_steps):

        # Normal tangent to the branch
        tangent = current - previous
        tangent /= np.linalg.norm(tangent)

        """
        Pseudo arc-length : 
        Follows the branch to find solutions to the HB problem
        """
        max_its = 10
        for _ in range(max_its):
            predictor = current + ds * tangent

            def res_on_perpendicular(y):
                z, omega = y[:-1], y[-1]
                balance = residual(z, omega)
                arc = np.dot(y - predictor, tangent)
                return np.r_[balance, arc]

            sol = least_squares(res_on_perpendicular, predictor, max_nfev=100)
            error = np.max(np.abs(res_on_perpendicular(sol.x)))

            if sol.success and error < 1e-6 and sol.x[-1] > 0:
                break

            ds *= 0.5  # Correction failed: try a shorter step
            if ds < ds_min:
                print('Continuation stopped: ds below ds_min.')
                break

        if ds < ds_min:
            break

        previous, current = current, sol.x
        branch.append(current.copy())

        """
        Adapatative step size :
            if convergence is slow : reduce step size
            if convergence is fast : increase step size
        """
        if sol.nfev >= 6 :
            ds = max(ds_min, ds * 0.5)  
        elif sol.nfev <= 4 :
            ds = min(ds_max, ds * 1.2)  

        # Maximum frequency
        if current[-1] >= omegamax:
            break

    branch = np.asarray(branch)
    omega = branch[:, -1]
    coeffs = branch[:, :-1].reshape(-1, nc, ndof)
    q = np.array([Phi @ Z for Z in coeffs])

    # Peak to peak amplitude/2 
    amplitude = (q.max(axis=1) - q.min(axis=1)) / 2

    return {'omega': omega,
            'frequency_hz': omega / (2 * np.pi),
            'amplitude': amplitude}

#--------------------------------
# NNM computation (3b-3c)
#--------------------------------

def nnm_pseudo_arclength(mode, A0=0.0005, Amax=0.25,
                         n_harmonics=5, n_samples=128,
                         n_steps=5000, ds=0.02,
                         ds_min=1e-6, ds_max=0.2):

    ndof = M.shape[0]
    nc = n_harmonics + 1
    theta = 2 * np.pi * np.arange(n_samples) / n_samples

    """
    Fourier decomposition:
    Only cosine harmonics are necessary
    """
    Phi = np.zeros((n_samples, nc))
    Phi[:, 0] = 0.5

    for k in range(1, nc):
        Phi[:, k] = np.cos(k * theta)

    """
    HBM residual:
    No damping, no external force.
    Frequency is unknown.
    """
    def residual(z, omega):
        Z = z.reshape(nc, ndof)

        q = Phi @ Z
        qddot = Phi @ (-(np.arange(nc) * omega)[:, None]**2 * Z)

        Fnl = np.zeros_like(q)
        Fnl[:, 0] = k4 * q[:, 0]**4
        Fnl[:, 1] = k3 * q[:, 1]**3

        r = qddot @ M.T + q @ K.T + Fnl

        R = 2 / n_samples * Phi.T @ r
        R[0] *= 2

        return R.ravel()

    """
    Initial guess from the linear modes
    """
    eigenvalues, eigenvectors = eigh(K, M)
    omega_linear = np.sqrt(eigenvalues[mode])
    shape = eigenvectors[:, mode] / eigenvectors[0, mode]

    """
    Find the first two points by prescribing
    the first cosine coefficient of DOF 1.
    """
    def first_point(A, guess):

        def equations(y):
            z, omega = y[:-1], y[-1]
            return np.r_[residual(z, omega), z[ndof] - A]

        sol = least_squares(equations, guess, xtol=1e-12,
                            ftol=1e-12, gtol=1e-12)

        return sol.x

    Z0 = np.zeros((nc, ndof))
    Z0[1] = A0 * shape

    guess = np.r_[Z0.ravel(), omega_linear]

    previous = first_point(A0, guess)
    current = first_point(1.2 * A0, previous)
    branch = [previous.copy(), current.copy()]

    for _ in range(n_steps):

        # Unit tangent to the branch
        tangent = current - previous
        tangent /= np.linalg.norm(tangent)

        """
        Pseudo-arclength continuation
        """
        converged = False

        for _ in range(10):
            predictor = current + ds * tangent

            def res_on_perpendicular(y):
                z, omega = y[:-1], y[-1]
                balance = residual(z, omega)
                arc = np.dot(y - predictor, tangent)
                return np.r_[balance, arc]

            sol = least_squares(res_on_perpendicular, predictor,
                                max_nfev=100)

            error = np.max(np.abs(res_on_perpendicular(sol.x)))

            if sol.success and error < 1e-6 and sol.x[-1] > 0:
                converged = True
                break

            ds *= 0.5

            if ds < ds_min:
                break

        if not converged:
            print(f"NNM {mode+1}: continuation stopped")
            break

        previous, current = current, sol.x
        branch.append(current.copy())

        """
        Adaptive step size
        """
        if sol.nfev >= 15:
            ds = max(ds_min, ds * 0.5)

        elif sol.nfev <= 5:
            ds = min(ds_max, ds * 1.2)

        # Maximum amplitude
        Z = current[:-1].reshape(nc, ndof)
        q = Phi @ Z
        amplitude = (q.max(axis=0) - q.min(axis=0)) / 2

        if amplitude[0] >= Amax:
            break

    """
    Extract frequencies, amplitudes and Fourier coefficients
    """
    branch = np.asarray(branch)
    omega = branch[:, -1]
    coeffs = branch[:, :-1].reshape(-1, nc, ndof)

    q = np.array([Phi @ Z for Z in coeffs])
    amplitude = (q.max(axis=1) - q.min(axis=1)) / 2

    return {'omega': omega,
            'frequency_hz': omega / (2*np.pi),
            'amplitude': amplitude}


"""
PLOTS
"""
# 3a
def plot_NLFR(F0,figure_dir):
    for dof in range(2):
        setfig()
        result = hbm_pseudo_arclength(F0 = F0,omega0=30.0, omegamax=300, n_harmonics=5, n_steps=5000, ds=0.02)
        plt.plot(result['frequency_hz'], result['amplitude'][:, dof])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Displacement amplitude (m)')
        plt.xlim(min(result["frequency_hz"]),max(result["frequency_hz"]))
        plt.ylim(0)
        savefig(f"NLFR_{dof+1}.pdf",figure_dir)


#3b-3c
def plot_NNM(F0,figure_dir):
    result = hbm_pseudo_arclength(F0 = F0,omega0=30.0, omegamax=300, n_harmonics=5, n_steps=5000, ds=0.02)
    NNMs = [0,0]
    for mode in range(2):
        NNMs[mode] = nnm_pseudo_arclength(mode)

    for dof in range(2):
        setfig()
        plt.plot(result['frequency_hz'], result['amplitude'][:, dof])
        for mode in range(2):

            plt.plot(NNMs[mode]["frequency_hz"],NNMs[mode]['amplitude'][:, dof],linestyle = '--')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Displacement amplitude (m)')
        plt.xlim(min(result["frequency_hz"]),max(result["frequency_hz"]))
        plt.ylim(0,0.3*F0/100)
        savefig(f"NNM_{dof+1}.pdf",figure_dir)

    setfig()
    for mode in range(2):
        NNM = NNMs[mode]
        ratio = NNM['amplitude'][:, 0] / NNM['amplitude'][:, 1]
        if mode == 1 : ratio *= -1
        plt.plot(NNM['amplitude'][:, 0], ratio,label=f'NNM {mode+1}')

    plt.axhline(1, color='black', linestyle='--',label='LNM 1',alpha = 0.6)
    plt.axhline(-1, color='black', linestyle='-.',label='LNM 2',alpha = 0.6)
    plt.xlabel(r'q1_0 (m)')
    plt.ylabel(r'$A_1/A_2$')
    plt.legend()
    savefig("mode_shapes.pdf",figure_dir)

# 3d
def plot_comparison_sinesweep(F0,mat_file,figure_dir,direction = "forward"):
    result = hbm_pseudo_arclength(F0 = F0,omega0=30.0, omegamax=300, n_harmonics=5, n_steps=5000, ds=0.02)
    t, q, v, a, force = loadmat(mat_file) 
    f = np.linspace(6,40,len(t)) if direction == "forward" else np.linspace(40,6,len(t))
    q = np.asarray(q, dtype=float)

    setfig()
    plt.plot(result["frequency_hz"], result["amplitude"][:, 0], label="NLFR", linewidth=2)
    plt.plot(f,q[:,0],label = "Sine sweep", alpha = 0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DOF 1 displacement amplitude")
    plt.xlim(6,40)
    plt.ylim(0)
    plt.legend()
    savefig("comp_sinesweep.pdf",figure_dir)