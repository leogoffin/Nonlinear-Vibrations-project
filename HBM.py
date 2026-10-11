

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

########################### DATA ##############################
M = np.array([[1, 0], 
              [0, 1]])

C = np.array([[3, -1], 
              [-1, 3]])

K = np.array([[2e4, -1e4], 
              [-1e4, 2e4]])

n_dof = 2 

#NB OF HARMONICS
H = 1

N_samples = 16# >2*H +1 #Nyquist

######################## FUNCTIONS ##############################

#Definition of the nonlinear function
def force_nonlinear_time(q, q_dot):
    
    f_nl = np.zeros_like(q)

    #FIND RIGHT FUNCTION!!
    k_nl = 1e6 
    f_nl[0, :] = k_nl * (q[0, :]**3) #exemple pour Duffing: k_nl * q1^3
    
    return f_nl


#Reconstruct the position and velocity from X (state vector in HBM) 
def reconstruct_time_signal(X, omega, t_vector):
   
    q = np.zeros((n_dof, len(t_vector)))
    q_dot = np.zeros((n_dof, len(t_vector)))
    
    q[:, :] += X[:n_dof, np.newaxis] #for harmonic 0
    
    idx = n_dof
    for h in range(1, H + 1): #for harmonics > 0
        A_h = X[idx : idx + n_dof].reshape(-1,1)  #coefficients cosinus
        B_h = X[idx + n_dof : idx + 2*n_dof].reshape(-1,1)   #coefficients sinus
        idx += 2 * n_dof

        cos_term = np.cos(h * omega * t_vector)
        sin_term = np.sin(h * omega * t_vector)

        q += A_h * cos_term + B_h * sin_term #Fourier
        q_dot += -h * omega * A_h * sin_term + h * omega * B_h * cos_term
        
    return q, q_dot


#Alternation frequency/time (AFT) to get F_nl in frequency.
def aft_procedure(X, omega):

    T = 2 * np.pi / omega
    t_vector = np.linspace(0, T, N_samples, endpoint=False)
    dt = t_vector[1] - t_vector[0]

    #Time domain
    q, q_dot = reconstruct_time_signal(X, omega, t_vector) 
    f_nl_t = force_nonlinear_time(q, q_dot)
    
    #Frequency domain
    F_nl_freq = np.zeros_like(X)
    F_nl_freq[:n_dof] = np.mean(f_nl_t, axis=1) #harmonic 0 
    
    idx = n_dof
    for h in range(1, H + 1):
        F_nl_freq[idx : idx + n_dof] = (2 / T) * np.trapezoid(f_nl_t * np.cos(h * omega * t_vector), dx=dt, axis=1) #cos
        F_nl_freq[idx + n_dof : idx + 2*n_dof] = (2 / T) * np.trapezoid(f_nl_t * np.sin(h * omega * t_vector), dx=dt, axis=1) #sin
        idx += 2 * n_dof
        
    return F_nl_freq

#Calcule the residual  R(X) = Z(omega)*X + F_nl(X) - F_ext
#Voir demo tablette
def hbm_residuals(X, omega, F_ext_freq):

    R = np.zeros_like(X)
    R[:n_dof] = K @ X[:n_dof] #Harmonic 0
    
    idx = n_dof
    for h in range(1, H + 1):
        A_h = X[idx : idx + n_dof]
        B_h = X[idx + n_dof : idx + 2*n_dof]
        
        R[idx : idx + n_dof] = (K - (h*omega)**2 * M) @ A_h + (h * omega * C) @ B_h #terms in cos
        R[idx + n_dof : idx + 2*n_dof] = -(h * omega * C) @ A_h + (K - (h*omega)**2 * M) @ B_h #terms in sin
        idx += 2 * n_dof
        
    F_nl_freq = aft_procedure(X, omega)
    
    return R + F_nl_freq - F_ext_freq 

#NLFR, sinus sweep
def run_sequential_continuation(omega_span, force_amplitude):
  
    size_X = n_dof * (2 * H + 1)
    X_current = np.zeros(size_X) #initial guess: at rest
 
    F_ext_freq = np.zeros(size_X)
    F_ext_freq[n_dof : 2*n_dof] = np.array([force_amplitude, 0]) #Ex: force sur DDL 1, 0 sur DDL 2
    
    amplitude_history_dof1 = []
    amplitude_history_dof2 = []
    
    print(f"Début de la continuation pour F = {force_amplitude}...")
    for omega in omega_span:
        #Newton-Raphson
        sol = root(hbm_residuals, X_current, args=(omega, F_ext_freq), method='lm') #Levenberg-Marquardt, adapts coeff of Fourier until R = 0
        
        if sol.success:
            X_current = sol.x #current solution becomes the next initial guess
            
            A1 = X_current[n_dof : 2*n_dof]
            B1 = X_current[2*n_dof : 3*n_dof]
            amp_dof1 = np.sqrt(A1[0]**2 + B1[0]**2)
            amp_dof2 = np.sqrt(A1[1]**2 + B1[1]**2)
            
            amplitude_history_dof1.append(amp_dof1)
            amplitude_history_dof2.append(amp_dof2)
        else:
            print(f"Convergence failed at omega = {omega:.2f} rad/s")
            break
            
    return np.array(amplitude_history_dof1), np.array(amplitude_history_dof2)

####################### PLOT #############################

#amplitude vs frequency
def plot_nlfr(omega_span, amp_dof1, amp_dof2, force_amplitude):
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'figure.titlesize': 14})
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    ax1.plot(omega_span[:len(amp_dof1)], amp_dof1, 'b-', linewidth=2, label='HBM (H=1)')
    ax1.set_ylabel('Amplitude Masse 1 ($q_1$)')
    ax1.set_title(f'Réponse en Fréquence Non-Linéaire (NLFR) - Force = {force_amplitude}')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    ax2.plot(omega_span[:len(amp_dof2)], amp_dof2, 'r-', linewidth=2, label='HBM (H=1)')
    ax2.set_xlabel('Fréquence d\'excitation $\omega$ (rad/s)')
    ax2.set_ylabel('Amplitude Masse 2 ($q_2$)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # plt.savefig('nlfr_response.png', dpi=300)
    
    plt.show()

###################### MAIN ###############################
if __name__ == "__main__":
   
    omega_range = np.linspace(50, 250, 200) 
    F_amp = 100 #< 500 statement 
    
    amp1, amp2 = run_sequential_continuation(omega_range, F_amp)
    print("Succeed.")
    plot_nlfr(omega_range, amp1, amp2, F_amp)

    