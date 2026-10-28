import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import spectrogram
from mpl_toolkits.mplot3d import Axes3D

def tfestimate(u, y, dt, Nwin=10):
    """
     h, fr = tfestimate(u, y, dt, Nwin=100)

    Estimate the transfer function from time series of the input u and output y
    sampled with sampling time dt

    Input:
    * u: input signal
    * y: output signal
    * dt: sampling time
    * Nwin: number of windows

    Output:
    * h: H1 estimator of the transfer function
    * fr: frequency vector
    """

    # Sampling frequency
    fs = 1/dt
    # Cross-power and power spectral densities
    fr,suu = sp.signal.csd(u, u, fs, nperseg = np.round(np.size(u)/Nwin))
    fr,suy = sp.signal.csd(u, y, fs, nperseg = np.round(np.size(u)/Nwin))
    # H1-estimator for the transfer function
    h = suy/suu
    return h, fr

def filterAndDifferentiate(x, v, f, t, fnl, w1 =10, w2 =500):
    """
    xf, vf, af, fnlf, fextf = filterAndDifferentiate(x, v, fext, t, fnl, w1 =100, w2 =500)

    From raw time series of the displacement x, velocity v and external force fext sampled
    at times t, returns a filtered version of these signals and of a=d^2x/dt^2 and
    fnl = fnl(v, x).

    Requires
    import numpy as np

    Input:
    * x: displacement signal
    * v: velocity signal
    * fext: external force signal
    * t: time vector
    * fnl: nonlinear force function fnl(v,x)
    * [w1, w2]: bandpass frequencies (filtered signals will have nontrivial content only
    in this frequency band) in rad/s

    Output:
    * xf: filtered version of x
    * vf: filtered version of dx/dt
    * af: filtered version of d^2x/dt^2
    * fnlf: filtered version of fnl(v, x)
    * ff: filtered version of f
    """
    # Input signals in the frequency domain
    X = np.fft.fft(x)
    V = np.fft.fft(v)
    FNL = np.fft.fft(fnl(v, x))
    F = np.fft.fft(f)

    # Sampling time
    dt = t[1] - t[0]

    # Frequency vector
    w = 2* np.pi*np.fft.fftfreq(np.size(X))/dt

    # Filter : keep frequency content in [w1, w2]
    indFiltered = np.logical_or(np.abs(w) <= w1, np.abs(w) >= w2)
    X[indFiltered] = 0
    V[indFiltered] = 0
    FNL[indFiltered] = 0
    F[indFiltered] = 0

    # Compute frequency content of filtered velocity and acceleration
    A = np.complex128(0, 1)*w*V

    # Return to the time domain
    xf = np.real(np.fft.ifft(X))
    fnlf = np.real(np.fft.ifft(FNL))
    vf = np.real(np.fft.ifft(V))
    af = np.real(np.fft.ifft(A))
    fextf = np.real(np.fft.ifft(F))
    
    return xf, vf, af, fnlf, fextf


#=============================
# EX 3 
#=============================
def load_npz(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}

def h_fr(file):
    data = load_npz(file)  
    x = data["x"]
    t = data["t"]
    f = data["f"]

    h, fr = tfestimate(f, x, t[1] - t[0])
    return h,fr

#-----
# Transfer function estimate of the system
#-----

h1,fr1 = h_fr("3_1N.npz")
h10,fr10 = h_fr("3_10N.npz")
h100,fr100 = h_fr("3_100N.npz")

plt.figure(figsize=(8,8/ 1.619))
plt.semilogx(fr1, 20 * np.log10(np.abs(h1)),label="1N")
plt.semilogx(fr10, 20 * np.log10(np.abs(h10)),label = "10N")
plt.semilogx(fr100, 20 * np.log10(np.abs(h100)),label = "100N")

plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$|H(f)|$ [dB]")
plt.legend()
plt.tight_layout()
plt.savefig("transfer_function.pdf",format="pdf")
plt.show()

#=============================
# EX 4
#=============================

#-----
# Sine sweep 
#-----

forward = ["4_1Nf.npz","4_10Nf.npz","4_100Nf.npz"]
backward = ["4_1Nb.npz","4_10Nb.npz","4_100Nb.npz"]
title = ["1N","10N","100N"]

fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

for i in range(3):
    data_f = load_npz(forward[i])
    data_b = load_npz(backward[i])

    ax[i].plot(data_f["omega"], data_f["x"], label="Forward")
    ax[i].plot(data_b["omega"], data_b["x"], label="Backward")
    ax[i].set_ylabel("Displacement x")
    ax[i].set_title(title[i])
    ax[i].legend()

ax[2].set_xlabel("Frequency [Hz]")

plt.tight_layout()
plt.savefig("sinesweep.pdf", bbox_inches="tight")
plt.show()


#=============================
# EX 5
#=============================

#-----
# Spectrogram
#-----
data = load_npz("4_100Nf.npz")
x = np.asarray(data["x"]).squeeze()
t = np.asarray(data["t"]).squeeze()
dt = t[1] - t[0]
fs = 1 / dt

nperseg = 1024
noverlap = 3 * nperseg // 4
fr_spec, t_spec, Sxx = spectrogram(x,fs=fs,window="hann",nperseg=nperseg,mode="magnitude")
Sdb = 20 * np.log10(np.maximum(Sxx / np.max(Sxx),1e-12))

plt.figure(figsize=(10, 6))
levels = np.linspace(-60, 0, 61)
plt.contourf(t_spec, fr_spec, Sdb, levels=levels, extend="min",cmap="magma")
plt.colorbar(label="Relative magnitude [dB]")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Displacement spectrogram — highest forcing amplitude")
plt.tight_layout()
plt.savefig("spectrogram.png")
plt.show()

#-----
# Sinesweep data
#-----

data = load_npz("4_100Nf.npz")

t = np.asarray(data["t"]).squeeze()
x = np.asarray(data["x"]).squeeze()
v = np.asarray(data["v"]).squeeze()
f = np.asarray(data["f"]).squeeze()

fnl_zero = lambda v, x: np.zeros_like(x)

xf, vf, af, fnlf_zero, ff = filterAndDifferentiate(x, v, f, t, fnl_zero, w1=50, w2=300)

step = max(1, len(xf) // 5000)

#-----
# Power law fitting
#-----

max_order = 8
threshold = 0.5  # Minimum NRMSE delta in %

std_af = np.std(af)

X = vf.reshape(-1, 1)
coeff, _, _, _ = np.linalg.lstsq(X, af, rcond=None)
af_pred = X @ coeff
rmse = np.sqrt(np.mean((af - af_pred)**2))

accepted_powers = []
accepted_coeff = None
current_nrmse =  rmse / std_af * 100
tested_powers = []
tested_deltas = []

for power in range(1, max_order + 1):

    powers = accepted_powers + [power]
    X = np.column_stack([xf**p for p in powers] + [vf])

    coeff, _, _, _ = np.linalg.lstsq(X, af, rcond=None)
    af_pred = X @ coeff

    rmse = np.sqrt(np.mean((af - af_pred)**2))
    nrmse = rmse / std_af * 100
    delta = current_nrmse - nrmse

    tested_powers.append(power)
    tested_deltas.append(delta)

    print(f"x^{power}: NRMSE = {nrmse:.2f}%, "f"delta = {delta:.2f}%")

    if delta >= threshold:
        accepted_powers.append(power)
        accepted_coeff = coeff
        current_nrmse = nrmse

print("\nAccepted powers:", accepted_powers)
print(f"Final NRMSE: {current_nrmse:.4f}%")

X = np.column_stack([xf**p for p in accepted_powers] + [vf])
accepted_coeff, _, _, _ = np.linalg.lstsq(X, af, rcond=None)
af_pred = X @ accepted_coeff

# Acceleration surface 
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

step = max(1, len(xf) // 5000)

ax.scatter(xf[::step], vf[::step], af[::step],s=2, alpha=0.2, label="Filtered data")

x_curve = np.linspace(np.min(xf), np.max(xf), 500)
a_curve = np.zeros_like(x_curve)

for i, power in enumerate(accepted_powers):
    a_curve += accepted_coeff[i] * x_curve**power

ax.plot(x_curve, np.zeros_like(x_curve), a_curve,color="red", linewidth=2, label="Fitted curves")
ax.set_xlabel("Displacement x")
ax.set_ylabel("Velocity v")
ax.set_zlabel("Acceleration a")
ax.legend()
plt.tight_layout()
plt.show()

# NRMSE delta 
plt.figure(figsize=(8, 4))
plt.bar(tested_powers, tested_deltas)
plt.xticks(tested_powers)
plt.xlabel("Candidate power")
plt.ylabel(r"$\Delta$ NRMSE (%)")
plt.tight_layout()
plt.show()

# log abs(coefficients)
coeff_x = accepted_coeff[:-1]

plt.figure(figsize=(8, 4))
plt.bar([f"x^{p}" for p in accepted_powers],np.log10(np.maximum(np.abs(coeff_x), 1e-12)))
plt.xlabel("Accepted polynomial term")
plt.ylabel(r"$\log_{10}(|c_p|)$")
plt.tight_layout()
plt.show()
