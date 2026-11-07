import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import h5py

def filterAndDifferentiate(x, v, f, t, fnl, w1 =5, w2 =500):
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


def inspect_mat(filename):
    with h5py.File(filename, 'r') as f:
        def print_name(name, obj):
            print(name, obj.shape if isinstance(obj, h5py.Dataset) else "Group")
        f.visititems(print_name)


def loadmat(filename):
    with h5py.File(filename, 'r') as f:
        t = np.array(f['data/t']).flatten()
        x = np.array(f['data/x'])
        xd = np.array(f['data/xd'])
        xdd = np.array(f['data/xdd'])
        pex = np.array(f['data/pex']).flatten()

    return t, x, xd, xdd, pex

t,x,xd,xdd,pex = loadmat("NI2D_test/G7_exp1_3.mat")
f = np.linspace(6,40,len(t))
plt.plot(f,x)
plt.show()