import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scintp
import clipboard_and_style_sheet
import scipy.constants as sc
import mkl_fft
import scipy.optimize as spo

clipboard_and_style_sheet.style_sheet()


def fft(x, axis=None):
    """
    :param x: array
    :param axis: if dimension is >1, specify which axis along which to perform the ifft
    :return: ifft of x

    calculates the 1D ifft of the numpy array x, if x is not 1D you need to specify the axis
    """

    assert (len(x.shape) == 1) or (isinstance(axis, int)), \
        "if x is not 1D, you need to provide an axis along which to perform the ifft"

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, axis=None):
    """
    :param x: array
    :param axis: if dimension is >1, specify which axis along which to perform the fft
    :return: fft of x

    calculates the 1D fft of the numpy array x, if x is not 1D you need to specify the axis
    """

    assert (len(x.shape) == 1) or (isinstance(axis, int)), \
        "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def Denoise(x, gamma):
    return np.where(abs(x) < gamma, 0, np.sign(x) * abs(x) - gamma * abs(x))


def calculate_spectrogram(Et, omega, t):
    ET2 = np.zeros((len(t), len(omega))).astype(np.complex128)
    ET2[:] = Et
    EW2 = fft(ET2, 1)
    EW2 *= np.exp(1j * np.c_[t] * omega)
    ET2 = ifft(EW2, 1)
    ET2 *= Et
    EW2 = fft(ET2, 1)
    return EW2.__abs__() ** 2


def fun(gamma, args):
    si, sexp, ndpnts, nwpnts = args
    return np.sqrt(np.sum(abs(si - gamma * sexp) ** 2) / (ndpnts * nwpnts))


# ______________________________________________________________________________________________________________________
spectrogram = np.genfromtxt('Data/Nazanins_Data/201118_with all the GLASS+1more').T
BBO_pm_curve = np.genfromtxt("Data/Nazanins_Data/BBO_50um_PhaseMatchingCurve.txt")
wl = np.genfromtxt("Data/Nazanins_Data/Wavelength2.txt")
T_fs = np.arange(-500, 502, 2)
omega = 2 * np.pi * sc.c * 1e-15 / (wl * 1e-9)  # rad / fs

# ______________________________________________________________________________________________________________________
autocorrelation = np.sum(spectrogram, 1)

ind_max = np.argmax(autocorrelation)
center = len(autocorrelation) // 2
ind_keep = min([ind_max, center])

# place T0 at max of the spectrogram
T_fs -= T_fs[ind_max]
T_fs = T_fs[ind_max - ind_keep: ind_max + ind_keep]
spectrogram = spectrogram[ind_max - ind_keep: ind_max + ind_keep]

# number of negative delay data pts = number of positive delay data pts
center = len(T_fs) // 2
ind_zero = np.argmin(abs(T_fs))
ind_throw = (ind_zero - center) * 2
if ind_throw > 0:
    T_fs = T_fs[ind_throw:]
    spectrogram = spectrogram[ind_throw:]
elif ind_throw < 0:
    T_fs = T_fs[:ind_throw]
    spectrogram = spectrogram[:ind_throw]

# ______________________________________________________________________________________________________________________
# filter spectrogram to wavelength of interest
omega1 = 2
omega2 = 3
ind_ll, ind_ul = np.argmin(abs(omega - omega2)), np.argmin(abs(omega - omega1))

spectrogram = spectrogram[:, ind_ll:ind_ul]
spectrogram -= np.mean(spectrogram[0])
wl = wl[ind_ll:ind_ul]
omega = omega[ind_ll:ind_ul]

# ______________________________________________________________________________________________________________________
# define and interpolate spectrogram to simulation grid
NWpnts = 2 ** 11
NDpnts = int(np.floor(len(T_fs) / 2) * 2)
omega_exp = np.linspace(omega[-1], omega[0], NWpnts)
T_fs_exp = np.linspace(T_fs[0], T_fs[-1], NDpnts)

gridded_spectrogram = scintp.interp2d(omega, T_fs, spectrogram, bounds_error=False, kind='linear', fill_value=0)
spectrogram_exp = gridded_spectrogram(omega_exp, T_fs_exp)

# ______________________________________________________________________________________________________________________
# divide out the phase matching curve
wl_pm = BBO_pm_curve[:, 0]
curve_pm = BBO_pm_curve[:, 1]

omega_pm = 2 * np.pi * sc.c * 1e-15 / (wl_pm * 1e-6)
gridded_pm = scintp.interp1d(omega_pm, curve_pm, kind='linear', bounds_error=False, fill_value=1)
curve_pm = gridded_pm(omega_exp)

spectrogram_exp /= curve_pm

# ______________________________________________________________________________________________________________________
# one last background subtraction
bckgnd = np.mean(spectrogram_exp[:, -1])  # use highest frequency line as background
spectrogram_exp -= bckgnd
spectrogram_exp[spectrogram_exp < 0] = 0
spectrogram_exp /= np.max(spectrogram_exp)

# ______________________________________________________________________________________________________________________
# initial guess (time axis is defined by the already-defined frequency axis)
domega = float(np.diff(omega_exp[[0, 1]]))
tnyq = 2 * np.pi / domega
t = np.linspace(-tnyq / 2, tnyq / 2, NWpnts)  # time axis

pti = np.sum(spectrogram_exp, 1).astype(np.complex128)
pti *= np.exp(1j * np.random.uniform(low=0, high=1, size=NDpnts) * np.pi / 8)
gridded_Et = scintp.interp1d(T_fs_exp, pti, kind='linear', bounds_error=False, fill_value=0)
pti = gridded_Et(t)
gti = pti.copy()

# ______________________________________________________________________________________________________________________
Iter = 500
Update = 50

Pti = np.zeros((Iter, NWpnts)).astype(np.complex128)
Error = np.zeros(Iter)

for n in range(Iter):

    j = np.random.permutation(len(T_fs_exp))

    for m in range(NDpnts):
        jiter = j[m]
        t_i = T_fs_exp[jiter]

        gwi = fft(gti)
        gwi *= np.exp(1j * omega_exp * t_i)
        gtishift = ifft(gwi)

        chiti = gtishift * pti
        chiwi = fft(chiti)
        chiwi = Denoise(np.real(chiwi), 1e-3) + 1j * Denoise(np.imag(chiwi), 1e-3)

        chiwiprime = np.sqrt(spectrogram_exp[jiter]) * np.exp(1j * np.arctan2(chiwi.imag, chiwi.real))

        chitiprime = ifft(chiwiprime)

        dchiti = chitiprime - chiti

        pupdate = gtishift.conj() / max(abs(gtishift) ** 2)
        beta_p = np.random.uniform(low=10 / 100, high=30 / 100)
        ptirecon = pti + beta_p + pupdate * dchiti

        ind_t0 = np.argmax(abs(ptirecon))
        ptirecon = np.roll(ptirecon, NWpnts // 2 - ind_t0)

        pti = ptirecon
        gti = ptirecon

    print(n % Update)
    si = calculate_spectrogram(pti, omega_exp, T_fs_exp)
    si /= np.max(si)
    res = spo.minimize(fun, np.array([1]), args=[si, spectrogram_exp, NDpnts, NWpnts])
    err = res.fun

    Error[n] = err
    Pti[n] = pti

    if n % Update == 0:
        plt.close()

        fig, ax = plt.subplots(3, 2)
        ax[0, 0].pcolormesh(T_fs_exp, omega_exp, spectrogram_exp.T, cmap='jet')
        ax[0, 0].set_xlabel("Delay (fs)")
        ax[0, 0].set_ylabel("Frequency (rad/fs)")

        ax[0, 1].pcolormesh(T_fs_exp, omega_exp, si.T, cmap='jet')
        ax[0, 1].set_xlabel("Delay (fs)")
        ax[0, 1].set_ylabel("Frequency (rad/fs)")

        ax[1, 0].plot(t, abs(pti) ** 2)
        ax_ = ax[1, 0].twinx()
        ax_.plot(t, np.unwrap(np.arctan2(pti.imag, pti.real)), color='C1')
        ax[1, 0].set_xlim(-500, 500)
        ax[1, 0].set_xlabel("Time (fs)")
        ax[1, 0].set_ylabel("Amplitude (arb. units)")

        ft = fft(pti)
        ax[1, 1].plot(omega_exp, abs(ft) ** 2)
        ax_ = ax[1, 1].twinx()
        ax_.plot(omega_exp, np.unwrap(np.arctan2(ft.imag, ft.real)), color='C1')
        ax[1, 1].set_xlabel("Frequency (rad/fs)")
        ax[1, 1].set_ylabel("Amplitude (arb. units)")

        ax[2, 0].plot(Error[:n])
        ax[2, 0].set_xlabel("Iteration")
        ax[2, 0].set_ylabel("Error")

        plt.pause(.2)
