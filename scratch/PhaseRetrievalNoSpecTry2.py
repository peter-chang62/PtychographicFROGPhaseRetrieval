import numpy as np
import scipy.constants as sc
import BBO as BBO
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.interpolate as spi
import pypret as pyp
import matplotlib.pyplot as plt
from scipy.integrate import simps


def normalize(vec):
    """
    normalize a numpy array
    """
    return vec / np.max(abs(vec))


def threshold_operation(x, threshold):
    """
    threshold operation from paper, quite close to just setting whatever is below a certain threshold in a numpy
    array to zero
    """

    return np.where(x < threshold, 0, x - threshold * np.sign(x))


def shift1D(AT, dT_fs, pulse_ref):
    """
    :param AT: complex 1D array, electric field in time domain
    :param dT_fs: float, time shift in femtoseconds
    :param pulse_ref: reference pulse for the frequency axis
    :return: shifted electric field in time domain
    """

    pulse_ref: fpn.Pulse
    AW = fft(AT)

    dT_ps = dT_fs * 1e-3
    AW *= np.exp(1j * pulse_ref.V_THz * dT_ps)
    return ifft(AW)


def shift2D(AT2D, dT_fs_vec, pulse_ref):
    """
    :param AT2D: complex 2D array, electric fields in time domain are row indexed
    :param dT_fs_vec: 1D numpy array, time shift in femtoseconds for each E-field (row) in AT2D
    :param pulse_ref: reference pulse for the frequency axis
    :return: shifted AT2D, each E-field (row) has been shifted by amounts specified in dT_fs_vec
    """

    pulse_ref: fpn.Pulse
    AW2D = fft(AT2D, axis=1)
    dT_ps_vec = dT_fs_vec * 1e-3

    phase = np.zeros(AW2D.shape, dtype=np.complex128)
    phase[:] = pulse_ref.V_THz
    phase *= 1j * dT_ps_vec[:, np.newaxis]
    phase = np.exp(phase)

    AW2D *= phase
    return ifft(AW2D, axis=1)


def calculate_spctgm(AT, dT_fs_vec, pulse_ref):
    """
    :param AT: 1D complex array, electric-field in time domain
    :param dT_fs_vec: delay time axis in femtoseconds
    :param pulse_ref: reference pulse for frequency axis
    :return: calculated spectrogram, time delay is row indexed, and frequency is column indexed
    """

    AT2D = np.zeros((len(dT_fs_vec), len(AT)), dtype=np.complex128)
    AT2D[:] = AT
    AT2D_shift = shift2D(AT2D, dT_fs_vec, pulse_ref)

    spectrogram = AT2D * AT2D_shift
    spectrogram = abs(fft(spectrogram, axis=1)) ** 2
    return spectrogram


def calculate_error(AT, dT_fs_vec, pulse_ref, spctgm_ref):
    """
    :param AT: 1D complex array, electric-field in time domain
    :param dT_fs_vec: 1D array, delay time axis in femtoseconds
    :param pulse_ref: reference pulse for frequency domain
    :param spctgm_ref: reference spectrogram to compare with the calculated spectrogram
    :return: float, error
    """

    calc_spctgm = calculate_spctgm(AT, dT_fs_vec, pulse_ref)
    num = np.sqrt(np.sum((calc_spctgm - spctgm_ref) ** 2))
    denom = np.sqrt(np.sum(abs(spctgm_ref) ** 2))
    return num / denom


def fft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


# %% Load the Data
DATA = np.genfromtxt("../TestData/new_alignment_method.txt")

# %% time and frequency axis
T_fs = DATA[:, 0][1:]
wl_nm = DATA[0][1:]
data = DATA[:, 1:][1:]

# %% useful variables
F_mks = sc.c / (wl_nm * 1e-9)
dT_fs = np.mean(np.diff(T_fs))

# %% center T0
ind_max = np.unravel_index(np.argmax(data), data.shape)[0]
data = np.roll(data, -(ind_max - len(data) // 2), axis=0)

# %% phasematching
bbo = BBO.BBOSHG()
R = bbo.R(wl_um=wl_nm * 1e-3 * 2,  # fundamental wavelength
          length_um=50.,  # crystal thickness
          theta_pm_rad=bbo.phase_match_angle_rad(1.55),  # crystal was phasematched for 1550 nm
          alpha_rad=np.arctan(.25 / 2)
          )

# %% correct for phasematching
ind500nm = (wl_nm > 500).nonzero()[0]
dataCorrected = np.copy(data)
dataCorrected[:, ind500nm] /= R[ind500nm]

# %% initial guess and set the simulation grid
pulse = fpn.Pulse(T0_ps=0.02,
                  center_wavelength_nm=1560,
                  time_window_ps=10,
                  NPTS=2 ** 14)

# %% interpolate experimental data onto the simulation grid
gridded = spi.interp2d(F_mks, T_fs, dataCorrected)
spctgm = gridded(pulse.F_mks * 2, T_fs)

# %% fftshifted spectrogram
spctgm_fftshift = np.fft.ifftshift(spctgm, axes=0)

# %% if the spectrogram is to be replicated,
# the power needs to match, so scale the pulse field accordingly
AW2 = fft(pulse.AT ** 2)
power_AW2 = simps(abs(AW2) ** 2)
power_spctgm = simps(spctgm_fftshift[0])
scale_power = (power_spctgm / power_AW2) ** .25
AT = pulse.AT * scale_power

# %% phase retrieval
scale_dT = dT_fs / (pulse.dT_ps * 1e3)
rng = np.random.default_rng()

ATshift = np.zeros(AT.shape, dtype=AT.dtype)
Psi = np.zeros(AT.shape, dtype=AT.dtype)
PsiPrime = np.zeros(AT.shape, dtype=AT.dtype)
Phi = np.zeros(AT.shape, dtype=AT.dtype)
phase = np.zeros(len(AT))
amp = np.zeros(len(AT))

result = []
error = []

maxiter = 100
print("initial error:", calculate_error(AT, T_fs, pulse, spctgm))
for it in range(maxiter):
    time_order = np.arange(0, 200, 1)
    rng.shuffle(time_order)
    alpha = rng.uniform(low=.1, high=.5)

    for roll_spctgm in time_order:
        roll_pulse = round(roll_spctgm * scale_dT)

        ATshift[:] = np.roll(AT, roll_pulse)
        Psi[:] = AT * ATshift
        Phi[:] = fft(Psi)

        phase[:] = np.arctan2(Phi.imag, Phi.real)
        amp[:] = np.sqrt(np.roll(spctgm_fftshift, roll_spctgm, axis=0)[0])
        Phi[:] = amp * np.exp(1j * phase)
        Phi[:] = threshold_operation(Phi, 1e-4)
        PsiPrime[:] = ifft(Phi)

        AT[:] += alpha * (ATshift.conj() / max(abs(ATshift) ** 2)) * (PsiPrime - Psi)

        AT[:] += np.roll((AT.conj() / max(abs(AT) ** 2)) * (PsiPrime - Psi), -roll_pulse)

    err = calculate_error(AT, T_fs, pulse, spctgm)
    print(it, err)
    result.append(AT)
    error.append(err)

error = np.array(error)
result = np.array(result)
