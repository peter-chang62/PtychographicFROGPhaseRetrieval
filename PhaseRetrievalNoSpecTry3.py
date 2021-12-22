import numpy as np
import scipy.constants as sc
import BBO as BBO
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.interpolate as spi
from scipy.integrate import simps
import pypret as pyp
import matplotlib.pyplot as plt


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


def scale_field_to_spctgm(AT, spctgm):
    """
    :param AT: 1D complex array, electric-field in time domain
    :param spctgm: reference spectrogram
    :return: 1D complex arrray, AT scaled to the correct power for the reference spectrogram
    """

    AW2 = fft(AT ** 2)
    power_AW2 = simps(abs(AW2) ** 2)
    spctgm_fftshift = np.fft.ifftshift(spctgm, axes=0)
    power_spctgm = simps(spctgm_fftshift[0])
    scale_power = (power_spctgm / power_AW2) ** 0.25
    return AT * scale_power


def interpolate_spctgm_to_grid(F_mks_input, F_mks_output, T_fs_input, T_fs_output, spctgm):
    gridded = spi.interp2d(F_mks_input, T_fs_input, spctgm)
    return gridded(F_mks_output, T_fs_output)


# %% Load the Data
DATA = np.genfromtxt("TestData/new_alignment_method.txt")

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
                  NPTS=2 ** 12)

# %% interpolate experimental data onto the simulation grid
spctgm = interpolate_spctgm_to_grid(F_mks_input=F_mks,
                                    F_mks_output=pulse.F_mks,
                                    T_fs_input=T_fs,
                                    T_fs_output=T_fs,
                                    spctgm=dataCorrected)

# %% fftshifted spectrogram
spctgm_fftshift = np.fft.ifftshift(spctgm, axes=0)

# %% if the spectrogram is to be replicated,
# the power needs to match, so scale the pulse field accordingly
pulse.set_AT(scale_field_to_spctgm(pulse.AT, spctgm))

# %% phase retrieval
