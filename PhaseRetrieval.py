import numpy as np
import scipy.constants as sc
import BBO as BBO
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.interpolate as spi
from scipy.integrate import simps
import matplotlib.pyplot as plt
import PullDataFromOSA as osa
import clipboard_and_style_sheet

bbo = BBO.BBOSHG()


def normalize(vec):
    """
    normalize a numpy array
    """
    return vec / np.max(abs(vec))


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


class Retrieval:
    def __init__(self):
        self._exp_T_fs = None
        self._exp_wl_nm = None
        self._data = None
        self._interp_data = None

        self.pulse = fpn.Pulse(T0_ps=0.02,
                               center_wavelength_nm=1560.0,
                               time_window_ps=10,
                               NPTS=2 ** 12)

    def load_data(self, path_to_data):
        data = np.genfromtxt(path_to_data)
        self._exp_T_fs = data[:, 0][1:]
        self._exp_wl_nm = data[0][1:]
        self._data = data[:, 1:][1:]

        # center T0
        ind_max = np.unravel_index(np.argmax(data), data.shape)[0]
        self._data = np.roll(data, -(ind_max - len(data) // 2), axis=0)

    def interpolate_data_to_sim_grid(self):
        self._interp_data = interpolate_spctgm_to_grid(F_mks_input=self.exp_F_mks,
                                                       F_mks_output=self.pulse.F_mks * 2,
                                                       T_fs_input=self.exp_T_fs,
                                                       T_fs_output=self.exp_T_fs,
                                                       spctgm=self.data)

    @property
    def exp_T_fs(self):
        if self._exp_T_fs is None:
            raise ValueError("no data loaded yet")
        return self._exp_T_fs

    @property
    def exp_wl_nm(self):
        if self._exp_wl_nm is None:
            raise ValueError("no data loaded yet")
        return self._exp_wl_nm

    @property
    def data(self):
        if self._data is None:
            raise ValueError("no data loaded yet")
        return self._data

    @property
    def exp_F_mks(self):
        return sc.c / (self.exp_wl_nm * 1e-9)

    @property
    def interp_data(self):
        if self._interp_data is None:
            raise ValueError("no data loaded yet")
        return self._interp_data

    def correct_for_phase_match(self, length_um=50.,
                                theta_pm_rad=bbo.phase_match_angle_rad(1.55),
                                alpha_rad=np.arctan(.25 / 2)):

        R = bbo.R(wl_um=self.exp_wl_nm * 1e-3 * 2,  # fundamental wavelength
                  length_um=length_um,  # crystal thickness
                  theta_pm_rad=theta_pm_rad,
                  alpha_rad=alpha_rad)

        ind = (self.exp_wl_nm > 500).nonzero()[0]
        self.data[:, ind] /= R[ind]

    def scale_initial_pwr_to_spctgm(self):
        self.pulse.set_AT(scale_field_to_spctgm(self.pulse.AT, self.interp_data))
