import gc
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


def shift1D(AT, AW, dT_fs, pulse_ref):
    pulse_ref: fpn.Pulse

    AW[:] *= np.exp(1j * pulse_ref.V_THz[:] * dT_fs[:] * 1e-3)
    ifft(AT, AW)


def shift2D(AT2D_to_shift, AW2D_to_shift, phase2D, dT_fs_vec, pulse_ref):
    pulse_ref: fpn.Pulse

    phase2D[:] = pulse_ref.V_THz[:]
    phase2D[:] *= 1j * dT_fs_vec[:, np.newaxis] * 1e-3
    phase2D[:] = np.exp(phase2D[:])

    AW2D_to_shift[:] *= phase2D[:]
    ifft(AT2D_to_shift, AW2D_to_shift, axis=1)


def calculate_spctgm(AT2D, AT2D_to_shift, AW2D_to_shift, spctgm_to_calc_Tdomain,
                     spctgm_to_calc_Wdomain, phase2D, dT_fs_vec, pulse_ref):
    shift2D(AT2D_to_shift, AW2D_to_shift, phase2D, dT_fs_vec, pulse_ref)

    spctgm_to_calc_Tdomain[:] = AT2D[:] * AT2D_to_shift[:]
    fft(spctgm_to_calc_Tdomain, spctgm_to_calc_Wdomain, axis=1)
    spctgm_to_calc_Wdomain[:] **= 2.


def calculate_error(AT2D, AT2D_to_shift, AW2D_to_shift, spctgm_to_calc_Tdomain,
                    spctgm_to_calc_Wdomain, phase2D, dT_fs_vec, pulse_ref, spctgm_ref):
    calculate_spctgm(AT2D, AT2D_to_shift, AW2D_to_shift, spctgm_to_calc_Tdomain,
                     spctgm_to_calc_Wdomain, phase2D, dT_fs_vec, pulse_ref)

    num = np.sqrt(np.sum((spctgm_to_calc_Wdomain - spctgm_ref) ** 2))
    denom = np.sqrt(np.sum(spctgm_ref ** 2))
    return num / denom


def fft(x, xw, axis=None):
    if axis is None:
        xw[:] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))
    else:
        xw[:] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, xw, axis=None):
    if axis is None:
        x[:] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(xw)))
    else:
        x[:] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(xw, axes=axis), axis=axis), axes=axis)


def scale_field_to_spctgm(AT, spctgm):
    """
    :param AT: 1D complex array, electric-field in time domain
    :param spctgm: reference spectrogram
    :return: 1D complex arrray, AT scaled to the correct power for the reference spectrogram
    """

    AW2 = np.zeros(AT.shape, AT.dtype)
    fft(AT ** 2, AW2)
    power_AW2 = simps(abs(AW2) ** 2)
    spctgm_fftshift = np.fft.ifftshift(spctgm, axes=0)
    power_spctgm = simps(spctgm_fftshift[0])
    scale_power = (power_spctgm / power_AW2) ** 0.25
    return AT * scale_power


def interpolate_spctgm_to_grid(F_mks_input, F_mks_output, T_fs_input, T_fs_output, spctgm):
    gridded = spi.interp2d(F_mks_input, T_fs_input, spctgm)
    return gridded(F_mks_output, T_fs_output)


class Retrieval:
    def __init__(self, maxiter=100):
        self._exp_T_fs = None
        self._exp_wl_nm = None
        self._data = None
        self._interp_data = None
        self.maxiter = maxiter

        self.corrected_for_phase_matching = False

        self.pulse = fpn.Pulse(T0_ps=0.02,
                               center_wavelength_nm=1560.0,
                               time_window_ps=10,
                               NPTS=2 ** 12)

        self.E_j = np.zeros(self.pulse.AT.shape, dtype=self.pulse.AT.dtype)
        self.Eshift_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.corr1 = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.corr2 = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.psi_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.psiPrime_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.phi_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.phase = np.zeros(self.E_j.shape)
        self.amp = np.zeros(self.E_j.shape)
        self.error = np.zeros(maxiter)
        self.Output_Ej = np.zeros((maxiter, len(self.E_j)), dtype=self.E_j.dtype)

        self._rng = np.random.default_rng()

    def load_data(self, path_to_data):
        data = np.genfromtxt(path_to_data)
        self._exp_T_fs = data[:, 0][1:]
        self._exp_wl_nm = data[0][1:]
        self._data = data[:, 1:][1:]

        # center T0
        ind_max = np.unravel_index(np.argmax(data), data.shape)[0]
        self._data = np.roll(data, -(ind_max - len(data) // 2), axis=0)

        # reset certain variables: have not yet corrected for phase matching
        # make sure to re-interpolate the data to the sim grid
        self.corrected_for_phase_matching = False
        del self._interp_data
        gc.collect()
        self._interp_data = None

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

        self.corrected_for_phase_matching = True

    def scale_initial_pwr_to_spctgm(self, corr_for_pm=True,
                                    start_time=None,
                                    end_time=None):

        if corr_for_pm:
            # make sure to correct for phase matcing
            if not self.corrected_for_phase_matching:
                self.correct_for_phase_match()

        # make sure to have interpolated the data to the simulation grid
        if self._interp_data is None:
            self.interpolate_data_to_sim_grid()

        # scale the pulse power to correspond to the spectrogram (very important!)
        self.pulse.set_AT(scale_field_to_spctgm(self.pulse.AT, self.interp_data))

        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = self.exp_T_fs[-1]

        ind_start = np.argmin((self.exp_T_fs - start_time) ** 2)
        ind_end = np.argmin((self.exp_T_fs - end_time) ** 2)

        delay_time = self.exp_T_fs[ind_start:ind_end]
        time_order = np.array([*zip(delay_time, np.arange(ind_start, ind_end))])

        self.E_j[:] = self.pulse.AT[:]
        # print("initial error:", calculate_error(self.E_j,
        #                                         delay_time,
        #                                         self.pulse,
        #                                         self.interp_data[ind_start:ind_end]))
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ind_wl = (self.pulse.wl_um > 0).nonzero()[0]
        # for i in range(self.maxiter):
        #     self._rng.shuffle(time_order, axis=0)
        #     alpha = self._rng.uniform(low=0.1, high=0.5)
        #     for dt, j in time_order:
        #         pass
