import gc
import numpy as np
import scipy.constants as sc
import BBO as BBO
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.interpolate as spi
from scipy.integrate import simps
import matplotlib.pyplot as plt
import pyfftw
import PullDataFromOSA as osa
import copy
import clipboard_and_style_sheet
from scipy.signal.windows import tukey

bbo = BBO.BBOSHG()


def normalize(vec):
    """
    normalize a numpy array
    """
    return vec / np.max(abs(vec))


def plot_ret_results(AT, dT_fs_vec, pulse_ref, spctgm_ref, filter_um=None, plot_um=(1, 2)):
    """
    1. calculates the spectrogram corresponding to AT using: AT, dT_fs_vec, and pulse_ref
    2. calculates the error with respect to the reference spectrogram: spctgm_ref, using either
    the entire bandwidth, or a filtered portion via filter_um.
    *Note that spctgm_ref* must match the time and frequency axis of AT

    The results are then plotted

    :param AT: 1D array
    :param dT_fs_vec: 1D array
    :param pulse_ref: PyNLO pulse instance
    :param spctgm_ref: 2D array
    :param filter_um: default None, otherwise is a list
    :param plot_um: wavelength axis limits for the plotting
    :return: calculated spectrogram, the figure and the axes of the plot
    """

    pulse_ref: fpn.Pulse

    if filter_um is not None:
        wl = pulse_ref.wl_um
        ll, ul = filter_um
        ind_filter = (np.logical_and(wl > ll, wl < ul)).nonzero()[0]
    else:
        ind_filter = np.arange(len(pulse_ref.wl_um))

    spctgm_calc = calculate_spctgm(AT, dT_fs_vec, pulse_ref)
    num = np.sqrt(np.sum((spctgm_calc[:, ind_filter] - spctgm_ref[:, ind_filter]) ** 2))
    denom = np.sqrt(np.sum(abs(spctgm_ref[:, ind_filter]) ** 2))
    error = num / denom

    indwl = np.logical_and(pulse_ref.wl_um > plot_um[0], pulse_ref.wl_um < plot_um[-1]).nonzero()[0]

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    pulse_ref: fpn.Pulse
    axs[0].plot(pulse_ref.T_ps, normalize(abs(AT) ** 2))
    AW = fft(AT)
    axs[1].plot(pulse_ref.wl_um[indwl], normalize(abs(AW[indwl]) ** 2))
    ax = axs[1].twinx()
    phase = np.unwrap(np.arctan2(AW[indwl].imag, AW[indwl].real)) * 180 / np.pi  # convert to deg.
    ax.plot(pulse_ref.wl_um[indwl], phase, 'C1')
    axs[2].pcolormesh(dT_fs_vec, pulse_ref.wl_um[indwl], spctgm_ref[:, indwl].T, cmap='jet')
    axs[3].pcolormesh(dT_fs_vec, pulse_ref.wl_um[indwl], spctgm_calc[:, indwl].T, cmap='jet')
    axs[0].set_xlabel("T (ps)")
    axs[1].set_xlabel("$\\mathrm{\\mu m}$")
    axs[2].set_xlabel("T (fs)")
    axs[2].set_ylabel("wavelength ($\\mathrm{\\mu m}$)")
    axs[3].set_xlabel("T (fs)")
    axs[3].set_ylabel("wavelength ($\\mathrm{\\mu m}$)")
    axs[2].set_title("Experiment")
    axs[3].set_title("Retrieved")
    fig.suptitle("Error: " + '%.3f' % error)

    return spctgm_calc, fig, axs


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


def interpolate_spctgm_to_grid(F_mks_input, F_mks_output, T_fs_input, T_fs_output, spctgm):
    gridded = spi.interp2d(F_mks_input, T_fs_input, spctgm, bounds_error=False, fill_value=0.0)
    return gridded(F_mks_output, T_fs_output)


# I've verified that the math here is the same as Sidorenko's matlab code,
# and that their matlab code matches what is stated in their paper
def denoise(x, gamma):
    """
    :param x: 1D array
    :param gamma: threshold (float)
    :return: 0 if abs(x) < gamma, otherwise returns x - gamma * sign(x)
    """
    return np.where(abs(x) < gamma, 0.0, np.sign(x) * (abs(x) - gamma))


# brick wall band pass filter convenience function
def apply_filter(AW, ll_um, ul_um, pulse_ref, fftshift=False):
    """
    applies a brick wall band pass filter to AW based on ll_um, and ul_um, and pulse_ref
    which is used to get a frequency axis

    :param AW: array with frequency on first (0th) axis
    :param ll_um: shorter wavelength limit (float)
    :param ul_um: longer wavelength limit (float)
    :param pulse_ref: reference pulse (used for wavelength axis)
    :param fftshift: is the AW array fftshifted? (bool)
    :return: None
    """
    pusle_ref: fpn.Pulse

    if fftshift:
        wl = np.fft.fftshift(pulse_ref.wl_um)
    else:
        wl = pulse_ref.wl_um

    ind_ll = (wl < ll_um).nonzero()[0]
    ind_ul = (wl > ul_um).nonzero()[0]

    AW[ind_ll] = 0.0
    AW[ind_ul] = 0.0


class Retrieval:
    """
    This is the main pulse retrieval class. The reason it is so long is because the fft's are done using
    pyfftw, which requires some bookkeeping
    """

    def __init__(self, maxiter=100, time_window_ps=10., NPTS=2 ** 12, center_wavelength_nm=1560.):
        """
        :param maxiter: maximum number of iterations for phase retrieval
        :param time_window_ps: time window to use for the pulse field
        :param NPTS: number of points in the pulse field array,
            together with time_window_ps, this sets the frequency axis
        :param center_wavelength_nm: center wavelength, you should give a number such that the initial guess
            at least has power where they should be non-zero power spectral density, from there if you're off
            the retrieved pulse field in the frequency domain will not be centered in the array
        """

        # internal variables to be used later
        self._exp_T_fs = None
        self._exp_wl_nm = None
        self._data = None
        self._interp_data = None
        self.maxiter = maxiter

        # used to track if the user has divided out the phase-matching curve yet
        self.corrected_for_phase_matching = False

        # pulse instance taken from PyNLO used to set the initial guess for phase-retrieval
        self.pulse = fpn.Pulse(T0_ps=0.02,
                               center_wavelength_nm=center_wavelength_nm,
                               time_window_ps=time_window_ps,
                               NPTS=NPTS)

        # random number generator
        self._rng = np.random.default_rng()

        # soft threshold used for de-noising
        self.gamma = 1e-3  # does not appear to sensitive whether it's 1e-3 or down to 1e-6

    # __________________________________________________________________________________________________________________
    # In self.shift1D, self.shift2D, self.calculate_spctgm, and self.calculate_error, all the input arrays are assumed
    # to be fftshifted! This is all taken care of in self.retrieve
    #
    # If you wish to do something similar to the above functions, I would suggest you use the ones defined
    # in the outer scope. Everything is there except shift1D
    # __________________________________________________________________________________________________________________

    def shift1D(self, AT_to_shift, AW, AW_to_shift, dT_fs, V_THz):
        """
        1. AW_to_shift is multiplied by a linear phase calculated from V_THz and dT_fs
        2. AW_to_shift is then input to self.ifft() such that AT_to_shift() becomes the time shifted field

        :param AT_to_shift: 1D array
        :param AW: 1D array
        :param AW_to_shift: 1D array
        :param dT_fs: float
        :param V_THz: 1D array
        """
        AW_to_shift[:] = AW[:] * np.exp(1j * V_THz[:] * dT_fs * 1e-3)

        self.fft_output[:] = AW_to_shift[:]
        AT_to_shift[:] = self.ifft()

    def shift2D(self, AW2D_to_shift, phase2D, dT_fs_vec, V_THz):
        """
        1. AW2D_to_shift is multiplied by a 2D array of linear phases calculated from V_THz and dT_fs_vec.
        The 2D array of linear phases is used to populate the phase2D array, so phase2D can be an array of anything
        so long as it matches the data type
        2. AW2D_to_shift is then input to self.ifft2() such that self.AT2D_to_shift becomes a 2D array of
        time shifted fields

        :param AW2D_to_shift: 2D array
        :param phase2D: 2D array
        :param dT_fs_vec: 1D array
        :param V_THz: 1D array
        """
        phase2D[:] = V_THz[:]
        phase2D[:] *= 1j * dT_fs_vec[:, np.newaxis] * 1e-3
        phase2D[:] = np.exp(phase2D[:])

        AW2D_to_shift[:] *= phase2D[:]

        self.fft2_output[:] = AW2D_to_shift[:]
        self.AT2D_to_shift[:] = self.ifft2()

    def calculate_spctgm(self, AT2D, AT2D_to_shift, AW2D_to_shift, spctgm_to_calc_Tdomain,
                         spctgm_to_calc_Wdomain, phase2D, dT_fs_vec, V_THz):
        """
        1. AW2D_to_shift and AT2D_to_shift, phase2D, dT_fs_vec and V_THz are used to calculate the time shifted
        fields via self.shift2D
        2. AT2D_to_shift is multiplied with AT2D to get the complex spectrogram in the time domain
        3. the spectrogram in the time domain is then input to self.fft2() to calculate the complex
        frequency domain spectrogram
        4. the spectrogram in the frequency domain is then multiplied by its conjugate
        to get the modulus squared

        :param AT2D: 2D array
        :param AT2D_to_shift: 2D array
        :param AW2D_to_shift: 2D array
        :param spctgm_to_calc_Tdomain: 2D array
        :param spctgm_to_calc_Wdomain: 2D array
        :param phase2D: 2D array
        :param dT_fs_vec: 1D array
        :param V_THz: 1D array
        """
        self.shift2D(AW2D_to_shift, phase2D, dT_fs_vec, V_THz)

        spctgm_to_calc_Tdomain[:] = AT2D[:] * AT2D_to_shift[:]

        self.fft2_input[:] = spctgm_to_calc_Tdomain[:]
        self.spctgm_to_calc_Wdomain[:] = self.fft2()

        spctgm_to_calc_Wdomain[:] *= spctgm_to_calc_Wdomain.conj()

    def calculate_error(self, AT2D, AT2D_to_shift, AW2D_to_shift, spctgm_to_calc_Tdomain,
                        spctgm_to_calc_Wdomain, phase2D, dT_fs_vec, V_THz, spctgm_ref,
                        ind_filter):
        """
        1. Calculates the spectrogram using AT2D, AT2D_to_shift, AW2D_to_shift, spctgm_to_calc_Tdomain,
        spctgm_to_calc_Wdomain, phase2D, dT_fs_vec, and V_THz using self.calculate_spctgm
        2. This is then used together with spctgm_ref to calculate sqrt(delta^2 / ref_spctgm^2)

        The user can specify the indices to use for calculating the error (for example, if you
        were to spectrally filter) Remember the input arrays are fftshifted when passing indices!

        :param AT2D: 2D array
        :param AT2D_to_shift: 2D array
        :param AW2D_to_shift: 2D array
        :param spctgm_to_calc_Tdomain: 2D array
        :param spctgm_to_calc_Wdomain: 2D array
        :param phase2D: 2D array
        :param dT_fs_vec: 1D array
        :param V_THz: 1D array
        :param spctgm_ref: 2D array
        :param ind_filter: 1D array
        :return:
        """

        self.calculate_spctgm(AT2D, AT2D_to_shift, AW2D_to_shift, spctgm_to_calc_Tdomain,
                              spctgm_to_calc_Wdomain, phase2D, dT_fs_vec, V_THz)

        num = np.sqrt(np.sum((spctgm_to_calc_Wdomain.real[:, ind_filter] - spctgm_ref[:, ind_filter]) ** 2))
        denom = np.sqrt(np.sum(spctgm_ref[:, ind_filter] ** 2))
        return num / denom

    def scale_field_to_spctgm(self, AT, spctgm):
        """
        :param AT: 1D complex array, electric-field in time domain
        :param spctgm: reference spectrogram
        :return: 1D complex arrray, AT scaled to the correct power for the reference spectrogram
        """

        # we'll compare integrated power at the center of the spectrogram,
        # which corresponds to a time delay of 0 (so AT(t) * AT(t - 0) = AT^2)
        AW2 = np.zeros(AT.shape, AT.dtype)
        self.fft_input[:] = AT[:] ** 2
        AW2[:] = self.fft()

        power_AW2 = simps(abs(AW2) ** 2)
        spctgm_fftshift = np.fft.ifftshift(spctgm, axes=0)
        power_spctgm = simps(spctgm_fftshift[0])
        scale_power = (power_spctgm / power_AW2) ** 0.25
        return AT * scale_power

    def load_data(self, path_to_data):
        """
        loads the data file and sets relevant parameters

        :param path_to_data: path to data file (string)
        """

        data = np.genfromtxt(path_to_data)
        self._exp_T_fs = data[:, 0][1:]
        self._exp_wl_nm = data[0][1:]
        self._data = data[:, 1:][1:]

        # center T0
        integral = simps(self._data, axis=1)
        ind_max = np.argmax(integral)
        ind_center = np.argmin(self.exp_T_fs ** 2)
        if ind_max < ind_center:
            diff = ind_center - ind_max
            self._data = self._data[:-diff]
            self._exp_T_fs = self.exp_T_fs[diff:]

        elif ind_max > ind_center:
            diff = ind_max - ind_center
            self._data = self._data[diff:]
            self._exp_T_fs = self.exp_T_fs[:-diff]

        # reset certain variables: have not yet corrected for phase matching
        # make sure to re-interpolate the data to the sim grid
        self.corrected_for_phase_matching = False
        del self._interp_data
        gc.collect()
        self._interp_data = None

    def setup_retrieval_arrays(self, delay_time):
        """
        initialize all arrays needed for pyfftw

        :param delay_time: 1D array (its length is used to set the size of relevant arrays)
        """

        # 1D arrays
        self.E_j = np.zeros(self.pulse.AT.shape, dtype=self.pulse.AT.dtype)
        self.EW_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.Eshift_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.EWshift_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.corr1 = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.corr2 = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.corr2W = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.psi_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.psiPrime_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.phi_j = np.zeros(self.E_j.shape, dtype=self.E_j.dtype)
        self.phase = np.zeros(self.E_j.shape)
        self.amp = np.zeros(self.E_j.shape)

        self.error = np.zeros(self.maxiter)

        # 2D arrays
        self.Output_Ej = np.zeros((self.maxiter, len(self.E_j)), dtype=self.E_j.dtype)
        self.Output_EWj = np.zeros((self.maxiter, len(self.E_j)), dtype=self.E_j.dtype)

        self.AT2D = np.zeros((len(delay_time), len(self.E_j)), dtype=np.complex128)
        self.AT2D_to_shift = np.zeros((len(delay_time), len(self.E_j)), dtype=np.complex128)
        self.AW2D_to_shift = np.zeros((len(delay_time), len(self.E_j)), dtype=np.complex128)
        self.spctgm_to_calc_Tdomain = np.zeros((len(delay_time), len(self.E_j)), dtype=np.complex128)
        self.spctgm_to_calc_Wdomain = np.zeros((len(delay_time), len(self.E_j)), dtype=np.complex128)
        self.phase2D = np.zeros((len(delay_time), len(self.E_j)), dtype=np.complex128)

        # 1D fft arrays
        self.fft_input = pyfftw.empty_aligned(self.E_j.shape, dtype='complex128')
        self.fft_output = pyfftw.empty_aligned(self.E_j.shape, dtype='complex128')

        # 2D fft arrays
        self.fft2_input = pyfftw.empty_aligned(self.AT2D.shape, dtype='complex128')
        self.fft2_output = pyfftw.empty_aligned(self.AT2D.shape, dtype='complex128')

        # 1D fft
        self.fft = pyfftw.FFTW(self.fft_input, self.fft_output, axes=[0], direction='FFTW_FORWARD',
                               flags=["FFTW_MEASURE"])
        self.ifft = pyfftw.FFTW(self.fft_output, self.fft_input, axes=[0], direction='FFTW_BACKWARD',
                                flags=["FFTW_MEASURE"])

        # 2D fft
        self.fft2 = pyfftw.FFTW(self.fft2_input, self.fft2_output, axes=[1], direction='FFTW_FORWARD',
                                flags=["FFTW_MEASURE"])
        self.ifft2 = pyfftw.FFTW(self.fft2_output, self.fft2_input, axes=[1], direction='FFTW_BACKWARD',
                                 flags=["FFTW_MEASURE"])

    def interpolate_data_to_sim_grid(self):
        """
        """
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
            raise ValueError("no data interpolated onto simulation grid yet")
        return self._interp_data

    def correct_for_phase_match(self, length_um=50.,
                                theta_pm_rad=bbo.phase_match_angle_rad(1.55),
                                alpha_rad=BBO.deg_to_rad(3.5)):

        if self.corrected_for_phase_matching:
            raise RuntimeWarning("already corrected for phase matching!")

        R = bbo.R(wl_um=self.exp_wl_nm * 1e-3 * 2,  # fundamental wavelength
                  length_um=length_um,  # crystal thickness
                  theta_pm_rad=theta_pm_rad,
                  alpha_rad=alpha_rad)

        ind = (self.exp_wl_nm > 440).nonzero()[0]
        # ind = (self.exp_wl_nm > 530).nonzero()[0]
        self.data[:, ind] /= R[ind]

        self.corrected_for_phase_matching = True

    def retrieve(self, corr_for_pm=True,
                 start_time_fs=None,
                 end_time_fs=None,
                 filter_um=None,
                 plot_update=True,
                 plot_wl_um=(1.0, 2.0),
                 initial_guess_T_ps_AT=None,
                 initial_guess_wl_um_AW=None,
                 forbidden_um=None,
                 meas_spectrum_um=None,
                 grad_ramp_for_meas_spectrum=False,
                 i_set_spectrum_to_meas=0,
                 debug_plotting=False):

        """
        :param corr_for_pm: divide out phase matching curve before starting? (bool)
                this assumes a number of default values (phase-matching angle, BBO thickness, and aoi), so
                if you divided out your own phase-matching curve, set this to False
        :param start_time_fs: start time of spectrogram to use for retrieval
        :param end_time_fs: end time of spectrogram to use for retrieval
        :param filter_um: wavelength limits of spectrogram to use (list)
        :param plot_update: plot the retrieval results after each iteration? (bool)
        :param plot_wl_um: if plot_update is True, these are the wavelength limits to use for plotting (list)
        :param initial_guess_T_ps_AT: initial pulse-field in the time domain
        :param initial_guess_wl_um_AW: initial pulse-field in the frequency domain
                (only provide either time or frequency)
        :param forbidden_um: band-pass filter (list): filter the pulse-field after each iteration with a tukey window
                *This is implemented only if the measured spectrum is not provided, otherwise it is ignored!*
        :param meas_spectrum_um: experimental power spectrum passed as (wl_um, spectrum)
                if provided, the power spectrum will be constrained in the retrieval. The default is None
                which does not constrain the power spectrum
        :param grad_ramp_for_meas_spectrum: if True, the power spectrum will be constrained, but will be gradually
                ramped from the current retrieval spectrum to the experimental ones over 10 iterations. The goal
                was to let the retrieved phase sort of "catch up" with the constraint, but doesn't actually seem
                to help ...
        :param i_set_spectrum_to_meas: how many iterations to run before starting to constrain the power spectrum
                (that is the user decides to constrain the power spectrum)
        :param debug_plotting: In addition to plot_update, debug_plotting will plot all the relevant arrays
                used during retrieval, to give the user an idea if his frequency and time bandwidth are sufficient
                to prevent aliasing during iterations
        """

        # _____________________ correct for phase matching _____________________________________________________________
        if corr_for_pm:
            # make sure to correct for phase matching
            if not self.corrected_for_phase_matching:
                self.correct_for_phase_match()

        # _____________________ calculate interpolated spectrogram _____________________________________________________
        # make sure to have interpolated the data to the simulation grid
        if self._interp_data is None:
            self.interpolate_data_to_sim_grid()

        # ___________________ set the start and end time of the spectrogram to be used for retrieval ___________________
        # start_time_fs and end_time_fs will either be a subset of the experimental time axis,
        # or its full extent. If start_time_fs or end_time_fs exceeds the experimental axis, the
        # experimental axis limits will be used instead
        if start_time_fs is None:
            start_time_fs = 0.0
        if end_time_fs is None:
            end_time_fs = self.exp_T_fs[-1]

        # _________________ setting the initial guess for the pulse field ______________________________________________
        # if we are constraining the power spectrum, set the initial guess to the transform limited field
        # calculated from the experimental power spectrum
        if meas_spectrum_um is not None:
            wl_um, spectrum = meas_spectrum_um
            aw = np.sqrt(abs(spectrum))
            self.pulse.set_AW_experiment(wl_um, aw)
            self.pulse.set_AT(ifft(self.pulse.AW))  # pynlo has ifft <-> fft defined in reverse

        else:
            # if no initial guess is provided for the pulse-field, default to setting the time domain to
            # the autocorrelation calculated from the interpolated spectrogram
            if (initial_guess_T_ps_AT is None) and (initial_guess_wl_um_AW is None):
                # default to autocorrelation
                initial_guess_T_ps_AT = np.sum(self._interp_data, axis=1)
                initial_guess_T_ps_AT -= min(initial_guess_T_ps_AT)
                self.pulse.set_AT_experiment(self.exp_T_fs * 1e-3, initial_guess_T_ps_AT)

            # only one of time domain or frequency domain initial guesses can be provided
            elif (initial_guess_T_ps_AT is not None) and (initial_guess_wl_um_AW is not None):
                raise AssertionError("only one of initial_guess_T_fs_AT or initial_guess_wl_um_AW can be defined")

            # time domain initial-guess
            elif initial_guess_T_ps_AT is not None:
                # initial guess generally can be complex
                T_ps, field = initial_guess_T_ps_AT
                self.pulse.set_AT_experiment(T_ps, field)

            # otherwise frequency domain initial-guess
            else:
                wl_um, field = initial_guess_wl_um_AW
                self.pulse.set_AW_experiment(wl_um, field)
                self.pulse.set_AT(ifft(self.pulse.AW))

        # _____________________________ set the frequency range to be used for phases retrieval ________________________
        # the user can set a range of wavelengths to be used for phase retrieval. This is useful if phase-matching
        # bandwidth was an issue. It's important to note that the indexing will be done for fftshifted arrays
        wl_um_fftshift = np.fft.ifftshift(self.pulse.wl_um)
        if filter_um is not None:
            ll_um, ul_um = filter_um
            ind_filter_fftshift = np.logical_and(wl_um_fftshift > ll_um, wl_um_fftshift < ul_um).nonzero()[0]

        else:
            ind_filter_fftshift = np.arange(len(wl_um_fftshift))  # in this case array[ind_filter_fftshift] = array[:]

        # _______________________ the user can decide to filter the retrieved spectrum after each iteration ____________
        # again, it is important to note that the indexing will be done for fftshifted arrays
        if forbidden_um is not None:
            ll_um, ul_um = forbidden_um
            wl_um = self.pulse.wl_um

            ind_allowed = np.logical_and(wl_um >= ll_um, wl_um <= ul_um).nonzero()[0]
            window_forbidden = tukey(len(ind_allowed), alpha=0.25)
            window_forbidden = np.pad(window_forbidden, (ind_allowed[0], len(wl_um) - 1 - ind_allowed[-1]),
                                      constant_values=0.0)

            window_forbidden = np.fft.fftshift(window_forbidden)

        # ______________________________________________________________________________________________________________
        # fftshift everything before fft's are calculated:
        #
        #   1. The initial guess is pulse.AT, everything is calculated off the initial guess so fftshifting this
        #   fftshifts everything that follows
        #
        #   2. The calculated spectrogram will be fftshifted, so the reference spectrogram used in the error
        #   calculation also needs to be fftshifted
        #
        #   3. The calculated spectrogram will be fftshifted, so the reference spectrogram used in the error
        #   calculation also needs to be fftshifted
        #
        #   4. Since the fields are fftshifted, the frequency axis used to calculate time shifted fields also
        #   needs to be fftshifted
        # ______________________________________________________________________________________________________________

        # used for initial guess and subsequent error calculations
        AT0_fftshift = np.fft.ifftshift(self.pulse.AT)
        interp_data_fftshift = np.fft.ifftshift(self._interp_data, axes=1)
        V_THz_fftshift = np.fft.ifftshift(self.pulse.V_THz)

        # spectrogram delay times to iterate over
        ind_start = np.argmin((self.exp_T_fs - start_time_fs) ** 2)
        ind_end = np.argmin((self.exp_T_fs - end_time_fs) ** 2)
        self.delay_time = self.exp_T_fs[ind_start:ind_end]
        time_order = np.array([*zip(self.delay_time, np.arange(ind_start, ind_end))])

        # h_meas_spectrum is either a gradual ramp, or an array of ones, depending
        # on the gradual_ramp flag
        if meas_spectrum_um is not None and grad_ramp_for_meas_spectrum:
            h_meas_spectrum = np.linspace(.001, 1, 10)
            h_meas_spectrum = np.repeat(h_meas_spectrum[:, np.newaxis], 5, 1).flatten()

            self.maxiter = i_set_spectrum_to_meas + len(h_meas_spectrum)
            print(f'maxiter has been adjusted to {self.maxiter}')

        else:
            h_meas_spectrum = np.ones(self.maxiter - i_set_spectrum_to_meas)

        # initialize the arrays to zeros
        self.setup_retrieval_arrays(self.delay_time)

        # scale the pulse power to correspond to the spectrogram (very important!)
        AT0_fftshift[:] = self.scale_field_to_spctgm(AT0_fftshift, interp_data_fftshift)

        # set the electric field initially to the guess electric field
        self.E_j[:] = AT0_fftshift[:]
        self.fft_input[:] = self.E_j[:]
        self.EW_j[:] = self.fft()

        self.AT2D[:] = self.E_j[:]
        self.AT2D_to_shift[:] = self.E_j[:]
        self.AW2D_to_shift[:] = self.EW_j[:]

        if meas_spectrum_um is not None:
            # interpolated experimental power spectrum, already fftshifted
            meas_amp_interp = abs(self.EW_j)

        if plot_update:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax3 = ax2.twinx()
            ind_wl = np.logical_and(self.pulse.wl_um >= plot_wl_um[0], self.pulse.wl_um <= plot_wl_um[-1]).nonzero()[0]

        error = self.calculate_error(self.AT2D,
                                     self.AT2D_to_shift,
                                     self.AW2D_to_shift,
                                     self.spctgm_to_calc_Tdomain,
                                     self.spctgm_to_calc_Wdomain,
                                     self.phase2D,
                                     self.delay_time,
                                     V_THz_fftshift,
                                     interp_data_fftshift[ind_start:ind_end],
                                     ind_filter_fftshift)

        print("initial error:", error)

        if debug_plotting:
            fig_debug, axs_debug = plt.subplots(2, 5)
            axs_debug = axs_debug.flatten()

        for i in range(self.maxiter):
            self._rng.shuffle(time_order, axis=0)
            alpha = self._rng.uniform(low=0.1, high=0.5)

            for n, (dt, j) in enumerate(time_order):
                self.shift1D(self.Eshift_j, self.EW_j, self.EWshift_j, dt, V_THz_fftshift)
                self.psi_j[:] = self.E_j[:] * self.Eshift_j[:]

                self.fft_input[:] = self.psi_j[:]
                self.phi_j[:] = self.fft()

                # ind_filter_fftshift can be used to utilize only a portion of the experimental spectrogram
                # for phase retrieval (spectrally incomplete spectrogram)
                # otherwise, it defaults to utilize the entire bandwidth
                # you might feel the need to use a threshold, e.g. (self.amp > threshold).nonzero()[0]
                # but this should be taken care of already when you call the denoise function
                self.amp[:] = 0
                self.amp[ind_filter_fftshift] = np.sqrt(interp_data_fftshift[int(j), ind_filter_fftshift])
                ind_nonzero = (self.amp > 0).nonzero()[0]

                self.phase[:] = np.arctan2(self.phi_j.imag, self.phi_j.real)
                # only replace known parts of the spectrum
                self.phi_j[ind_nonzero] = self.amp[ind_nonzero] * np.exp(1j * self.phase[ind_nonzero])

                # denoise
                self.phi_j[:] = denoise(self.phi_j.real, self.gamma) + 1j * denoise(self.phi_j.imag, self.gamma)

                self.fft_output[:] = self.phi_j[:]
                self.psiPrime_j[:] = self.ifft()

                self.corr1[:] = alpha * self.Eshift_j.conj() * \
                                (self.psiPrime_j[:] - self.psi_j[:]) / max(abs(self.Eshift_j) ** 2)
                self.corr2[:] = alpha * self.E_j.conj() * \
                                (self.psiPrime_j[:] - self.psi_j[:]) / max(abs(self.E_j) ** 2)

                self.fft_input[:] = self.corr2[:]
                self.corr2W[:] = self.fft()

                self.shift1D(self.corr2, self.corr2W, self.corr2W, -dt, V_THz_fftshift)

                # update time domain
                self.E_j[:] += self.corr1 + self.corr2

                # keep time domain centered, doesn't seem necessary
                ind_max = np.argmax(abs(self.E_j))
                self.E_j[:] = np.roll(self.E_j, -ind_max)

                # update frequency domain
                self.fft_input[:] = self.E_j[:]
                self.EW_j[:] = self.fft()

                if (meas_spectrum_um is not None) and i >= i_set_spectrum_to_meas:
                    # start constraining the spectrum
                    if i == i_set_spectrum_to_meas:
                        starting_field = abs(self.EW_j)
                        # difference b/w experimental spectrum and spectrum at iteration i_set_spectrum_to_meas
                        diff = meas_amp_interp - starting_field
                        h = 0

                    self.amp[:] = starting_field + diff * h_meas_spectrum[h]
                    self.phase[:] = np.arctan2(self.EW_j.imag, self.EW_j.real)
                    self.EW_j[:] = self.amp[:] * np.exp(1j * self.phase[:])

                    self.fft_output[:] = self.EW_j[:]
                    self.E_j[:] = self.ifft()

                # do not apply filter if power spectrum is known (so elif)
                elif forbidden_um is not None:
                    # power is not allowed at these wavelengths (filter them out!)
                    self.EW_j *= window_forbidden

                    self.fft_output[:] = self.EW_j[:]
                    self.E_j[:] = self.ifft()

                # temporary debugging
                if debug_plotting and n % 10 == 0:
                    [i.clear() for i in axs_debug]
                    axs_debug[0].plot(self.pulse.F_THz, abs(np.fft.fftshift(self.EW_j)) ** 2)
                    axs_debug[1].plot(self.pulse.F_THz, abs(np.fft.fftshift(self.phi_j)))
                    axs_debug[2].plot(self.pulse.F_THz, abs(fft(np.fft.fftshift(self.corr1))))
                    axs_debug[3].plot(self.pulse.F_THz, abs(fft(np.fft.fftshift(self.corr2))))

                    axs_debug[0].set_title("EW_j")
                    axs_debug[1].set_title("phi_j after amp replacement")
                    axs_debug[2].set_title("corr1 W")
                    axs_debug[3].set_title("corr2 W")

                    axs_debug[4].plot(self.pulse.T_ps, abs(np.fft.fftshift(self.E_j)) ** 2)
                    axs_debug[5].plot(self.pulse.T_ps, abs(np.fft.fftshift(self.psi_j)))
                    axs_debug[6].plot(self.pulse.T_ps, abs(np.fft.fftshift(self.psiPrime_j)))
                    axs_debug[7].plot(self.pulse.T_ps, abs(np.fft.fftshift(self.corr1)))
                    axs_debug[8].plot(self.pulse.T_ps, abs(np.fft.fftshift(self.corr2)))

                    axs_debug[4].set_title("E_j")
                    axs_debug[5].set_title("psi_j after amp replacement")
                    axs_debug[6].set_title("psiPrime_j")
                    axs_debug[7].set_title("corr1")
                    axs_debug[8].set_title("corr2")

                    fig_debug.suptitle('{n}/{N}'.format(n=n, N=len(time_order)))

                    plt.pause(.001)

            if (meas_spectrum_um is not None) and i >= i_set_spectrum_to_meas:
                h += 1

            self.AT2D[:] = self.E_j[:]
            self.AW2D_to_shift[:] = self.EW_j[:]
            error = self.calculate_error(self.AT2D,
                                         self.AT2D_to_shift,
                                         self.AW2D_to_shift,
                                         self.spctgm_to_calc_Tdomain,
                                         self.spctgm_to_calc_Wdomain,
                                         self.phase2D,
                                         self.delay_time,
                                         V_THz_fftshift,
                                         interp_data_fftshift[ind_start:ind_end],
                                         ind_filter_fftshift)
            self.error[i] = error
            print(i, self.error[i])
            self.Output_Ej[i] = self.E_j
            self.Output_EWj[i] = self.EW_j

            if plot_update:
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax1.plot(self.pulse.T_ps, abs(np.fft.fftshift(self.E_j)) ** 2)
                ax2.plot(self.pulse.wl_um[ind_wl], abs(np.fft.fftshift(self.EW_j)[ind_wl]) ** 2)
                phase = np.unwrap(np.arctan2(np.fft.fftshift(self.EW_j.imag)[ind_wl],
                                             np.fft.fftshift(self.EW_j.real)[ind_wl]))
                ax3.plot(self.pulse.wl_um[ind_wl], phase, 'C1')
                fig.suptitle("iteration " + str(i) + "; error: " + "%.3f" % self.error[i])
                plt.pause(.001)

        if meas_spectrum_um is not None:
            if not grad_ramp_for_meas_spectrum:
                self.error = self.error[i_set_spectrum_to_meas:]
                self.Output_Ej = self.Output_Ej[i_set_spectrum_to_meas:]
                self.Output_EWj = self.Output_EWj[i_set_spectrum_to_meas:]

                bestind = np.argmin(self.error)
            else:
                bestind = -1
        else:
            bestind = np.argmin(self.error)

        self.AT_ret = np.fft.fftshift(self.Output_Ej[bestind])
        self.AW_ret = np.fft.fftshift(self.Output_EWj[bestind])
