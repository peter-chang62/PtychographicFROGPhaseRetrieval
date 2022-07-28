import copy
import numpy as np
import mkl_fft
import matplotlib.pyplot as plt
import BBO as BBO
import PullDataFromOSA as OSA
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.integrate as scint
import scipy.interpolate as spi
import scipy.constants as sc
import scipy.optimize as spo


def normalize(x):
    """
    :param x: normalizes the array x by abs(max(x))
    :return: normalized x
    """
    return x / np.max(abs(x))


# ______________________________________________________________________________________________________________________
# PyNLO has fft and ifft defined in reverse!
# ______________________________________________________________________________________________________________________


def ifft(x, axis=None):
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


def fft(x, axis=None):
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


def shift(x, freq, shift, axis=None, freq_is_angular=True):
    """
    :param x: 1D or 2D array
    :param freq: frequency axis
    :param shift: time shift
    :param axis: if x is 2D, the axis along which to perform the shift
    :param freq_is_angular: bool specifying whether freq is in angular frequency, default is True
    :return: shifted x

    The units of freq and shift can be anything, but they need to be consistent with each other (so if freq is in
    THz, then shift should be in ps)
    """

    assert (len(x.shape) == 1) or (isinstance(axis, int)), \
        "if x is not 1D, you need to provide an axis along which to perform the shift"
    assert isinstance(freq_is_angular, bool)

    phase = np.zeros(x.shape, dtype=np.complex128)
    ft = fft(x, axis)

    if freq_is_angular:
        V = freq
    else:
        V = freq * 2 * np.pi

    if axis is None:
        # 1D scenario
        phase[:] = np.exp(1j * V * shift)
        ft *= phase
        return ifft(ft)

    else:
        assert shift.shape == (x.shape[0],), "shift must be a 1D array, one shift for each row of x"
        phase[:] = 1j * V
        phase = np.exp(phase * np.c_[shift])
        ft *= phase
        return ifft(ft, axis)


def calculate_spectrogram(pulse, T_fs):
    """
    :param pulse: pulse instance
    :param T_fs: Time axis of the spectrogram
    :return: 2D array for the spectrogram
    """

    assert isinstance(pulse, fpn.Pulse), "pulse must be a Pulse instance"
    pulse: fpn.Pulse

    AT = np.zeros((len(T_fs), len(pulse.AT)), dtype=np.complex128)
    AT[:] = pulse.AT
    AT_shift = shift(AT, pulse.V_THz, T_fs * 1e-3, axis=1)  # THz and ps
    AT2 = AT * AT_shift
    AW2 = fft(AT2, axis=1)
    return abs(AW2) ** 2


def denoise(x, gamma):
    """
    :param x: array
    :param gamma: float that is the threshold
    :return: denoised x
    """

    # this is how Sidorenko has it implemented in his code, the one difference is that the threshold condition is on
    # abs(x), and then x - gamma * sign(x) is applied to the real and imaginary parts separately
    # Note: ____________________________________________________________________________________________________________
    #   np.sign(x) operates on the real only if you pass it a complex x:
    #   np.sign(1 + 1j) = 1 + 0j
    # __________________________________________________________________________________________________________________
    return np.where(abs(x) >= gamma, x.real - gamma * np.sign(x.real), 0) + \
           1j * np.where(abs(x) >= gamma, x.imag - gamma * np.sign(x.imag), 0)


def load_data(path):
    """
    :param path: str - path to data
    :return: wavelength (nm) 1D array, frequency (THz) 1D array, spectrogram 2D array
    """

    # extract relevant variables from the spectrogram data:
    #   1. time axis
    #   2. wavelength axis
    #   3. frequency axis
    # no alteration to the data besides truncation along the time-axis to center T0 is done here

    spectrogram = np.genfromtxt(path)
    T_fs = spectrogram[:, 0][1:]  # time is on the column
    wl_nm = spectrogram[0][1:]  # wavelength is on the row
    F_THz = sc.c * 1e-12 / (wl_nm * 1e-9)  # experimental frequency axis from wl_nm
    spectrogram = spectrogram[1:, 1:]

    # center T0
    x = scint.simps(spectrogram, axis=1)
    ind = np.argmax(x)
    ind_keep = min([ind, len(spectrogram) - ind])
    spectrogram = spectrogram[ind - ind_keep: ind + ind_keep]
    T_fs -= T_fs[ind]
    T_fs = T_fs[ind - ind_keep: ind + ind_keep]

    return wl_nm, F_THz, T_fs, normalize(spectrogram)


def func(gamma, args):
    spctgm, spctgm_exp = args
    return np.sqrt(np.mean(abs(normalize(spctgm) - gamma * normalize(spctgm_exp)) ** 2))


class Retrieval:
    def __init__(self):
        self._wl_nm = None
        self._F_THz = None
        self._T_fs = None
        self._spectrogram = None
        self._min_sig_fthz = None
        self._min_pm_fthz = None
        self._max_sig_fthz = None
        self._max_pm_fthz = None
        self._pulse = None
        self._pulse_data = None
        self._spectrogram_interp = None
        self._ind_pm_fthz = None
        self._error = None
        self._AT2D = None

    # _________________________________ variables to keep track of _____________________________________________________

    @property
    def wl_nm(self):
        assert isinstance(self._wl_nm, np.ndarray), "no spectrogram data has been loaded yet"
        return self._wl_nm

    @property
    def F_THz(self):
        assert isinstance(self._F_THz, np.ndarray), "no spectrogram data has been loaded yet"
        return self._F_THz

    @property
    def T_fs(self):
        assert isinstance(self._T_fs, np.ndarray), "no spectrogram data has been loaded yet"
        return self._T_fs

    @property
    def spectrogram(self):
        assert isinstance(self._spectrogram, np.ndarray), "no spectrogram data has been loaded yet"
        return self._spectrogram

    @property
    def min_sig_fthz(self):
        assert isinstance(self._min_sig_fthz, float), "no signal frequency range has been set yet"
        return self._min_sig_fthz

    @property
    def max_sig_fthz(self):
        assert isinstance(self._max_sig_fthz, float), "no signal frequency range has been set yet"
        return self._max_sig_fthz

    @property
    def min_pm_fthz(self):
        assert isinstance(self._min_pm_fthz, float), "no phase matching bandwidth has been defined yet"
        return self._min_pm_fthz

    @property
    def max_pm_fthz(self):
        assert isinstance(self._max_pm_fthz, float), "no phase matching bandwidth has been defined yet"
        return self._max_pm_fthz

    @property
    def ind_pm_fthz(self):
        assert isinstance(self._ind_pm_fthz, np.ndarray), "no phase matching bandwidth has been defined yet"
        return self._ind_pm_fthz

    @property
    def pulse(self):
        assert isinstance(self._pulse, fpn.Pulse), "no initial guess has been set yet"
        return self._pulse

    @property
    def pulse_data(self):
        assert isinstance(self._pulse_data, fpn.Pulse), "no spectrum data has been loaded yet"
        return self._pulse_data

    @property
    def error(self):
        assert isinstance(self._error, np.ndarray), "no retrieval has been run yet"
        return self._error

    @property
    def AT2D(self):
        assert isinstance(self._AT2D, np.ndarray), "no retrieval has been run yet"
        return self._AT2D

    @property
    def spectrogram_interp(self):
        assert isinstance(self._spectrogram_interp, np.ndarray), \
            "spectrogram has not been interpolated to the simulation grid"
        return self._spectrogram_interp

    # _______________________ functions ________________________________________________________________________________

    def load_data(self, path):
        """
        :param path: path to data

        loads the data
        """
        self._wl_nm, self._F_THz, self._T_fs, self._spectrogram = load_data(path)

    def set_signal_freq(self, min_sig_fthz, max_sig_fthz):
        """
        :param min_sig_fthz: minimum signal frequency
        :param max_sig_fthz: maximum signal frequency

        sets the minimum and maximum signal frequency and then denoises the parts of the spectrogram that is outside
        this frequency range, this is used purely for calling denoise on the spectrogram, and does not set the
        frequency range to be used for retrieval (that is instead set by the phase matching bandwidth)
        """

        self._min_sig_fthz, self._max_sig_fthz = float(min_sig_fthz), float(max_sig_fthz)
        self.denoise_spectrogram()

    def _get_ind_fthz_nosig(self):
        """
        :return: an array of integers

        This gets an array of indices for the experimental wavelength axis that falls inside the signal frequency
        range (the one that is used to denoise the spectrogram). This can only be called after min_sig_fthz and
        max_sig_fthz have been set by set_signal_freq
        """

        mask_fthz_sig = np.logical_and(self.F_THz >= self.min_sig_fthz, self.F_THz <= self.max_sig_fthz)

        ind_fthz_nosig = np.ones(len(self.F_THz))
        ind_fthz_nosig[mask_fthz_sig] = 0
        ind_fthz_nosig = ind_fthz_nosig.nonzero()[0]

        return ind_fthz_nosig

    def denoise_spectrogram(self):
        """
        denoise the spectrogram using min_sig_fthz and max_sig_fthz
        """
        self.spectrogram[:] = normalize(self.spectrogram)

        ind_fthz_nosig = self._get_ind_fthz_nosig()
        self.spectrogram[:, ind_fthz_nosig] = denoise(self.spectrogram[:, ind_fthz_nosig], 1e-3).real

    def correct_for_phase_matching(self, deg=5.5):
        """
        :param deg: non-collinear angle incident into the BBO is fixed at 5.5 degrees

        the spectrogram is divided by the phase-matching curve, and then denoised, so this can only be called
        after calling set_signal_freq
        """
        assert deg == 5.5

        bbo = BBO.BBOSHG()
        R = bbo.R(self.wl_nm * 1e-3 * 2, 50, bbo.phase_match_angle_rad(1.55), BBO.deg_to_rad(5.5))
        ind_10perc = np.argmin(abs(R[300:] - .1)) + 300  # the frog spectrogram doesn't usually extend past here

        self.spectrogram[:, ind_10perc:] /= R[ind_10perc:]
        self.denoise_spectrogram()

        self._min_pm_fthz = min(self.F_THz)
        self._max_pm_fthz = self.F_THz[ind_10perc]

    def set_initial_guess(self, center_wavelength_nm=1560, time_window_ps=10, NPTS=2 ** 12):
        """
        :param center_wavelength_nm: center wavelength in nanometers
        :param time_window_ps: time window in picoseconds
        :param NPTS: number of points

        This initializes a pulse using PyNLO with a sech envelope, whose time bandwidth is set according to the
        intensity autocorrelation of the spectrogram. Realize that the spectrogram could have been slightly altered
        depending on whether it's been denoised (called by either set_signal_freq or correct_for_phase_matching,
        but this should not influence the time bandwidth significantly)
        """
        # integrate experimental spectrogram across wavelength axis
        x = - scint.simpson(self.spectrogram, x=self.F_THz, axis=1)

        spl = spi.UnivariateSpline(self.T_fs, normalize(x) - .5, s=0)
        roots = spl.roots()

        T0 = np.diff(roots[[0, -1]]) * 0.65 / 1.76
        self._pulse = fpn.Pulse(T0_ps=T0 * 1e-3, center_wavelength_nm=center_wavelength_nm,
                                time_window_ps=time_window_ps, NPTS=NPTS)
        phase = np.random.uniform(low=0, high=1, size=self.pulse.NPTS) * np.pi / 8
        self._pulse.set_AT(self._pulse.AT * np.exp(1j * phase))  # random phase

    def load_spectrum_data(self, wl_um, spectrum):
        """
        :param wl_um: wavelength in um
        :param spectrum: power spectrum (from a spectrometer or monochromator)

        This can only be called after having already called set_initial_guess. It clones the original pulse
        and sets the envelope in the frequency domain to the transform limited pulse calculated from the
        power spectrum
        """

        # when converting dB to linear scale for data taken by the monochromator, sometimes you get negative
        # values at wavelengths where you have no (or very little) power (experimental error)
        assert np.all(spectrum > 0), "a negative spectrum is not physical"

        pulse_data: fpn.Pulse
        pulse_data = copy.deepcopy(self.pulse)
        pulse_data.set_AW_experiment(wl_um, np.sqrt(spectrum))
        self._pulse_data = pulse_data

    def _intrplt_spctrgrm_to_sim_grid(self):
        """
        This interpolates the spectrogram to the simulation grid. This can only be called after calling
        set_initial_guess and correct_for_phase_matching because the simulation grid is defined by the pulse's
        frequency grid, and the interpolation range is narrowed down to the phase-matching bandwidth
        """

        gridded = spi.interp2d(self.F_THz, self.T_fs, self.spectrogram, bounds_error=True)
        spectrogram_interp = gridded(self.pulse.F_THz[self.ind_pm_fthz] * 2, self.T_fs)

        # scale the interpolated spectrogram to match the pulse energy. I do it here instead of to the experimental 
        # spectrogram, because the interpolated spectrogram has the same integration frequency axis as the pulse 
        # instance 
        x = calculate_spectrogram(self.pulse, self.T_fs)
        factor = scint.simpson(scint.simpson(x[:, self.ind_pm_fthz])) / scint.simpson(scint.simpson(spectrogram_interp))
        spectrogram_interp *= factor
        self._spectrogram_interp = spectrogram_interp

    def retrieve(self, start_time, end_time, itermax, iter_set=None, plot_update=True):
        """
        :param start_time:
        :param end_time:
        :param itermax:
        :param iter_set:
        """

        assert (iter_set is None) or (isinstance(self.pulse_data, fpn.Pulse) and isinstance(iter_set, int))

        # self._ind_pm_fthz = np.logical_and(self.pulse.F_THz * 2 >= self.min_pm_fthz,
        #                                    self.pulse.F_THz * 2 <= self.max_pm_fthz).nonzero()[0]

        # I use self.ind_pm_fthz to set the retrieval's frequency bandwidth. Previously I set the retrieval's
        # frequency bandwidth to the phase-matching bandwidth (hence the name), but now I want to set it to the
        # signal frequency bandwidth. I haven't removed the previous line though, since it's not called repeatedly
        # during retrieval (so it's not a waste of time), and it's a useful way to check that the user has called
        # self.correct_for_phase_matching
        self._ind_pm_fthz = np.logical_and(self.pulse.F_THz * 2 >= self.min_sig_fthz,
                                           self.pulse.F_THz * 2 <= self.max_sig_fthz).nonzero()[0]

        self._intrplt_spctrgrm_to_sim_grid()

        ind_start = np.argmin(abs(self.T_fs - start_time))
        ind_end = np.argmin(abs(self.T_fs - end_time))
        delay_time = self.T_fs[ind_start:ind_end]
        time_order_ps = np.c_[delay_time * 1e-3, np.arange(ind_start, ind_end)]

        j_excl = np.ones(len(self.pulse.F_THz))
        j_excl[self.ind_pm_fthz] = 0
        j_excl = j_excl.nonzero()[0]  # everything but ind_pm_fthz

        error = np.zeros(itermax)
        rng = np.random.default_rng()

        AT = np.zeros((itermax, len(self.pulse.AT)), dtype=np.complex128)

        if plot_update:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax3 = ax2.twinx()

        for iter in range(itermax):
            rng.shuffle(time_order_ps, axis=0)
            alpha = abs(0.2 + rng.standard_normal(1) / 20)
            for dt, j in time_order_ps:
                j = int(j)

                AT_shift = shift(self.pulse.AT, self.pulse.V_THz, dt)
                psi_j = AT_shift * self.pulse.AT
                phi_j = fft(psi_j)

                amp = abs(phi_j)
                amp[self.ind_pm_fthz] = np.sqrt(self.spectrogram_interp[j])
                phase = np.arctan2(phi_j.imag, phi_j.real)
                phi_j[:] = amp * np.exp(1j * phase)

                # denoise everything that is not inside the wavelength range of the spectrogram that is being used
                # for retrieval. Intuitively, this is all the frequencies that you don't think the spectrogram gives
                # reliable results for. The threshold is the max of phi_j / 1000. Otherwise, depending on what pulse
                # energy you decided to run with during retrieval, the 1e-3 threshold can do different things.
                # Intuitively, the threshold should be set close to the noise floor, which is determined by the
                # maximum.
                phi_j[j_excl] = denoise(phi_j[j_excl], 1e-3 * abs(phi_j).max())
                # phi_j[:] = denoise(phi_j[:], 1e-3 * abs(phi_j).max())  # or not

                psi_jp = ifft(phi_j)
                corr1 = AT_shift.conj() * (psi_jp - psi_j) / np.max(abs(AT_shift) ** 2)
                corr2 = self.pulse.AT.conj() * (psi_jp - psi_j) / np.max(abs(self.pulse.AT) ** 2)
                corr2 = shift(corr2, self.pulse.V_THz, -dt)

                self.pulse.set_AT(
                    self.pulse.AT
                    + alpha * corr1
                    + alpha * corr2
                )

                # ______________________________________________________________________________________________________________
                # substitution of power spectrum
                if iter_set is not None:
                    if iter >= iter_set:
                        phase = np.arctan2(self.pulse.AW.imag, self.pulse.AW.real)
                        self.pulse.set_AW(abs(self.pulse_data.AW) * np.exp(1j * phase))
                # ______________________________________________________________________________________________________________

            # __________________________________________________________________________________________________________________
            # preparing for substitution of power spectrum
            if iter_set is not None:
                if iter == iter_set - 1:  # the one before iter_set
                    self.pulse_data.set_epp(self.pulse.calc_epp())
            # __________________________________________________________________________________________________________________

            if plot_update:
                [ax.clear() for ax in [ax1, ax2, ax3]]
                ax1.plot(self.pulse.T_ps, self.pulse.AT.__abs__() ** 2)
                ax2.plot(self.pulse.F_THz, self.pulse.AW.__abs__() ** 2)
                ax3.plot(self.pulse.F_THz, np.unwrap(np.arctan2(self.pulse.AW.imag, self.pulse.AW.real)), color='C1')
                ax2.set_xlim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)
                fig.suptitle(iter)
                plt.pause(.1)

            s = calculate_spectrogram(self.pulse, self.T_fs)[ind_start:ind_end, self.ind_pm_fthz]
            # error[iter] = np.sqrt(np.sum(abs(s - self.spectrogram_interp) ** 2)) / np.sqrt(
            #     np.sum(abs(self.spectrogram_interp) ** 2))
            res = spo.minimize(func, np.array([1]), args=[s, self.spectrogram_interp[ind_start:ind_end]])
            error[iter] = res.fun
            AT[iter] = self.pulse.AT

            print(iter, error[iter])

        self._error = error
        self._AT2D = AT

    def plot_results(self, set_to_best=True):
        if set_to_best:
            self.pulse.set_AT(self.AT2D[np.argmin(self.error)])

        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()

        # plot time domain
        ax[0].plot(self.pulse.T_ps, self.pulse.AT.__abs__() ** 2)

        # plot frequency domain
        ax[1].plot(self.pulse.F_THz, self.pulse.AW.__abs__() ** 2)
        ax[1].set_xlim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)

        # plot the phase on same plot as frequency domain
        axp = ax[1].twinx()
        ind_sig = np.logical_and(self.pulse.F_THz * 2 >= self.min_sig_fthz,
                                 self.pulse.F_THz * 2 <= self.max_sig_fthz).nonzero()[0]
        phase = BBO.rad_to_deg(np.unwrap(np.arctan2(self.pulse.AW[ind_sig].imag,
                                                    self.pulse.AW[ind_sig].real)))
        axp.plot(self.pulse.F_THz[ind_sig], phase, color='C1')

        # plot the experimental spectrogram
        ax[2].pcolormesh(self.T_fs, self.F_THz / 2, self.spectrogram.T, cmap='jet')
        ax[2].set_ylim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)

        # plot the retrieved spectrogram
        s = calculate_spectrogram(self.pulse, self.T_fs)
        ind_spctrmtr = np.logical_and(self.pulse.F_THz * 2 >= min(self.F_THz),
                                      self.pulse.F_THz * 2 <= max(self.F_THz)).nonzero()[0]
        ax[3].pcolormesh(self.T_fs, self.pulse.F_THz[ind_spctrmtr], s[:, ind_spctrmtr].T, cmap='jet')
        ax[3].set_ylim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)

        # plot the experimental power spectrum
        if isinstance(self._pulse_data, fpn.Pulse):
            factor = max(self.pulse.AW.__abs__() ** 2) / max(self.pulse_data.AW.__abs__() ** 2)
            ax[1].plot(self.pulse_data.F_THz, self.pulse_data.AW.__abs__() ** 2 * factor, color='C2')
