import gc
import numpy as np
import scipy.constants as sc
import BBO as BBO
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.interpolate as spi
from scipy.integrate import simps
import matplotlib.pyplot as plt
import PullDataFromOSA as OSA
import copy
import clipboard_and_style_sheet
import PhaseRetrieval as pr
import mkl_fft


def normalize(vec):
    return vec / np.max(abs(vec))


def fft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def denoise(x, gamma):
    return np.where(abs(x) < gamma, 0.0, np.sign(x) * (abs(x) - gamma))


class Retrieval(pr.Retrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def retrieve(self, corr_for_pm=True,
                 start_time_fs=None,
                 end_time_fs=None,
                 plot_update=True,
                 initial_guess_T_ps_AT=None,
                 initial_guess_wl_um_AW=None,
                 filter_um=None,
                 forbidden_um=None,
                 meas_spectrum_um=None,
                 i_set_spectrum_to_meas=0,
                 plot_wl_um=(1.0, 2.0),
                 debug_plotting=False):

        """
        :param corr_for_pm:
        :param start_time_fs:
        :param end_time_fs:
        :param plot_update:
        :param initial_guess_T_ps_AT:
        :param initial_guess_wl_um_AW:
        :param filter_um:

        :param forbidden_um: brick wall filter after each iteration, I'll leave it here since I already wrote it but
        generally it's not a good idea, unless you implement some sort of window to avoid edges from developing

        :param meas_spectrum_um:
        :param i_set_spectrum_to_meas:
        :param plot_wl_um:
        :param debug_plotting:
        :return:
        """

        if corr_for_pm:
            # make sure to correct for phase matching
            if not self.corrected_for_phase_matching:
                self.correct_for_phase_match()

        # make sure to have interpolated the data to the simulation grid
        if self._interp_data is None:
            self.interpolate_data_to_sim_grid()

        if start_time_fs is None:
            start_time_fs = 0.0
        if end_time_fs is None:
            end_time_fs = self.exp_T_fs[-1]

        if meas_spectrum_um is not None:
            wl_um, spectrum = meas_spectrum_um
            aw = np.sqrt(abs(spectrum))
            self.pulse.set_AW_experiment(wl_um, aw)
            self.pulse.set_AT(ifft(self.pulse.AW))  # pynlo has ifft <-> fft defined in reverse

        else:

            if (initial_guess_T_ps_AT is None) and (initial_guess_wl_um_AW is None):
                # default to autocorrelation
                initial_guess_T_ps_AT = np.sum(self._interp_data, axis=1)

                # interestingly enough, symmetrizing doesn't help
                # initial_guess_T_fs_AT[:] = (initial_guess_T_fs_AT[:] + initial_guess_T_fs_AT[::-1]) / 2

                initial_guess_T_ps_AT -= min(initial_guess_T_ps_AT)

                self.pulse.set_AT_experiment(self.exp_T_fs * 1e-3, initial_guess_T_ps_AT)

            elif (initial_guess_T_ps_AT is not None) and (initial_guess_wl_um_AW is not None):
                raise RuntimeError("only one of initial_guess_T_fs_AT or initial_guess_wl_um_AW can be defined")

            elif initial_guess_T_ps_AT is not None:
                # initial guess generally can be complex
                T_ps, field = initial_guess_T_ps_AT
                self.pulse.set_AT_experiment(T_ps, field)

            else:
                wl_um, field = initial_guess_wl_um_AW
                self.pulse.set_AW_experiment(wl_um, field)
                self.pulse.set_AT(ifft(self.pulse.AW))

        # for incomplete spectrograms, the user can set a range of wavelengths to be used for phase retrieval. It's
        # important to note that the indexing will be done for fftshifted arrays
        wl_um_fftshift = np.fft.ifftshift(self.pulse.wl_um)
        if filter_um is not None:
            ll_um, ul_um = filter_um
            ind_filter_fftshift = np.logical_and(wl_um_fftshift > ll_um, wl_um_fftshift < ul_um).nonzero()[0]

        else:
            # in this case array[ind_filter_fftshift] = array[:]
            ind_filter_fftshift = np.arange(len(wl_um_fftshift))

        if forbidden_um is not None:
            ll_um, ul_um = forbidden_um
            ind_forbidden_fftshift = np.logical_or(wl_um_fftshift <= ll_um, wl_um_fftshift >= ul_um).nonzero()[0]

        """fftshift everything before fft's are calculated 

        The initial guess is pulse.AT, everything is calculated off the initial guess so fftshifting this fftshifts 
        everything that follows 

        The calculated spectrogram will be fftshifted, so the reference spectrogram used in the error calculation 
        also needs to be fftshifted 

        Since the fields are fftshifted, the frequency axis used to calculate time shifted fields also needs to be 
        fftshifted """

        AT0_fftshift = np.fft.ifftshift(self.pulse.AT)
        interp_data_fftshift = np.fft.ifftshift(self._interp_data, axes=1)
        V_THz_fftshift = np.fft.ifftshift(self.pulse.V_THz)

        # delay times to iterate over for the phase retrieval. We set this to the experimentally measured delayed
        # times. The user has the option to narrow the time window to a subset of what was measured
        # experimentally
        ind_start = np.argmin((self.exp_T_fs - start_time_fs) ** 2)
        ind_end = np.argmin((self.exp_T_fs - end_time_fs) ** 2)
        self.delay_time = self.exp_T_fs[ind_start:ind_end]
        time_order = np.array([*zip(self.delay_time, np.arange(ind_start, ind_end))])

        # only used if meas_spectrum_um is not None
        if meas_spectrum_um is not None:
            h_meas_spectrum = np.linspace(.001, 1, 10)
            h_meas_spectrum = np.repeat(h_meas_spectrum[:, np.newaxis], 5, 1).flatten()

            self.maxiter = i_set_spectrum_to_meas + len(h_meas_spectrum) - 1
            print(f'maxiter has been adjusted to {self.maxiter}')

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
            meas_amp_interp = abs(self.EW_j)

        if plot_update:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax3 = ax2.twinx()
            ind_wl = np.logical_and(self.pulse.wl_um >= plot_wl_um[0], self.pulse.wl_um <= plot_wl_um[-1]).nonzero()[0]
            # ind_wl = (self.pulse.wl_um > 0).nonzero()

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
            # alpha = abs(0.2 + self._rng.normal(0, 1) / 20)

            for n, (dt, j) in enumerate(time_order):
                self.shift1D(self.Eshift_j, self.EW_j, self.EWshift_j, dt, V_THz_fftshift)
                self.psi_j[:] = self.E_j[:] * self.Eshift_j[:]

                self.fft_input[:] = self.psi_j[:]
                self.phi_j[:] = self.fft()

                self.amp[:] = 0
                self.amp[ind_filter_fftshift] = np.sqrt(interp_data_fftshift[int(j), ind_filter_fftshift])
                ind_nonzero = (self.amp > 0).nonzero()[0]

                self.phase[:] = np.arctan2(self.phi_j.imag, self.phi_j.real)
                # only replace known parts of the spectrum
                self.phi_j[ind_nonzero] = self.amp[ind_nonzero] * np.exp(1j * self.phase[ind_nonzero])
                # self.phi_j[:] = self.amp[:] * np.exp(1j * self.phase[:])

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
                    if i == i_set_spectrum_to_meas:
                        starting_field = abs(self.EW_j)
                        diff = meas_amp_interp - starting_field
                        h = 0

                    self.amp[:] = starting_field + diff * h_meas_spectrum[h]
                    self.phase[:] = np.arctan2(self.EW_j.imag, self.EW_j.real)
                    self.EW_j[:] = self.amp[:] * np.exp(1j * self.phase[:])

                    self.fft_output[:] = self.EW_j[:]
                    self.E_j[:] = self.ifft()

                # do not apply filter if power spectrum is known
                elif forbidden_um is not None:
                    # power is not allowed at these wavelengths (brick wall filter)
                    self.EW_j[ind_forbidden_fftshift] = 0.0

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
                    axs_debug[7].set_title("corr2")
                    axs_debug[8].set_title("corr2")

                    fig_debug.suptitle('{n}/{N}'.format(n=n, N=len(time_order)))

                    plt.pause(.001)

            if (meas_spectrum_um is not None) and i >= i_set_spectrum_to_meas:
                h += 1
                # print(h)

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

        # if meas_spectrum_um is not None:
        #     self.error = self.error[i_set_spectrum_to_meas:]
        #     self.Output_Ej = self.Output_Ej[i_set_spectrum_to_meas:]
        #     self.Output_EWj = self.Output_EWj[i_set_spectrum_to_meas:]

        if meas_spectrum_um is not None:
            bestind = -1
        else:
            bestind = np.argmin(self.error)

        self.AT_ret = np.fft.fftshift(self.Output_Ej[bestind])
        self.AW_ret = np.fft.fftshift(self.Output_EWj[bestind])


# %%
center_wavelength_nm = 1560.
maxiter = 25
time_window_ps = 80
NPTS = 2 ** 15

ret = Retrieval(maxiter=25,
                time_window_ps=time_window_ps,
                NPTS=NPTS,
                center_wavelength_nm=center_wavelength_nm)

# %%
ret.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", data_is_log=False)

# %% retrieval without measured power spectrum
# ret.retrieve(corr_for_pm=True,
#              start_time_fs=0,
#              end_time_fs=275,
#              plot_update=True,
#              initial_guess_T_ps_AT=None,
#              initial_guess_wl_um_AW=None,
#              filter_um=None,
#              forbidden_um=None,
#              meas_spectrum_um=None,
#              i_set_spectrum_to_meas=0,
#              plot_wl_um=[1.54, 1.58],
#              debug_plotting=False)

# %% retrieval with measured power spectrum
ret.retrieve(corr_for_pm=True,
             start_time_fs=-275,
             end_time_fs=275,
             plot_update=True,
             initial_guess_T_ps_AT=None,
             initial_guess_wl_um_AW=None,
             filter_um=None,
             forbidden_um=None,
             meas_spectrum_um=[osa.x * 1e-3, osa.y],
             i_set_spectrum_to_meas=10,
             plot_wl_um=[1.54, 1.58],
             debug_plotting=False)

# %%
spctgm, fig, axs = pr.plot_ret_results(ret.AT_ret, ret.exp_T_fs, ret.pulse, ret.interp_data, plot_um=[1.54, 1.58])
