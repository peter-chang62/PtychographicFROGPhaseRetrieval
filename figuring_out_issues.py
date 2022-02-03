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


# Henry's phase matching curve, otherwise just set corr_for_pm to True
# def correct_for_phase_matching(exp_wl_um, spctgm):
#     pulse_ref: fpn.Pulse
#
#     data_phasematching = np.genfromtxt("ptych_FROG_Timmers/BBO_50um_PhaseMatchingCurve.txt")
#     wl, r = data_phasematching[:, 0], data_phasematching[:, 1]
#     r_gridded = spi.interp1d(wl, r)
#     ind = np.logical_and(exp_wl_um < min(wl), exp_wl_um > max(wl)).nonzero()[0]
#     r_ = r_gridded(exp_wl_um[ind])
#     spctgm[:, ind] /= r_


def normalize(vec):
    return vec / np.max(abs(vec))


# %%
center_wavelength_nm = 1560.
maxiter = 25
time_window_ps = 80
NPTS = 2 ** 15

# %%
ret = pr.Retrieval(maxiter=maxiter, time_window_ps=time_window_ps, NPTS=NPTS, center_wavelength_nm=center_wavelength_nm)
# ret.load_data("TestData/sanity_check_data.txt")
# ret.load_data("Data/01-17-2022/realigned_spectrometer_input.txt")
# ret.load_data("Data/01-18-2022/spectrogram_grating_pair_output.txt")
ret.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")

# %%
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)

# %%
ret.retrieve(corr_for_pm=True,
             start_time_fs=0,
             end_time_fs=275,
             plot_update=True,
             plot_wl_um=[1.54, 1.58],
             initial_guess_T_ps_AT=None,
             initial_guess_wl_um_AW=None,
             filter_um=None,
             forbidden_um=None,
             meas_spectrum_um=None,
             grad_ramp_for_meas_spectrum=False,
             i_set_spectrum_to_meas=5,
             debug_plotting=False
             )

# %%
spctgm, fig, axs = pr.plot_ret_results(ret.AT_ret, ret.exp_T_fs, ret.pulse, ret.interp_data, plot_um=[1.54, 1.58])

# %% save retrieval results
# fthz = ret.pulse.F_THz
# imag = ret.AW_ret.imag
# real = ret.AW_ret.real
# imag = np.hstack((fthz[:, np.newaxis], imag[:, np.newaxis]))
# real = np.hstack((fthz[:, np.newaxis], real[:, np.newaxis]))
# np.savetxt("Data/01-24-2022/consecutive_retrieval_attempts/fthz_imag_9.txt", imag)
# np.savetxt("Data/01-24-2022/consecutive_retrieval_attempts/fthz_real_9.txt", real)
