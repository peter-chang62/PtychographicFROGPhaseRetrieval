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
import PhaseRetrieval as pr


def correct_for_phase_matching(exp_wl_um, spctgm):
    pulse_ref: fpn.Pulse

    data_phasematching = np.genfromtxt("ptych_FROG_Timmers/BBO_50um_PhaseMatchingCurve.txt")
    wl, r = data_phasematching[:, 0], data_phasematching[:, 1]
    r_gridded = spi.interp1d(wl, r)
    ind = np.logical_and(exp_wl_um < min(wl), exp_wl_um > max(wl)).nonzero()[0]
    r_ = r_gridded(exp_wl_um[ind])
    spctgm[:, ind] /= r_


# %%
ret = pr.Retrieval(maxiter=25, time_window_ps=10, NPTS=2 ** 12, center_wavelength_nm=1000.0)
# ret.load_data("TestData/sanity_check_data.txt")
# ret.load_data("Data/01-14-2022/successfully_symmetric_frog.txt")
ret.load_data("Data/01-17-2022/realigned_spectrometer_input.txt")

# %%
# ret.data[:] = ret.data[::-1]  # worked too! (with enough reruns but that's usual)
# ret.data[:] = (ret.data[:] + ret.data[::-1]) / 2

# %% set initial guess from power spectrum, doesn't work well
spectrum = osa.Data("Data/01-17-2022/SPECTRUM_FOR_FROG.CSV", False)
# pulse = copy.deepcopy(ret.pulse)
# pulse.set_AW_experiment(spectrum.x * 1e-3, np.sqrt(spectrum.y))
# initial_guess = pr.ifft(pulse.AW)

# %% or initial guess to be a phase retrieval result, but transform limited
imag = np.genfromtxt("Data/01-14-2022/retrieval_no_spec_imag_f_thz_2.txt")
real = np.genfromtxt("Data/01-14-2022/retrieval_no_spec_real_f_thz_2.txt")
fthz = real[:, 0]
real = real[:, 1]
imag = imag[:, 1]
AW = real + 1j * imag

pulse = copy.deepcopy(ret.pulse)
pulse.set_AW_experiment(sc.c * 1e6 / (fthz * 1e12), abs(AW))  # transform limited
initial_guess = pr.ifft(pulse.AW)

# %%

# using an initial guess from phase retrieval result
# ret.retrieve(corr_for_pm=True,
#              plot_update=True,
#              initial_guess_T_fs_AT=[pulse.T_ps * 1e3, initial_guess],
#              filter_um=[.500 * 2, ret.exp_wl_nm[-1] * 2])

# constraining power spectrum to phase retrieval result
# ret.retrieve(corr_for_pm=True,
#              plot_update=True,
#              initial_guess_T_fs_AT=None,
#              filter_um=[.500 * 2, ret.exp_wl_nm[-1] * 2],
#              meas_spectrum_um=[sc.c * 1e6 / (fthz * 1e12), abs(AW) ** 2])

p = fpn.Pulse(center_wavelength_nm=1560.)
aw = p.AW

# honest phase retrieval
correct_for_phase_matching(ret.exp_wl_nm * 1e-3, ret.data)
# ret.correct_for_phase_match(alpha_rad=np.arctan(.25 / 2))
ind = (ret.exp_wl_nm < 500).nonzero()[0]
ret.data[:, ind] = 0
ret.retrieve(corr_for_pm=False,
             plot_update=True,
             initial_guess_T_ps_AT=None,
             initial_guess_wl_um_AW=[p.wl_um, aw],
             filter_um=None,
             plot_wl_um=[0., 2],
             meas_spectrum_um=None)

# %%
pr.apply_filter(ret.AW_ret, 0.9, 2, ret.pulse)
ret.AT_ret = pr.ifft(ret.AW_ret)
pr.plot_ret_results(ret.AT_ret, ret.exp_T_fs, ret.pulse, ret.interp_data,
                    filter_um=[.500 * 2, ret.exp_wl_nm[-1] * 2])
