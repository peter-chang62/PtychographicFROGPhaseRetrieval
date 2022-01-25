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


def correct_for_phase_matching(exp_wl_um, spctgm):
    pulse_ref: fpn.Pulse

    data_phasematching = np.genfromtxt("ptych_FROG_Timmers/BBO_50um_PhaseMatchingCurve.txt")
    wl, r = data_phasematching[:, 0], data_phasematching[:, 1]
    r_gridded = spi.interp1d(wl, r)
    ind = np.logical_and(exp_wl_um < min(wl), exp_wl_um > max(wl)).nonzero()[0]
    r_ = r_gridded(exp_wl_um[ind])
    spctgm[:, ind] /= r_


def normalize(vec):
    return vec / np.max(abs(vec))


# %%
center_wavelength_nm = 1560.

# %%
ret = pr.Retrieval(maxiter=25, time_window_ps=80, NPTS=2 ** 15, center_wavelength_nm=center_wavelength_nm)
# ret.load_data("TestData/sanity_check_data.txt")
# ret.load_data("Data/01-17-2022/realigned_spectrometer_input.txt")
# ret.load_data("Data/01-18-2022/spectrogram_grating_pair_output.txt")
ret.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")

# %%
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)

# %%
ret.retrieve(corr_for_pm=True,
             plot_update=False,
             initial_guess_wl_um_AW=None,
             plot_wl_um=[1.54, 1.6],
             filter_um=[1.0, 1.8],
             debug_plotting=False,
             meas_spectrum_um=[osa.x * 1e-3, osa.y],
             # meas_spectrum_um=None,
             i_set_spectrum_to_meas=0
             )

# %%
p = fpn.Pulse(center_wavelength_nm=center_wavelength_nm, NPTS=2 ** 12, time_window_ps=40)
p.set_AW_experiment(ret.pulse.wl_um, ret.AW_ret)
AT = pr.ifft(p.AW)
spctgm = pr.calculate_spctgm(AT, ret.exp_T_fs, p)

fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(ret.exp_T_fs, ret.exp_wl_nm * 1e-3 * 2, ret.data.T, cmap='jet')
ax[0].set_ylim(1.54, 1.58)
ax[1].pcolormesh(ret.exp_T_fs, p.wl_um, spctgm.T, cmap='jet')
ax[1].set_ylim(1.54, 1.58)

