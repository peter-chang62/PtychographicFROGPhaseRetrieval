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
ret = pr.Retrieval(25, time_window_ps=20)
ret.load_data("Data/01-18-2022/spectrogram_grating_pair_output.txt")
# ret.data[:] = ret.data[::-1]

# %%
spec = osa.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", data_is_log=False)

# %%
pulse = fpn.Pulse(.2, 1560, 10)

# %%
# correct_for_phase_matching(ret.exp_wl_nm * 1e-3, ret.data)
ret.retrieve(corr_for_pm=True,
             plot_update=True,
             meas_spectrum_um=[spec.x * 1e-3, spec.y],
             filter_um=[1.55, 1.575],
             i_set_spectrum_to_meas=5,
             plot_wl_um=[1.54, 1.58])

# %%
spec = pr.calculate_spctgm(ret.AT_ret, ret.exp_T_fs, ret.pulse)
fig, axs = plt.subplots(1, 2)
axs[0].pcolormesh(ret.exp_T_fs, ret.pulse.wl_um, ret.interp_data.T, cmap='jet')
axs[1].pcolormesh(ret.exp_T_fs, ret.pulse.wl_um, spec.T, cmap='jet')
[i.set_ylim(1.53, 1.59) for i in axs]
fig.suptitle("0 " + str(min(ret.error)))
