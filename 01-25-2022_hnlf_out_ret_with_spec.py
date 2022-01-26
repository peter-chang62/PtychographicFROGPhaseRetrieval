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
from scipy.signal.windows import tukey


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
# osa_short = OSA.Data("Data/01-17-2022/SPECTRUM_FOR_FROG.CSV", False)
# osa_long = OSA.Data("Data/01-17-2022/SPECTRUM_HNLF_2um_OSA.CSV", False)
#
# gridded_short = spi.interp1d(osa_short.x, normalize(osa_short.y), bounds_error=False, fill_value=0.0)
# gridded_long = spi.interp1d(osa_long.x, normalize(osa_long.y), bounds_error=False, fill_value=0.0)
# wl = np.linspace(osa_short.x[0], osa_long.x[-1], 5000)
#
# specshort = gridded_short(wl)
# speclong = gridded_long(wl)
#
# spec = np.where(wl < 1450, specshort, speclong)
# window = tukey(4500, .15)
# window = np.hstack((window, np.zeros(len(spec) - len(window))))
# spec_windowed = spec * window

# %%
# clipboard_and_style_sheet.style_sheet()
# plt.figure()
# plt.plot(wl, spec_windowed)
# plt.xlabel("wavelength (nm)")

# %%
# final = np.hstack((wl[:, np.newaxis], spec_windowed[:, np.newaxis])
# np.savetxt("Data/01-17-2022/Spectrum_Stitched_Together_wl_nm.txt", final)

# %%
center_wavelength_nm = 1560.
time_window_ps = 10
NPTS = 2 ** 14

maxiter = 25

# %%
ret = pr.Retrieval(maxiter=maxiter,
                   time_window_ps=time_window_ps,
                   NPTS=NPTS,
                   center_wavelength_nm=center_wavelength_nm)

ret.load_data("Data/01-17-2022/realigned_spectrometer_input.txt")


# %%

class OSAClass:
    def __init__(self):
        data = np.genfromtxt(
            "Data/01-17-2022/Spectrum_Stitched_Together_wl_nm.txt"
        )
        self.wl_nm = data[:, 0]
        self.wl_um = self.wl_nm * 1e-3
        self.spectrum = data[:, 1]


osa = OSAClass()

# %%
ret.retrieve(corr_for_pm=True,
             start_time_fs=0,
             end_time_fs=None,
             plot_update=True,
             initial_guess_T_ps_AT=None,
             initial_guess_wl_um_AW=None,
             filter_um=None,
             forbidden_um=None,
             meas_spectrum_um=[osa.wl_um, osa.spectrum],
             # meas_spectrum_um=None,
             i_set_spectrum_to_meas=10,
             plot_wl_um=[0.8, 2.2],
             debug_plotting=False
             )

# %%
pr.plot_ret_results(ret.AT_ret, ret.exp_T_fs, ret.pulse, ret.interp_data)
