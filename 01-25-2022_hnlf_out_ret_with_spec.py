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
osa_short = OSA.Data("Data/01-17-2022/SPECTRUM_FOR_FROG.CSV", False)
osa_long = OSA.Data("Data/01-17-2022/SPECTRUM_HNLF_2um_OSA.CSV", False)

gridded_short = spi.interp1d(osa_short.x, normalize(osa_short.y), bounds_error=False, fill_value=0.0)
gridded_long = spi.interp1d(osa_long.x, normalize(osa_long.y), bounds_error=False, fill_value=0.0)
wl = np.linspace(osa_short.x[0], osa_long.x[-1], 5000)

specshort = gridded_short(wl)
speclong = gridded_long(wl)

spec = np.where(wl < 1450, specshort, speclong)
window = tukey(4500, .15)
window = np.hstack((window, np.zeros(len(spec) - len(window))))
spec_windowed = spec * window
final = np.hstack((wl[:, np.newaxis], spec_windowed[:, np.newaxis]))

# np.savetxt("Data/01-17-2022/Spectrum_Stitched_Together_wl_nm.txt", final)
