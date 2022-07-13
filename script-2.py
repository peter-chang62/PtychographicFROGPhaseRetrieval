import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import Fiber_PPLN_NLSE as fpn
import PullDataFromOSA as OSA
import phase_retrieval as pr

clipboard_and_style_sheet.style_sheet()

data = np.load("retrieval_results_Tps_10_NPTS_2xx12.npy")
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)
osa.y = abs(osa.y)

# %% ___________________________________________________________________________________________________________________
pulse = fpn.Pulse(center_wavelength_nm=1560, time_window_ps=10, NPTS=2 ** 12)

fig, ax = plt.subplots(1, 1)
for n, at in enumerate(data):
    pulse.set_AT(at)

    ax.clear()
    ax.plot(osa.x * 1e-3, pr.normalize(osa.y))
    ax.plot(pulse.wl_um, pr.normalize(pulse.AW.__abs__() ** 2))
    ax.set_xlim(1.53, 1.59)
    ax.set_title(n)
    plt.pause(.1)
