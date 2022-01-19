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

# %%
ret = pr.Retrieval(25, time_window_ps=30, NPTS=2 ** 13)
ret.load_data("Data/01-18-2022/spectrogram_grating_pair_output.txt")
# ret.data[:] = ret.data[::-1]

# %%
spec = osa.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", data_is_log=False)

# %%
ret.retrieve(corr_for_pm=True,
             plot_update=True,
             meas_spectrum_um=[spec.x * 1e-3, spec.y],
             filter_um=[1.54, 1.58],
             i_set_spectrum_to_meas=10,
             plot_wl_um=[1.54, 1.58])

# ret.retrieve(corr_for_pm=True,
#              plot_update=True,
#              meas_spectrum_um=None,
#              filter_um=[1.54, 1.58],
#              i_set_spectrum_to_meas=0,
#              plot_wl_um=[1.54, 1.58])

# %%
# plt.pcolormesh(ret.exp_T_fs, ret.exp_wl_nm, ret.data.T, cmap='jet')
# plt.ylim(760, 800)

# %%
spec = pr.calculate_spctgm(ret.AT_ret, ret.exp_T_fs, ret.pulse)
fig, axs = plt.subplots(1, 2)
axs[0].pcolormesh(ret.exp_T_fs, ret.pulse.wl_um, ret.interp_data.T, cmap='jet')
axs[1].pcolormesh(ret.exp_T_fs, ret.pulse.wl_um, spec.T, cmap='jet')
[i.set_ylim(1.53, 1.59) for i in axs]
fig.suptitle("0 " + str(min(ret.error)))

# %%
# centerexp = np.fft.fftshift(ret.interp_data, axes=0)[0]
# centercalc = np.fft.fftshift(spec, axes=0)[0]
# ind = (ret.pulse.wl_um > 0).nonzero()[0]
# plt.figure()
# plt.plot(ret.pulse.wl_um[ind], centerexp[ind])
# plt.plot(ret.pulse.wl_um[ind], centercalc)
# plt.xlim(1.54, 1.58)
