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
ret = pr.Retrieval(25)
ret.load_data("Data/01-18-2022/spectrogram_grating_pair_output.txt")

# %%
ret.retrieve(plot_update=True)

# %%
# plt.pcolormesh(ret.exp_T_fs, ret.exp_wl_nm, ret.data.T, cmap='jet')
# plt.ylim(760, 800)
