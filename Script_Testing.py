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
ret = pr.Retrieval(maxiter=50)
# ret.load_data("TestData/sanity_check_data.txt")
# ret.load_data("Data/01-14-2022/successfully_symmetric_frog.txt")
ret.load_data("Data/01-17-2022/realigned_spectrometer_input.txt")

# %%
# ret.data[:] = ret.data[::-1]  # worked too! (with enough reruns but that's usual)
# ret.data[:] = (ret.data[:] + ret.data[::-1]) / 2

# %% set initial guess from power spectrum, doesn't work well
# spectrum = osa.Data("Data/01-17-2022/SPECTRUM_FOR_FROG.CSV", False)
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

# honest phase retrieval
ret.retrieve(corr_for_pm=True,
             plot_update=True,
             initial_guess_T_fs_AT=None,
             filter_um=[1.0, 1.9],
             plot_wl_um=[1, 2])

# %%
pr.apply_filter(ret.AW_ret, 0.9, 2, ret.pulse)
ret.AT_ret = pr.ifft(ret.AW_ret)
pr.plot_ret_results(ret.AT_ret, ret.exp_T_fs, ret.pulse, ret.interp_data,
                    filter_um=[.500 * 2, ret.exp_wl_nm[-1] * 2])
