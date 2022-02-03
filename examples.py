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


def normalize(vec):
    return vec / np.max(abs(vec))


# %%
center_wavelength_nm = 1560.
maxiter = 25
time_window_ps = 80
NPTS = 2 ** 15
ret_grating = pr.Retrieval(maxiter=maxiter, time_window_ps=time_window_ps, NPTS=NPTS,
                           center_wavelength_nm=center_wavelength_nm)
ret_grating.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")

# %%
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)

# %%
center_wavelength_nm = 1560.
maxiter = 25
time_window_ps = 10
NPTS = 2 ** 14
ret_hnlf = pr.Retrieval(maxiter=maxiter, time_window_ps=time_window_ps, NPTS=NPTS,
                        center_wavelength_nm=center_wavelength_nm)
ret_hnlf.load_data("Data/01-17-2022/realigned_spectrometer_input.txt")

# %%
center_wavelength_nm = 1560.
maxiter = 25
time_window_ps = 10
NPTS = 2 ** 12
ret_sanity = pr.Retrieval(maxiter=maxiter, time_window_ps=time_window_ps, NPTS=NPTS,
                          center_wavelength_nm=center_wavelength_nm)
ret_sanity.load_data("TestData/sanity_check_data.txt")

# %%
ret_sanity.retrieve(corr_for_pm=True,
                    start_time_fs=None,
                    end_time_fs=None,
                    plot_update=True,
                    plot_wl_um=[1, 2],
                    initial_guess_T_ps_AT=None,
                    initial_guess_wl_um_AW=None,
                    filter_um=None,
                    forbidden_um=None,
                    meas_spectrum_um=None,
                    grad_ramp_for_meas_spectrum=False,
                    i_set_spectrum_to_meas=0,
                    debug_plotting=False
                    )

spctgm, fig, axs = pr.plot_ret_results(ret_sanity.AT_ret, ret_sanity.exp_T_fs, ret_sanity.pulse,
                                       ret_sanity.interp_data, plot_um=[1.54, 1.58])
fig.suptitle("sanity output")

# %%
ret_grating.retrieve(corr_for_pm=True,
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

spctgm, fig, axs = pr.plot_ret_results(ret_grating.AT_ret, ret_grating.exp_T_fs, ret_grating.pulse,
                                       ret_grating.interp_data, plot_um=[1.54, 1.58])
fig.suptitle("grating pair")

# %%
ret_hnlf.retrieve(corr_for_pm=True,
                  start_time_fs=None,
                  end_time_fs=None,
                  plot_update=True,
                  plot_wl_um=[1, 2],
                  initial_guess_T_ps_AT=None,
                  initial_guess_wl_um_AW=None,
                  filter_um=None,
                  forbidden_um=None,
                  meas_spectrum_um=None,
                  grad_ramp_for_meas_spectrum=False,
                  i_set_spectrum_to_meas=0,
                  debug_plotting=False
                  )

spctgm, fig, axs = pr.plot_ret_results(ret_hnlf.AT_ret, ret_hnlf.exp_T_fs, ret_hnlf.pulse,
                                       ret_hnlf.interp_data, plot_um=[1., 2])
fig.suptitle("hnlf output")
