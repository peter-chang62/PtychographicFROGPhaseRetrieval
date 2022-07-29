"""Retrieving this one is hopeless, just hopeless """
import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet
import PullDataFromOSA as OSA
import phase_retrieval as pr

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
ret = pr.Retrieval()
ret.load_data("Data/01-17-2022/spectrogram_realigned_spectrometer_input.txt")

# %% ___________________________________________________________________________________________________________________
ind_ll, ind_ul = 400, 1803
ret.spectrogram[:, :ind_ll] = 0.0
ret.spectrogram[:, ind_ul:] = 0.0

# %% ___________________________________________________________________________________________________________________
spectrum = np.genfromtxt("Data/01-17-2022/Spectrum_Stitched_Together_wl_nm.txt")

# %% ___________________________________________________________________________________________________________________
ret.set_signal_freq(290, 650)
ret.correct_for_phase_matching()

# %% ___________________________________________________________________________________________________________________
ret.set_initial_guess(1564.8, 5, 2 ** 12)
ret.load_spectrum_data(spectrum[:, 0] * 1e-3, spectrum[:, 1])
ret.retrieve(-275, 275, 45, iter_set=None, plot_update=True)
ret.plot_results()

# %% ___________________________________________________________________________________________________________________
# T_ret = np.arange(50, 500, 5)
# AT = np.zeros((len(T_ret) * 5, len(ret.pulse.AT)), np.complex128)
#
# h = 0
# for n, t in enumerate(T_ret):
#     for m in range(5):
#         ret.set_initial_guess(1560, 10, 2 ** 12)
#         ret.retrieve(0, t, 70, iter_set=None, plot_update=False)
#         AT[h] = ret.pulse.AT
#         h += 1
#
#         print(f'_________________________________{len(AT) - h}_________________________________________')
#
# np.save(f"retrieval_results_Tps_10_NPTS_2xx12.npy", AT)
