import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet
import PullDataFromOSA as OSA
import phase_retrieval as pr

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)
osa.y = abs(osa.y)

# %% ___________________________________________________________________________________________________________________
ret = pr.Retrieval()
ret.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")

# %% ___________________________________________________________________________________________________________________
ind_ll, ind_ul = 1066, 1227
ret.spectrogram[:, :ind_ll] = 0.0
ret.spectrogram[:, ind_ul:] = 0.0

# %% ___________________________________________________________________________________________________________________
ret.set_signal_freq(350, 450)
ret.correct_for_phase_matching()

# %% ___________________________________________________________________________________________________________________
ret.set_initial_guess(1560, 20, 2 ** 11)
ret.load_spectrum_data(osa.x * 1e-3, osa.y)
ret.retrieve(-0, 435, 50, iter_set=None, plot_update=True)
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
