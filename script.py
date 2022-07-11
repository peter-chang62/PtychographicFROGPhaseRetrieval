import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet
import PullDataFromOSA as OSA
import phase_retrieval as pr

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)
osa.y = abs(osa.y)

ret = pr.Retrieval()

ret.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")
ret.set_signal_freq(367, 400)

# ret.load_data("Data/01-17-2022/realigned_spectrometer_input.txt")
# ret.set_signal_freq(284, 620)

ret.correct_for_phase_matching()
ret.set_initial_guess(1560, 10, 2 ** 12)
ret.load_spectrum_data(osa.x * 1e-3, osa.y)
ret.retrieve(0, 250, 70, iter_set=None)
ret.plot_results()
