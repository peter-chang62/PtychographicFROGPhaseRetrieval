import numpy as np
import matplotlib.pyplot as plt
import phase_retrieval as pr
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
# spctgm = np.genfromtxt("Data/Nazanins_Data/201118_with all the GLASS+1more").T
# wl = np.genfromtxt("Data/Nazanins_Data/Wavelength2.txt")
# T_fs = np.arange(-501, 501, 2)
# spctgm[spctgm != 0] -= np.mean(spctgm[0][spctgm[0] != 0])
# spctgm[spctgm < 0] = 0.0
#
# spctgm = np.vstack([wl, spctgm])
# spctgm = np.hstack([np.hstack([np.nan, T_fs])[:, np.newaxis], spctgm])
# np.savetxt("Data/Nazanins_Data/all_the_glass_plus_1_more_peter.txt", spctgm)

# %% ___________________________________________________________________________________________________________________
ret = pr.Retrieval()
ret.load_data("Data/Nazanins_Data/all_the_glass_plus_1_more_peter.txt")
ll, ul = 2 * 1e3 / (2 * np.pi), 3 * 1e3 / (2 * np.pi)
ret.set_signal_freq(ll, ul)
ret.set_initial_guess(1550, 15, 2 ** 12)
ret.retrieve(-478, 478, 70, iter_set=None, plot_update=True)
