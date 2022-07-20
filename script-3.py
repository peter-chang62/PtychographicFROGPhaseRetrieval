import numpy as np
import matplotlib.pyplot as plt
import phase_retrieval as pr
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

# spctgm = np.genfromtxt("Data/Nazanins_Data/201118_with all the GLASS+1more")
# wl = np.genfromtxt("Data/Nazanins_Data/Wavelength2.txt")
# T_fs = np.arange(-501, 501, 2)
# spctgm = abs((spctgm.T - spctgm.T[-1]).T)
#
# spctgm = np.vstack([wl, spctgm.T])
# spctgm = np.hstack([np.hstack([np.nan, T_fs])[:, np.newaxis], spctgm])
# np.savetxt("Data/Nazanins_Data/all_the_glass_plus_1_more_peter.txt", spctgm)

ret = pr.Retrieval()
ret.load_data("Data/Nazanins_Data/all_the_glass_plus_1_more_peter.txt")
ret.set_signal_freq(351, 456)
ret.set_initial_guess(1550, 10, 2 ** 12)
ret.retrieve(0, 300, 70, iter_set=None, plot_update=True)
