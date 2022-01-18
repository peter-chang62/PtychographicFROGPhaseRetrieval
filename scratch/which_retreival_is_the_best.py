# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/peterchang/Documents/Github/PtychographicFROGPhaseRetrieval/")
import PullDataFromOSA as osa
import scipy.constants as sc
import scipy.interpolate as spi


def normalize(vec):
    return vec / np.max(abs(vec))


# %%
path = "../Data/01-17-2022/"
path2 = "../Data/01-14-2022/"
file_osa = "SPECTRUM_FOR_FROG.CSV"
spec = osa.Data(path + file_osa, data_is_log=False)

# %%
imag = np.genfromtxt(path + "retrieved_no_spec_imag_f_thz.txt")
real = np.genfromtxt(path + "retrieved_no_spec_real_f_thz.txt")
fthz = imag[:, 0]

AW = real[:, 1] + 1j * imag[:, 1]

wl_um = sc.c * 1e6 / (fthz * 1e12)
ind = (wl_um > 0).nonzero()[0]
wl_um = wl_um[ind]
AW = AW[ind]

plt.figure()
plt.plot(wl_um, normalize(abs(AW) ** 2))
plt.plot(spec.x * 1e-3, normalize(spec.y))
plt.xlim(1, 2)

# %%
ind_wl = np.logical_and(wl_um > min(spec.x * 1e-3), wl_um < max(spec.x * 1e-3)).nonzero()[0]
spl = spi.interp1d(spec.x, spec.y)
interp_osa = spl(wl_um[ind_wl] * 1e3)

# %%
diff1 = normalize(abs(AW[ind_wl]) ** 2) - normalize(interp_osa)
plt.title("%.5f" % np.mean(diff1 ** 2))

# %%
imag = np.genfromtxt(path2 + "retrieval_no_spec_imag_f_thz.txt")
real = np.genfromtxt(path2 + "retrieval_no_spec_real_f_thz.txt")
fthz = imag[:, 0]

AW = real[:, 1] + 1j * imag[:, 1]
AW = AW[ind]

plt.figure()
plt.plot(wl_um, normalize(abs(AW) ** 2))
plt.plot(spec.x * 1e-3, normalize(spec.y))
plt.xlim(1, 2)

# %%
diff2 = normalize(abs(AW[ind_wl]) ** 2) - normalize(interp_osa)
plt.title("%.5f" % np.mean(diff2 ** 2))

# %%

imag = np.genfromtxt(path2 + "retrieval_no_spec_imag_f_thz_2.txt")
real = np.genfromtxt(path2 + "retrieval_no_spec_real_f_thz_2.txt")
fthz = imag[:, 0]

AW = real[:, 1] + 1j * imag[:, 1]
AW = AW[ind]

plt.figure()
plt.plot(wl_um, normalize(abs(AW) ** 2))
plt.plot(spec.x * 1e-3, normalize(spec.y))
plt.xlim(1, 2)

# %%
diff3 = normalize(abs(AW[ind_wl]) ** 2) - normalize(interp_osa)
plt.title("%.5f" % np.mean(diff3 ** 2))

"""The best one is diff2: so 01-14-2022/retrieval_no_spec_imag_f_thz.txt is likely the best retrieval you have
"""
