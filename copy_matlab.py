"""I will exactly copy Henry's matlab code here, to help see if there is an obvious mistake to my code """
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sintp
import scipy.interpolate as sintg
import scipy.constants as sc

Iter = 500
Update = 50
S = np.genfromtxt("Data/Nazanins_Data/201118_with all the GLASS+1more")
Delay0 = np.arange(-500, 500 + 2, 2)
Wavelength = np.genfromtxt("Data/Nazanins_Data/Wavelength2.txt")
Omega = 2 * np.pi * 300 / Wavelength

# %% ___________________________________________________________________________________________________________________
autocorrelation = np.sum(S, 0)
PeakInd1 = np.argmax(autocorrelation)
PeakInd2 = len(Delay0) - PeakInd1
PeakIndVec = [PeakInd1, PeakInd2]
PeakInd = min(PeakIndVec)
Which = np.argmin(PeakIndVec)

if Which == 0:
    raise AssertionError("hello world")

else:
    Delay = Delay0[len(Delay0) - 2 * (PeakInd - 1) + 1::]
    S = S[:, len(Delay0) - 2 * (PeakInd - 1) + 1::]

# %% ___________________________________________________________________________________________________________________
# in reverse, not sure why but whatever
w1 = 2
w2 = 3
Ind3 = np.argmin(abs(Omega - w1))
Ind4 = np.argmin(abs(Omega - w2))
Omega = Omega[Ind4:Ind3 + 1]
S = S[Ind4:Ind3 + 1, :]

# %% ___________________________________________________________________________________________________________________
