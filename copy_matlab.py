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
