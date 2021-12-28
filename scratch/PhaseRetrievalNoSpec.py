import numpy as np
import scipy.constants as sc
import BBO as BBO
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from scipy.integrate import simps


def normalize(vec):
    return vec / np.max(abs(vec))


def threshold_operation(x, threshold):
    return np.where(x < threshold, 0, x - threshold * np.sign(x))


def calculate_spectrogram(AT, delay_window_fs, T_fs):
    dT_fs = np.mean(np.diff(T_fs))
    N = delay_window_fs / dT_fs
    nrolls = np.arange(-N // 2, N // 2, 1).astype(int)
    spectrogram = np.zeros((len(nrolls), len(AT))).astype(np.complex128)
    for n, roll in enumerate(nrolls):
        spectrogram[n][:] = np.roll(AT, roll) * AT
    spectrogram = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spectrogram, axes=1)), axes=1)
    return abs(spectrogram) ** 2


# %% Load the Data
DATA = np.genfromtxt("../TestData/new_alignment_method.txt")

# %% time and frequency axis
T_fs = DATA[:, 0][1:]
dT_fs = np.mean(np.diff(T_fs))
wl_nm = DATA[0][1:]
nu_mks = sc.c / (wl_nm * 1e-9)
data = DATA[:, 1:][1:]

# %% center T0
ind_max = np.unravel_index(np.argmax(data), data.shape)[0]
data = np.roll(data, -(ind_max - len(data) // 2), axis=0)

# %% phasematching
bbo = BBO.BBOSHG()
R = bbo.R(wl_um=wl_nm * 1e-3 * 2,  # fundamental wavelength
          length_um=50.,  # crystal thickness
          theta_pm_rad=bbo.phase_match_angle_rad(1.55),  # crystal was phasematched for 1550 nm
          alpha_rad=np.arctan(.25 / 2)
          )

# %% correct for phasematching
ind500nm = (wl_nm > 500).nonzero()[0]
dataCorrected = np.copy(data)
dataCorrected[:, ind500nm] /= R[ind500nm]

# %% zoomed in relevant time window for spectrogram: -200fs to 200fs
ll = np.argmin((T_fs + 200) ** 2)
ul = np.argmin((T_fs - 200) ** 2)
T_fs_zoom = T_fs[ll:ul]
dataCorrected_zoom_T = dataCorrected[ll:ul]

# %% initial guess
pulse = fpn.Pulse(T0_ps=0.02, center_wavelength_nm=1560, time_window_ps=10, NPTS=2 ** 14)
E0 = pulse.AT

# %% interpolate experimental spectrogram onto simulation grid
gridded = spi.interp2d(nu_mks, T_fs_zoom, dataCorrected_zoom_T, kind='linear', fill_value=0.0)
spectrogram = normalize(gridded(pulse.F_mks * 2.0, T_fs_zoom))

# %%
scale = dT_fs / (pulse.dT_ps * 1e3)
maxiter = 300
spectrogram_fftshift = np.fft.ifftshift(spectrogram, axes=0)

area_spectrogram_center = simps(spectrogram_fftshift[0])
E0W2 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E0 ** 2)))
area_E0W2 = simps(abs(E0W2) ** 2)
E0 *= (area_spectrogram_center / area_E0W2) ** 0.25

Ej = np.copy(E0)
psi = np.zeros(Ej.shape, dtype=Ej.dtype)

rng = np.random.default_rng()

for i in range(2):
    random_order_index = np.arange(len(T_fs_zoom) // 2)
    np.random.shuffle(random_order_index)
    alpha = rng.uniform(low=0.1, high=0.5)

    # inside the for loop
    for n in random_order_index:
        nrolls = round(n * scale)

        Ejshifted = np.roll(E0, nrolls)
        psi[:] = Ej * Ejshifted

        Phi = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi)))
        phase = np.arctan2(Phi.imag, Phi.real)
        amp = np.sqrt(np.roll(spectrogram_fftshift, n, axis=0)[0])
        Phi = amp * np.exp(1j * phase)
        Phi = threshold_operation(Phi, 1e-4)

        psi_prime = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Phi)))

        Ej += alpha * (Ejshifted.conj() / (max(abs(Ejshifted) ** 2))) * (psi_prime - psi)

    print(i)
