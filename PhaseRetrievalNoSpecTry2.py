import numpy as np
import scipy.constants as sc
import BBO as BBO
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.interpolate as spi
import pypret as pyp
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


def calculate_error(AT, delay_window_fs, T_fs, ref_spctgm):
    dT_fs = np.mean(np.diff(T_fs))
    N = delay_window_fs / dT_fs
    nrolls = np.arange(0, N).astype(int)
    spectrogram = np.zeros((len(nrolls), len(AT))).astype(np.complex128)

    for n, roll in enumerate(nrolls):
        spectrogram[n][:] = np.roll(AT, roll) * AT

    spectrogram = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spectrogram, axes=1)), axes=1)
    spectrogram = abs(spectrogram) ** 2

    num = np.sqrt(np.sum((spectrogram - ref_spctgm[nrolls[0]:nrolls[-1] + 1]) ** 2))
    denom = np.sqrt(np.sum((ref_spctgm[nrolls[0]:nrolls[-1] + 1]) ** 2))
    return num / denom


def fft(x):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def ifft(x):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))


# %% Load the Data
DATA = np.genfromtxt("TestData/new_alignment_method.txt")

# %% time and frequency axis
T_fs = DATA[:, 0][1:]
wl_nm = DATA[0][1:]
data = DATA[:, 1:][1:]

# %% useful variables
F_mks = sc.c / (wl_nm * 1e-9)
dT_fs = np.mean(np.diff(T_fs))

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

# %% initial guess
pulse = fpn.Pulse(T0_ps=0.02,
                  center_wavelength_nm=1560,
                  time_window_ps=10,
                  NPTS=2 ** 14)

# %% interpolate experimental data onto the simulation grid
gridded = spi.interp2d(F_mks, T_fs, dataCorrected)
spctgm = gridded(pulse.F_mks * 2, T_fs)

# %% ffftshifted spectrogram
spctgm_fftshift = np.fft.ifftshift(spctgm, axes=0)

# %% if the spectrogram is to be replicated,
# the power needs to match, so scale the pulse field accordingly
AW2 = fft(pulse.AT ** 2)
power_AW2 = simps(abs(AW2) ** 2)
power_spctgm = simps(spctgm_fftshift[0])
scale_power = (power_spctgm / power_AW2) ** .25
AT = pulse.AT * scale_power

# %% phase retrieval
scale_dT = dT_fs / (pulse.dT_ps * 1e3)
rng = np.random.default_rng()

ATshift = np.zeros(AT.shape, dtype=AT.dtype)
Psi = np.zeros(AT.shape, dtype=AT.dtype)
PsiPrime = np.zeros(AT.shape, dtype=AT.dtype)
Phi = np.zeros(AT.shape, dtype=AT.dtype)
phase = np.zeros(len(AT))
amp = np.zeros(len(AT))

result = []
error = []

maxiter = 100
print("initial error:", calculate_error(AT, 400, pulse.T_ps * 1e3, spctgm))
for it in range(maxiter):
    time_order = np.arange(0, 200, 1)
    rng.shuffle(time_order)
    alpha = rng.uniform(low=.1, high=.5)

    for roll_spctgm in time_order:
        roll_pulse = round(roll_spctgm * scale_dT)

        ATshift[:] = np.roll(AT, roll_pulse)
        Psi[:] = AT * ATshift
        Phi[:] = fft(Psi)

        phase[:] = np.arctan2(Phi.imag, Phi.real)
        amp[:] = np.sqrt(np.roll(spctgm_fftshift, roll_spctgm, axis=0)[0])
        Phi[:] = amp * np.exp(1j * phase)
        Phi[:] = threshold_operation(Phi, 1e-4)
        PsiPrime[:] = ifft(Phi)

        AT[:] += alpha * (ATshift.conj() / max(abs(ATshift) ** 2)) * (PsiPrime - Psi)

        AT[:] += np.roll((AT.conj() / max(abs(AT) ** 2)) * (PsiPrime - Psi), -roll_pulse)

    err = calculate_error(AT, 400, pulse.T_ps * 1e3, spctgm)
    print(it, err)
    result.append(AT)
    error.append(err)

error = np.array(error)
result = np.array(result)
