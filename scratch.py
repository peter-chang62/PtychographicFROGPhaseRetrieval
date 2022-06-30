import numpy as np
import mkl_fft
import matplotlib.pyplot as plt
import BBO as BBO
import PullDataFromOSA as OSA
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.integrate as scint
import scipy.interpolate as spi


# ______________________________________________________________________________________________________________________
# PyNLO has fft and ifft defined in reverse!
# ______________________________________________________________________________________________________________________

def normalize(x):
    return x / np.max(abs(x))


def ifft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if (len(x.shape) > 1) and (axis is None):
        raise AssertionError("if x has shape >1D you need to provide an axis along which to perform the fft")

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def fft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if (len(x.shape) > 1) and (axis is None):
        raise AssertionError("if x has shape >1D you need to provide an axis along which to perform the ifft")

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def shift(x, freq, shift, axis=None):
    if (len(x.shape) > 1) and (axis is None):
        raise AssertionError("if x has shape >1D you need to provide an axis along which to perform the shift")

    phase = np.zeros(x.shape, dtype=np.complex128)
    ft = fft(x, axis)

    if axis is None:
        # 1D scenario
        phase[:] = np.exp(1j * 2 * np.pi * freq * shift)
        ft *= phase
        return ifft(ft).real

    else:
        assert shift.shape == (x.shape[0],), "shift must be a 1D array, one shift for each row of x"
        phase[:] = 1j * 2 * np.pi * freq
        phase = np.exp(phase * np.c_[shift])
        ft *= phase
        return ifft(ft, axis).real


def calculate_spectrogram(pulse, T_fs):
    assert isinstance(pulse, fpn.Pulse), "pulse must be a Pulse instance"
    pulse: fpn.Pulse

    AT = np.zeros((len(T_fs), len(pulse.AT)), dtype=np.complex128)
    AT[:] = pulse.AT
    AT_ = shift(AT, pulse.V_THz, T_fs * 1e-3, axis=1)  # THz and ps
    AT2 = AT * AT_
    AW2 = fft(AT2, axis=1)
    return abs(AW2) ** 2


# %% ___________________________________________________________________________________________________________________
# load the experimental data
spectrogram = np.genfromtxt("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")
T_fs = spectrogram[:, 0][1:]  # time is on the row
wl_nm = spectrogram[0][1:]  # wavelength is on the column
spectrogram = spectrogram[1:, 1:]

# center T0
x = scint.simps(spectrogram, axis=1)
center = len(x) // 2
ind = np.argmax(x)
ind_keep = min([ind, len(spectrogram) - ind])
spectrogram = spectrogram[ind - ind_keep: ind + ind_keep]
T_fs -= T_fs[ind]
T_fs = T_fs[ind - ind_keep: ind + ind_keep]

# %% ___________________________________________________________________________________________________________________
# times to iterate over
start_time = 0  # fs
end_time = 250  # fs
ind_start = np.argmin(abs(T_fs - start_time))
ind_end = np.argmin(abs(T_fs - end_time))
delay_time = T_fs[ind_start:ind_end]
time_order = np.c_[delay_time, np.arange(ind_start, ind_end)]

# %% ___________________________________________________________________________________________________________________
# initial guess is a sech pulse with duration based on autocorrelation
x = scint.simps(spectrogram, axis=1)
spl = spi.UnivariateSpline(T_fs, normalize(x) - .5, s=0)
roots = spl.roots()
assert len(roots) == 2, "there should only be two roots, otherwise your autocorrelation is weird"
T0 = np.diff(roots) * 0.65
pulse = fpn.Pulse(T0_ps=T0 * 1e-3, center_wavelength_nm=1560, time_window_ps=10, NPTS=2 ** 12)

# %% ___________________________________________________________________________________________________________________
