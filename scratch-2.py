import copy
import numpy as np
import mkl_fft
import matplotlib.pyplot as plt
import BBO as BBO
import PullDataFromOSA as OSA
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.integrate as scint
import scipy.interpolate as spi
import scipy.constants as sc


def normalize(x):
    """
    :param x: normalizes the array x by abs(max(x))
    :return: normalized x
    """
    return x / np.max(abs(x))


# ______________________________________________________________________________________________________________________
# PyNLO has fft and ifft defined in reverse!
# ______________________________________________________________________________________________________________________


def ifft(x, axis=None):
    """
    :param x: array
    :param axis: if dimension is >1, specify which axis along which to perform the ifft
    :return: ifft of x

    calculates the 1D ifft of the numpy array x, if x is not 1D you need to specify the axis
    """

    assert (len(x.shape) == 1) or (isinstance(axis, int)), \
        "if x is not 1D, you need to provide an axis along which to perform the ifft"

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def fft(x, axis=None):
    """
    :param x: array
    :param axis: if dimension is >1, specify which axis along which to perform the fft
    :return: fft of x

    calculates the 1D fft of the numpy array x, if x is not 1D you need to specify the axis
    """

    assert (len(x.shape) == 1) or (isinstance(axis, int)), \
        "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def shift(x, freq, shift, axis=None, freq_is_angular=True):
    """
    :param x: 1D or 2D array
    :param freq: frequency axis
    :param shift: time shift
    :param axis: if x is 2D, the axis along which to perform the shift
    :param freq_is_angular: bool specifying whether freq is in angular frequency, default is True
    :return: shifted x

    The units of freq and shift can be anything, but they need to be consistent with each other (so if freq is in
    THz, then shift should be in ps)
    """

    assert (len(x.shape) == 1) or (isinstance(axis, int)), \
        "if x is not 1D, you need to provide an axis along which to perform the shift"
    assert isinstance(freq_is_angular, bool)

    phase = np.zeros(x.shape, dtype=np.complex128)
    ft = fft(x, axis)

    if freq_is_angular:
        V = freq
    else:
        V = freq * 2 * np.pi

    if axis is None:
        # 1D scenario
        phase[:] = np.exp(1j * V * shift)
        ft *= phase
        return ifft(ft)

    else:
        assert shift.shape == (x.shape[0],), "shift must be a 1D array, one shift for each row of x"
        phase[:] = 1j * V
        phase = np.exp(phase * np.c_[shift])
        ft *= phase
        return ifft(ft, axis)


def calculate_spectrogram(pulse, T_fs):
    """
    :param pulse: pulse instance
    :param T_fs: Time axis of the spectrogram
    :return: 2D array for the spectrogram
    """

    assert isinstance(pulse, fpn.Pulse), "pulse must be a Pulse instance"
    pulse: fpn.Pulse

    AT = np.zeros((len(T_fs), len(pulse.AT)), dtype=np.complex128)
    AT[:] = pulse.AT
    AT_shift = shift(AT, pulse.V_THz, T_fs * 1e-3, axis=1)  # THz and ps
    AT2 = AT * AT_shift
    AW2 = fft(AT2, axis=1)
    return abs(AW2) ** 2


def denoise(x, gamma):
    """
    :param x: array
    :param gamma: float that is the threshold
    :return: denoised x
    """

    # this is how Sidorenko has it implemented in his code, the one difference is that the threshold condition is on
    # abs(x), and then x - gamma * sign(x) is applied to the real and imaginary parts separately
    # Note: ____________________________________________________________________________________________________________
    #   np.sign(x) operates on the real only if you pass it a complex x:
    #   np.sign(1 + 1j) = 1 + 0j
    # __________________________________________________________________________________________________________________
    return np.where(abs(x) >= gamma, x.real - gamma * np.sign(x.real), 0) + \
           1j * np.where(abs(x) >= gamma, x.imag - gamma * np.sign(x.imag), 0)


def load_data(path):
    """
    :param path: str - path to data
    :return: wavelength (nm) 1D array, frequency (THz) 1D array, spectrogram 2D array
    """

    # extract relevant variables from the spectrogram data:
    #   1. time axis
    #   2. wavelength axis
    #   3. frequency axis
    # no alteration to the data besides truncation of the time-axis to center T0 is done here

    spectrogram = np.genfromtxt(path)
    T_fs = spectrogram[:, 0][1:]  # time is on the row
    wl_nm = spectrogram[0][1:]  # wavelength is on the column
    F_THz = sc.c * 1e-12 / (wl_nm * 1e-9)  # experimental frequency axis from wl_nm
    spectrogram = spectrogram[1:, 1:]

    # center T0
    x = scint.simps(spectrogram, axis=1)
    ind = np.argmax(x)
    ind_keep = min([ind, len(spectrogram) - ind])
    spectrogram = spectrogram[ind - ind_keep: ind + ind_keep]
    T_fs -= T_fs[ind]
    T_fs = T_fs[ind - ind_keep: ind + ind_keep]

    return wl_nm, F_THz, T_fs, normalize(spectrogram)
