import numpy as np
import mkl_fft
import matplotlib.pyplot as plt
import BBO as BBO
import PullDataFromOSA as OSA
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import scipy.integrate as scint
import scipy.interpolate as spi
import scipy.constants as sc
import scipy.signal.windows as wd


# ______________________________________________________________________________________________________________________
# PyNLO has fft and ifft defined in reverse!
# ______________________________________________________________________________________________________________________

def normalize(x):
    """
    :param x: normalizes the array x by abs(max(x))
    :return: normalized x
    """
    return x / np.max(abs(x))


def ifft(x, axis=None):
    """
    :param x: 1D or 2D array
    :param axis: if 2D array, specify which axis to perform the ifft
    :return: ifft of x

    calculates the 1D fft of the numpy array x if x is not 1D you need to specify the axis
    """

    if (len(x.shape) > 1) and (axis is None):
        raise AssertionError("if x has shape >1D you need to provide an axis along which to perform the fft")

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def fft(x, axis=None):
    """
    :param x: 1D or 2D array
    :param axis: if 2D array, specify which axis to perform the fft
    :return: fft of x

    calculates the 1D ifft of the numpy array x if x is not 1D you need to specify the axis
    """

    if (len(x.shape) > 1) and (axis is None):
        raise AssertionError("if x has shape >1D you need to provide an axis along which to perform the ifft")

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

    if (len(x.shape) > 1) and (axis is None):
        raise AssertionError("if x has shape >1D you need to provide an axis along which to perform the shift")

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
    AT_ = shift(AT, pulse.V_THz, T_fs * 1e-3, axis=1)  # THz and ps
    AT2 = AT * AT_
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


# %% ___________________________________________________________________________________________________________________
# load the experimental data
# spectrogram = np.genfromtxt("TestData/sanity_check_data.txt")
spectrogram = np.genfromtxt("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")
# spectrogram = np.genfromtxt("Data/01-17-2022/realigned_spectrometer_input.txt")
T_fs = spectrogram[:, 0][1:]  # time is on the row
wl_nm = spectrogram[0][1:]  # wavelength is on the column
F_THz = sc.c * 1e-12 / (wl_nm * 1e-9)  # experimental frequency axis from wl_nm
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
# denoise the spectrogram, I think it helps
spectrogram = normalize(spectrogram)
spectrogram = denoise(spectrogram, 1e-3).real

# %% ___________________________________________________________________________________________________________________
# divide through by the phase-matching curve: the phase-matching curve has 0 points which gives division errors. The
# spectrogram, however, should be heavily suppressed there. So, I divide through by the phase-matching curve wherever
# spectrogram >= 1e-3 its max, and otherwise I set it to 0

bbo = BBO.BBOSHG()
R = bbo.R(wl_nm * 1e-3 * 2, 50, bbo.phase_match_angle_rad(1.55), BBO.deg_to_rad(5.0))  # 5 deg incidence?
for n, spectrum in enumerate(spectrogram):
    spectrogram[n] = np.where(spectrogram[n] >= 1e-3 * spectrogram.max(), spectrogram[n] / R, 0)

# %% ___________________________________________________________________________________________________________________
# initial guess is a sech pulse with duration based on intensity autocorrelation
x = - scint.simpson(spectrogram, x=F_THz, axis=1)  # integrate experimental spectrogram across wavelength axis
spl = spi.UnivariateSpline(T_fs, normalize(x) - .5, s=0)
roots = spl.roots()
T0 = np.diff(roots[[0, -1]]) * 0.65 / 1.76
pulse = fpn.Pulse(T0_ps=T0 * 1e-3, center_wavelength_nm=1560, time_window_ps=10, NPTS=2 ** 12)

# %% ___________________________________________________________________________________________________________________
# interpolate the spectrogram onto the simulation grid
ind_fthz = np.logical_and(pulse.F_THz * 2 >= min(F_THz), pulse.F_THz * 2 <= max(F_THz)).nonzero()[0]
gridded = spi.interp2d(F_THz, T_fs, spectrogram)
spectrogram_interp = gridded(pulse.F_THz[ind_fthz] * 2, T_fs)

# %% ___________________________________________________________________________________________________________________
# scale the interpolated spectrogram to match the pulse energy
# I do it here instead of to the experimental spectrogram, because the
# interpolated spectrogram has the same integration frequency axis as the pulse instance
x = calculate_spectrogram(pulse, T_fs)
factor = scint.simpson(scint.simpson(x[:, ind_fthz])) / scint.simpson(scint.simpson(spectrogram_interp))
spectrogram_interp *= factor

# %% ___________________________________________________________________________________________________________________
# times to iterate over
start_time = 0  # fs
end_time = 250  # fs
ind_start = np.argmin(abs(T_fs - start_time))
ind_end = np.argmin(abs(T_fs - end_time))
delay_time = T_fs[ind_start:ind_end]
time_order_ps = np.c_[delay_time * 1e-3, np.arange(ind_start, ind_end)]

# %% ___________________________________________________________________________________________________________________
# From here we should be ready for phase retrieval!
# %% ___________________________________________________________________________________________________________________

# phase retrieval based on:
#   [1] P. Sidorenko, O. Lahav, Z. Avnat, and O. Cohen, Ptychographic Reconstruction Algorithm for Frequency-Resolved
#   Optical Gating: Super-Resolution and Supreme Robustness, Optica 3, 1320 (2016).

j_excl = np.ones(len(pulse.F_THz))
j_excl[ind_fthz] = 0
j_excl = j_excl.nonzero()[0]  # everything but ind_fthz

itermax = 40
error = np.zeros(itermax)
rng = np.random.default_rng()

AT = np.zeros((itermax, len(pulse.AT)), dtype=np.complex128)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax3 = ax2.twinx()

for iter in range(itermax):
    rng.shuffle(time_order_ps, axis=0)
    alpha = abs(0.2 + rng.standard_normal(1) / 20)
    for dt, j in time_order_ps:
        j = int(j)

        AT_shift = shift(pulse.AT, pulse.V_THz, dt)
        psi_j = AT_shift * pulse.AT
        phi_j = fft(psi_j)
        amp = abs(phi_j)
        amp[ind_fthz] = np.sqrt(spectrogram_interp[j])
        phase = np.arctan2(phi_j.imag, phi_j.real)
        phi_j[:] = amp * np.exp(1j * phase)
        phi_j[j_excl] = denoise(phi_j[j_excl], 1e-3)
        psi_jp = ifft(phi_j)
        corr1 = AT_shift.conj() * (psi_jp - psi_j) / np.max(abs(AT_shift) ** 2)
        corr2 = pulse.AT.conj() * (psi_jp - psi_j) / np.max(abs(pulse.AT) ** 2)
        corr2 = shift(corr2, pulse.V_THz, -dt)
        pulse.set_AT(
            pulse.AT + alpha * corr1
            + alpha * corr2
        )

    [ax.clear() for ax in [ax1, ax2, ax3]]
    ax1.plot(pulse.T_ps, pulse.AT.__abs__() ** 2)
    ax2.plot(pulse.F_THz, pulse.AW.__abs__() ** 2)
    ax3.plot(pulse.F_THz, np.unwrap(np.arctan2(pulse.AW.imag, pulse.AW.real)), color='C1')
    ax2.set_xlim(188, 198)
    plt.pause(.1)

    s = calculate_spectrogram(pulse, T_fs)[:, ind_fthz]
    error[iter] = np.sqrt(np.sum(abs(s - spectrogram_interp) ** 2)) / np.sqrt(np.sum(abs(spectrogram_interp) ** 2))
    AT[iter] = pulse.AT

    print(iter, error[iter])
