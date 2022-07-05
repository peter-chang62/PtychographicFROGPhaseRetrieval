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


def denoise(x, gamma):
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
# spectrogram = np.genfromtxt("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")
spectrogram = np.genfromtxt("TestData/sanity_check_data.txt")
T_fs = spectrogram[:, 0][1:]  # time is on the row
wl_nm = spectrogram[0][1:]  # wavelength is on the column
F_THz = sc.c * 1e-12 / (wl_nm * 1e-9)
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
# divide through by the phase-matching curve:
#   the phase-matching curve has 0 points which gives division errors. The spectrogram, however, should be heavily
#   suppressed there. so I divide through by the phase-matching curve wherever the spectrogram is above .001x its max,
#   and otherwise I set it to 0

# bbo = BBO.BBOSHG()
# R = bbo.R(wl_nm * 1e-3 * 2, 50, bbo.phase_match_angle_rad(1.55), BBO.deg_to_rad(5.0))  # 5 deg incidence?
# for n, spectrum in enumerate(spectrogram):
#     spectrogram[n] = np.where(R > 1e-3, spectrogram[n] / R, 0)

# %% ___________________________________________________________________________________________________________________
# initial guess is a sech pulse with duration based on intensity autocorrelation
x = - scint.simpson(spectrogram, x=F_THz, axis=1)
spl = spi.UnivariateSpline(T_fs, normalize(x) - .5, s=0)
roots = spl.roots()
assert len(roots) == 2, "there should only be two roots, otherwise your autocorrelation is weird"
T0 = np.diff(roots) * 0.65
pulse = fpn.Pulse(T0_ps=T0 * 1e-3, center_wavelength_nm=1560, time_window_ps=10, NPTS=2 ** 12)

# %% ___________________________________________________________________________________________________________________
# scale the experimental spectrogram to match the pulse energy
# I do this based on the integrated area under the intensity autocorrelation
x = calculate_spectrogram(pulse, T_fs)
factor = scint.simpson(scint.simpson(x)) / scint.simpson(scint.simpson(spectrogram))
spectrogram *= factor

# %% ___________________________________________________________________________________________________________________
# interpolate the spectrogram onto the simulation grid
ind_fthz = np.logical_and(pulse.F_THz * 2 >= min(F_THz), pulse.F_THz * 2 <= max(F_THz)).nonzero()[0]
gridded = spi.interp2d(F_THz, T_fs, spectrogram)
spectrogram_interp = gridded(pulse.F_THz[ind_fthz] * 2, T_fs)

# %% ___________________________________________________________________________________________________________________
# times to iterate over
start_time = 0  # fs
end_time = 250  # fs
ind_start = np.argmin(abs(T_fs - start_time))
ind_end = np.argmin(abs(T_fs - end_time))
delay_time = T_fs[ind_start:ind_end]
time_order_ps = np.c_[delay_time * 1e-3, np.arange(ind_start, ind_end)]

# %% ___________________________________________________________________________________________________________________
# phase retrieval based on:
#   [1] P. Sidorenko, O. Lahav, Z. Avnat, and O. Cohen, Ptychographic Reconstruction Algorithm for Frequency-Resolved
#   Optical Gating: Super-Resolution and Supreme Robustness, Optica 3, 1320 (2016).

rng = np.random.default_rng()
psi_j = np.zeros(pulse.AT.shape, pulse.AT.dtype)
psi_j_prime = np.zeros(pulse.AT.shape, pulse.AT.dtype)
PHI_j = np.zeros(pulse.AT.shape, pulse.AT.dtype)
AT_shift = np.zeros(pulse.AT.shape, pulse.AT.dtype)
amp = np.zeros(len(ind_fthz))
phase = np.zeros(pulse.AT.shape)
itermax = 100

fig, (ax1, ax2) = plt.subplots(1, 2)
ax3 = ax2.twinx()

for n in range(itermax):
    rng.shuffle(time_order_ps, axis=0)
    alpha = abs(0.2 + rng.standard_normal(1) / 20)
    for dt, index in time_order_ps:
        index = int(index)

        AT_shift[:] = shift(pulse.AT, pulse.V_THz, dt)
        psi_j[:] = pulse.AT * AT_shift
        PHI_j[:] = fft(psi_j)
        amp[:] = spectrogram_interp[index] ** 0.5 * wd.tukey(len(amp))
        phase[:] = np.arctan2(PHI_j.imag, PHI_j.real)
        PHI_j[ind_fthz] = amp * np.exp(1j * phase[ind_fthz])
        psi_j_prime[:] = ifft(PHI_j)
        pulse.set_AT(
            pulse.AT + alpha * AT_shift.conj() * (psi_j_prime - psi_j) / np.max(abs(AT_shift) ** 2)
        )

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.plot(pulse.T_ps, abs(pulse.AT) ** 2)
    ax2.plot(pulse.F_THz, abs(pulse.AW) ** 2)
    ax3.plot(pulse.F_THz, np.unwrap(np.arctan2(pulse.AW.imag, pulse.AW.real)), color='C1')
    fig.suptitle(n)
    plt.pause(.1)
    print(n, pulse.calc_epp())
