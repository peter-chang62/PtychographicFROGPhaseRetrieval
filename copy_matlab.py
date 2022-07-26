"""I will exactly copy Henry's matlab code here, to help see if there is an obvious mistake to my code """
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sintp
import scipy.interpolate as sintg
import scipy.optimize as so
import scipy.constants as sc


def Denoise(M, gamma):
    return np.where(M < gamma, 0, np.sign(M) * (abs(M) - gamma * abs(M)))


def makePFROG(Et, Gt, Omega, Delay):
    Gw = np.zeros((len(Delay), len(Omega)), dtype=np.complex128)
    Gw[:] = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Gt)))
    Gw *= np.exp(1j * np.c_[Delay] * Omega)
    Gttau = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Gw, axes=1), axis=1), axes=1)
    Sit = Gttau * Et
    Siw = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Sit, axes=1), axis=1), axes=1)
    Si = abs(Siw) ** 2
    return Si.T


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
# interpolate FROG trace onto simulation grid (interpolates twice with interp1d)
NWpnts = 2 ** 11
NDpnts = int(np.floor(len(Delay) / 2) * 2)
OmegaExp = np.linspace(Omega[-1], Omega[0], NWpnts)

S1 = np.zeros((len(OmegaExp), len(Delay)))
for i in range(len(Delay)):
    S1[:, i] = sintp.interp1d(Omega, S[:, i])(OmegaExp)
    S1[np.isnan(S1)] = 0

DelayExp = np.linspace(Delay[0], Delay[-1], NDpnts)

S2 = np.zeros((len(OmegaExp), len(DelayExp)))
for i in range(len(OmegaExp)):
    S2[i, :] = sintp.interp1d(Delay, S1[i, :])(DelayExp)
    S2[np.isnan(S2)] = 0

DelayRange = np.diff(Delay[[0, -1]])[0]
DelayExp = np.linspace(-DelayRange / 2, DelayRange / 2, NDpnts)

# %% ___________________________________________________________________________________________________________________
# Divide out BBO phase-matching curve
PMCurve = np.genfromtxt("ptych_FROG_Timmers/BBO_50um_PhaseMatchingCurve.txt")
PMLambda = PMCurve[:, 0]
PMOmega = 2 * np.pi * 0.3 / PMLambda
PMEff = PMCurve[:, 1]

PMEffExp = sintp.interp1d(PMOmega, PMEff)(OmegaExp)
PMEffExp[np.isnan(PMEffExp)] = 1

for i in range(len(DelayExp)):
    S2[:, i] /= PMEffExp

# %% ___________________________________________________________________________________________________________________
# Background Subtract and normalization
Bckgnd = S2[-1, :]
Bckgnd = np.sum(Bckgnd) / NDpnts
S2 -= Bckgnd
S2[S2 < 0] = 0

SExp = S2 / np.max(S2)

# %% ___________________________________________________________________________________________________________________
# Initial Guess for Pulse and Gate Field
DeltaOmega = np.diff(OmegaExp[[0, 1]])[0]
tnyq = 2 * np.pi / DeltaOmega
t = np.linspace(-tnyq / 2, tnyq, NWpnts)

Et = np.sum(SExp, 0).astype(np.complex128)
Et *= np.exp(1j * np.random.uniform(size=NDpnts) * np.pi / 8)
Et = sintp.interp1d(DelayExp, Et, bounds_error=False, fill_value=0)(t)
Gt = Et

# %% ___________________________________________________________________________________________________________________
# Reconstruction

Pti = Et
Gti = Gt
Err = np.zeros(int(Iter))
Ptvec = np.zeros((int(Iter), len(Pti)), dtype=np.complex128)
Gtvec = np.zeros((int(Iter), len(Pti)), dtype=np.complex128)

fig, ax = plt.subplots(3, 2)
ax_ = ax[1, 0].twinx()
ax2_ = ax[1, 1].twinx()

for n in range(int(Iter / Update)):
    for m in range(Update):
        j = np.random.permutation(len(DelayExp))
        for i in range(NDpnts):
            jiter = j[i]
            Delayi = DelayExp[jiter]

            # UPDATE PULSE FIELD
            # time delayed gate pulse
            Gwi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Gti)))
            Gtishift = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Gwi * np.exp(1j * OmegaExp * Delayi))))

            # Fourier Transform of product field
            chiti = Gtishift * Pti
            chiwi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(chiti)))
            chiwi = Denoise(np.real(chiwi), 1e-3) + 1j * Denoise(np.imag(chiwi), 1e-3)

            # replace modulus with the measurement
            chiwiprime = np.sqrt(SExp[:, jiter]) * np.exp(1j * np.arctan2(chiwi.imag, chiwi.real))

            # inverse Fourier transformation
            chitiprime = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(chiwiprime)))

            # difference
            Deltachiti = chitiprime - chiti

            # update object function
            Pupdate = np.conj(Gtishift) / np.max(abs(Gtishift) ** 2)
            beta_P = np.random.uniform(10 / 100, 30 / 100)
            Ptirecon = Pti + beta_P * Pupdate * Deltachiti

            Indt0 = np.argmax(Ptirecon)
            Ptirecon = np.roll(Ptirecon, int(NWpnts / 2 - Indt0))

            Pti = Ptirecon
            Pti[np.isnan(Pti)] = 0
            Gti = Ptirecon
            Gti[np.isnan(Gti)] = 0

        # Update reconstructed FROG trace and calculate error
        Si = makePFROG(Pti, Gti, OmegaExp, DelayExp)
        Si = Si / np.max(Si)
        fun = lambda gamma: np.sqrt(np.sum(abs(Si - gamma * SExp) ** 2) / (NDpnts * NWpnts))
        res = so.minimize(fun, np.array([1]))
        [gamma_min, err] = res.fun, res.x

        Err[(n - 1) * Update + m] = err
        Ptvec[(n - 1) * Update + m, :] = Pti
        Gtvec[(n - 1) * Update + m, :] = Gti

        print(m)

        [i.clear() for i in ax.flatten()]
        ax[0, 0].pcolormesh(DelayExp, OmegaExp, SExp, cmap='jet')
        ax[0, 0].set_xlabel("Delay (fs)")
        ax[0, 0].set_ylabel("Frequency (rad/fs)")

        ax[0, 1].pcolormesh(DelayExp, OmegaExp, Si)
        ax[0, 1].set_xlabel("Delay (fs)")
        ax[0, 1].set_ylabel("Frequency (rad/fs)")

        ax[1, 0].plot(t, abs(Pti) ** 2)
        ax_.plot(t, np.unwrap(np.arctan2(Pti.imag, Pti.real)))
        ax[1, 0].set_xlim(-500, 500)
        ax[1, 0].set_xlabel("Time (fs)")
        ax[1, 0].set_ylabel("Amplitude (arb. units)")

        y = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Pti)))
        ax[1, 1].plot(OmegaExp, abs(y) ** 2)
        ax2_.plot(OmegaExp, np.unwrap(np.arctan2(y.imag, y.real)))
        ax[1, 1].set_xlabel("Frequency (rad/fs)")
        ax[1, 1].set_ylabel("Amplitude (arb. units)")

        ax[2, 0].plot(Err)
        ax[2, 0].set_xlabel("Iteration")
        ax[2, 0].set_ylabel("Error")

        plt.pause(.25)
