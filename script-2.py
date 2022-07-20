import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import Fiber_PPLN_NLSE as fpn
import PullDataFromOSA as OSA
import phase_retrieval as pr
import copy
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from pynlo_peter import fiber_SSFM_sim_header as sfh

clipboard_and_style_sheet.style_sheet()

data = np.load("retrieval_results_Tps_10_NPTS_2xx12.npy")
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)
osa.y = abs(osa.y)

# %% ___________________________________________________________________________________________________________________
pulse = fpn.Pulse(center_wavelength_nm=1560, time_window_ps=10, NPTS=2 ** 12)
pulse_data = copy.deepcopy(pulse)
pulse_data.set_AW_experiment(osa.x * 1e-3, osa.y ** 0.5)

# %% ___________________________________________________________________________________________________________________
# values fixed by the simulation data
T_ret = np.arange(50, 500, 5)
T_ret = np.array([np.repeat(i, 5) for i in T_ret]).flatten()

# %% ___________________________________________________________________________________________________________________
error = np.zeros(len(data))

AW_sim_end = np.zeros((len(data), 100, len(data[0])), dtype=np.complex128)

# fig, ax = plt.subplots(1, 1)
for n, at in enumerate(data):
    pulse.set_AT(at)

    spec1 = pr.normalize(pulse.AW.__abs__() ** 2)
    spec2 = pr.normalize(pulse_data.AW.__abs__() ** 2)
    error[n] = np.sqrt(np.sum((spec1 - spec2) ** 2))

    sim = sfh.simulate(pulse, sfh.fiber_adhnlf, 15, 4, 100)
    AW_sim_end[n] = sim.AW

    print(f'______________________________________________ {n} _______________________________________________________')

    # ax.clear()
    # ax.plot(osa.x * 1e-3, pr.normalize(osa.y))
    # ax.plot(pulse.wl_um, pr.normalize(pulse.AW.__abs__() ** 2))
    # ax.set_xlim(1.53, 1.59)
    # title = str(n) + "; " + str(np.round(error[n], 3))
    # if error[n] == min(error[:n + 1]):
    #     title += "*"
    # ax.set_title(title)
    # plt.pause(.1)

j_best = np.argmin(error)
pulse.set_AT(data[j_best])

# %% ___________________________________________________________________________________________________________________
# roughly speaking the best retrieval time:
error_avg = error.reshape((90, 5))
error_avg = np.mean(error_avg, 1)
spl = UnivariateSpline(T_ret[::5], error_avg, s=0.05, k=4)
T_ret_best = spl.derivative(1).roots()[-1]  # roughly 435 fs

plt.figure()
plt.plot(T_ret[::5], error_avg, 'o')
plt.plot(np.linspace(*T_ret[[0, -1]], 5000), spl(np.linspace(*T_ret[[0, -1]], 5000)))
plt.xlabel("T (fs)")
plt.ylabel("error")

plt.figure()
plt.plot(pulse.wl_um, pr.normalize(pulse_data.AW.__abs__() ** 2))
plt.plot(pulse.wl_um, pr.normalize(pulse.AW.__abs__() ** 2))
plt.xlim(1.53, 1.59)
plt.xlabel("wavelength ($\mathrm{\mu m}$)")

# %% ___________________________________________________________________________________________________________________
fig, ax = plt.subplots(1, 1)
spec_sim_end = AW_sim_end.__abs__() ** 2
ind = np.argmin(abs(pulse.F_THz)) + 1
for n, i in enumerate(spec_sim_end):
    ax.clear()
    # ax.pcolormesh(pulse.wl_um, sim.zs * 100, i, cmap='jet')
    ax.plot(pulse.wl_um[ind:], i[-1][ind:])
    ax.set_xlim(1, 2)
    ax.set_title(n)
    print(n)
    plt.pause(.2)
