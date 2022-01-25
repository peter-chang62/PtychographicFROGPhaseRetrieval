import simulation_header as sh
import matplotlib.pyplot as plt
import numpy as np
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import clipboard_and_style_sheet
import scipy.constants as sc
import PullDataFromOSA as OSA

normalize = lambda vec: vec / np.max(abs(vec))


def import_retrieval(N, trans_form_limit=False):
    path = "Data/01-24-2022/consecutive_retrieval_attempts/"
    imag_path = path + "fthz_imag_{N}.txt".format(N=N)
    real_path = path + "fthz_real_{N}.txt".format(N=N)
    imag = np.genfromtxt(imag_path)
    real = np.genfromtxt(real_path)
    fthz = imag[:, 0]
    imag = imag[:, 1]
    real = real[:, 1]

    p = fpn.Pulse(center_wavelength_nm=1560., NPTS=len(fthz), time_window_ps=80)
    AW = real + 1j * imag

    if trans_form_limit:
        p.set_AW(abs(AW))
    else:
        p.set_AW(AW)
    return p


def plot_evolv(sim):
    evolv = fpn.get_2d_evolv(sim.AW)
    plt.figure()
    plt.pcolormesh(sim.pulse.wl_um, sim.zs * 100, evolv, cmap='jet')
    plt.xlim(1, 2)

    plt.figure()
    indwl = (sim.pulse.wl_um > 0).nonzero()[0]
    plt.plot(sim.pulse.wl_um[indwl], normalize(abs(sim.pulse.AW[indwl]) ** 2))
    plt.xlim(1, 2)


# %% good, imported the retrieval correctly
# osa = OSA.Data("../01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)
# plt.figure()
# plt.plot(osa.x * 1e-3, normalize(osa.y), 'o')
# p = import_retrieval(1)
# plt.plot(p.wl_um, normalize(abs(p.AW) ** 2))
# plt.xlim(1, 2)

# %%
p1 = import_retrieval(1)
sim1_ = sh.simulate(p1, sh.fiber_pm1550, 18., 3.6, 200)
sim1 = sh.simulate(sim1_.pulse, sh.fiber_adhnlf, 5, 3.5, 200)

# %%
plot_evolv(sim1)

# %%
p2 = import_retrieval(2)
sim2_ = sh.simulate(p2, sh.fiber_pm1550, 18., 4., 200)
sim2 = sh.simulate(sim2_.pulse, sh.fiber_adhnlf, 5, 3.9, 200)

# %%
plot_evolv(sim2)

# %%
p3 = import_retrieval(3)
sim3_ = sh.simulate(p3, sh.fiber_pm1550, 18, 3.75, 200)
sim3 = sh.simulate(sim3_.pulse, sh.fiber_adhnlf, 5, 3.65, 200)

# %%
plot_evolv(sim3)
