import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import clipboard_and_style_sheet
import scipy.constants as sc
import scipy.interpolate as spi


# dB/km to 1/m
def dBkm_to_m(dBkm):
    km = 10 ** (-dBkm / 10)
    return km * 1e-3


def simulate(pulse, fiber, length_cm, epp_nJ, nsteps=100):
    pulse: fpn.Pulse
    fiber: fpn.Fiber
    _ = copy.deepcopy(fiber)
    _.length = length_cm * .01
    __ = copy.deepcopy(pulse)
    __.set_epp(epp_nJ * 1.e-9)
    return fpn.FiberFourWaveMixing().propagate(__, _, nsteps)


def get_2d_time_evolv(at2d):
    norm = np.max(abs(at2d) ** 2, axis=1)
    toplot = abs(at2d) ** 2
    toplot = (toplot.T / norm).T
    return toplot


def plot_freq_evolv(sim, ax=None, xlims=None):
    evolv = fpn.get_2d_evolv(sim.AW)

    ind = (sim.pulse.wl_um > 0).nonzero()

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.pcolormesh(sim.pulse.wl_um[ind], (sim.zs * 100.), evolv[:, ind][:, 0, :],
                  cmap='jet',
                  shading='auto')

    if xlims is None:
        ax.set_xlim(1, 2)
    else:
        ax.set_xlim(*xlims)
    ax.set_xlabel("$\mathrm{\mu m}$")
    ax.set_ylabel("cm")


def plot_time_evolv(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    toplot = get_2d_time_evolv(sim.AT)
    ax.pcolormesh(sim.pulse.T_ps, (sim.zs * 100.), toplot, cmap='jet',
                  shading='auto')
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("ps")
    ax.set_ylabel("cm")


def video(sim, save=False, figsize=[12.18, 4.8], xlims=None):
    awevolv = fpn.get_2d_evolv(sim.AW)
    atevolv = get_2d_time_evolv(sim.AT)

    ind = (sim.pulse.wl_um > 0).nonzero()

    if save:
        plt.ioff()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if xlims is None:
        xlims = [1.3, 1.9]

    if save:
        for i in range(len(sim.zs)):
            ax1.clear()
            ax2.clear()
            ax2.set_xlim(*xlims)
            ax1.set_xlim(-1, 1)
            ax1.set_xlabel("ps")
            ax2.set_xlabel("$\mathrm{\mu m}$")

            ax1.plot(sim.pulse.T_ps, atevolv[i])
            ax2.plot(sim.pulse.wl_um[ind], awevolv[i][ind])
            fig.suptitle('%.2f' % (sim.zs[i] * 100.))
            plt.savefig("figuresForVideos/" + str(i) + ".png")

            print(str(i + 1) + "/" + str(len(sim.zs)))

        plt.ion()
        return

    for i in range(len(sim.zs)):
        ax1.clear()
        ax2.clear()
        ax2.set_xlim(*xlims)
        ax1.set_xlim(-1, 1)
        ax1.set_xlabel("ps")
        ax2.set_xlabel("$\mathrm{\mu m}$")

        ax1.plot(sim.pulse.T_ps, atevolv[i])
        ax2.plot(sim.pulse.wl_um[ind], awevolv[i][ind])
        fig.suptitle('%.2f' % (sim.zs[i] * 100.))
        plt.pause(.1)


def create_mp4(fps, name):
    command = "ffmpeg -r " + \
              str(fps) + \
              " -f image2 -s 1920x1080 -y -i figuresForVideos/%d.png " \
              "-vcodec libx264 -crf 25  -pix_fmt yuv420p " + \
              name
    os.system(command)


# fiber Parameters:
# OFS AD HNLF parameters
adhnlf = {
    "D": 5.4,
    "Dprime": 0.028,
    "gamma": 10.9,
    "Alpha": 0.74,
}

# OFS ND HNLF parameters
ndhnlf = {
    "D": -2.6,
    "Dprime": 0.026,
    "gamma": 10.5,
    "Alpha": 0.8,
}

pm1550 = {
    "D": 18,
    "Dprime": 0.0612,
    "gamma": 1.,
    "Alpha": 0.18
}

fiber_adhnlf = fpn.Fiber()
fiber_adhnlf.generate_fiber(.2,
                            1550.,
                            [adhnlf["D"], adhnlf["Dprime"]],
                            adhnlf["gamma"] * 1e-3,
                            gain=dBkm_to_m(adhnlf["Alpha"]),
                            dispersion_format="D")

fiber_ndhnlf = fpn.Fiber()
fiber_ndhnlf.generate_fiber(.2,
                            1550.,
                            [ndhnlf["D"], ndhnlf["Dprime"]],
                            ndhnlf["gamma"] * 1e-3,
                            gain=dBkm_to_m(ndhnlf["Alpha"]),
                            dispersion_format="D")

fiber_pm1550 = fpn.Fiber()
fiber_pm1550.generate_fiber(.2,
                            1550.,
                            [pm1550["D"], pm1550["Dprime"]],
                            pm1550["gamma"] * 1e-3,
                            gain=dBkm_to_m(pm1550["Alpha"]),
                            dispersion_format="D")
