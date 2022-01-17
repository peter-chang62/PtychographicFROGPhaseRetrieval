import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

normalize = lambda vec: vec / np.max(abs(vec))


def get_data(path):
    return np.genfromtxt(path, delimiter=',', skip_header=29)


class Data:
    def __init__(self, path, data_is_log=True):
        data = get_data(path)
        if data_is_log:
            data[:, 1] = 10 ** (data[:, 1] / 10)
        self.x = data[:, 0]
        self.y = data[:, 1]

        self.bandwidth_wl = self.DB3Threshold()
        self.bandwidth = np.diff(self.bandwidth_wl)

    def plot(self, ax=None, color=None, title="", label=None, dB=True,
             norm=True):
        if (not dB) and (not norm):
            y = self.y
        else:
            y = np.copy(self.y)

        if norm:
            y = normalize(y)
        if dB:
            y = 10 * np.log10(y)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if label is not None:
            if color is not None:
                ax.plot(self.x, y, color=color, label=label)
            else:
                ax.plot(self.x, y, label=label)
        else:
            if color is not None:
                ax.plot(self.x, y, color=color)
            else:
                ax.plot(self.x, y)

        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("a.u.")
        ax.set_title(title)

    def DB3Threshold(self):
        level = 10 ** (-3 / 10)
        spl = scipy.interpolate.UnivariateSpline(self.x,
                                                 normalize(self.y) - level, s=0)
        roots = spl.roots()
        return roots[[0, -1]]
