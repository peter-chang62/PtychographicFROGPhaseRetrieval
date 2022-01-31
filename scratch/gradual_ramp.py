import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 5, 1000)
y = x ** 2

ramp = np.zeros((100, len(x)))
diff = y - x

EPS = np.linspace(0, 1, len(ramp))
for n, eps in enumerate(EPS):
    ramp[n][:] = x + diff * eps

# %% good!
fig, ax = plt.subplots(1, 1)
for i in ramp:
    ax.clear()
    ax.plot(i)
    plt.pause(.01)
