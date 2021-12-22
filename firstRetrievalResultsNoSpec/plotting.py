import numpy as np
import matplotlib.pyplot as plt

normalize = lambda vec: vec / np.max(abs(vec))

data = np.genfromtxt("AT_T_ps_real_imag.txt")
T_ps = data[:, 0]
real = data[:, 1]
imag = data[:, 2]

AT = real + 1j * imag

plt.figure()
plt.plot(T_ps * 1e3, normalize(abs(AT) ** 2))
plt.xlim(-20,10)
