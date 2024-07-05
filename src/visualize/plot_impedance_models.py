# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..\..\src")
from zpinn.impedance_models import miki_model, R_plus_2, RMK_plus_1
import zpinn.plot as zplot
f = np.linspace(100, 2000, 1000)

# %% Miki model

z = miki_model(41e3,f, 0.05)
zr_miki = z.real
zi_miki = z.imag

# %% RMK+1 model

coeffs ={
    "G": 555,
    "K": 15e3,
    "M": 0.05,
    "R_1": 0.0,
    "gamma": -0.55,
}

zr, zi = RMK_plus_1(coeffs, f)

plt.subplots(1, 1)
# plt.plot(f, zr, label="Real pred", color="red")
# plt.plot(f, zi, label="Imaginary pred", color="blue", linestyle="--")
plt.semilogx(f, zr_miki, label="Real", color="red")
plt.semilogx(f, zi_miki, label="Imaginary", color="blue", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Normal Specific Impedance ()")
plt.grid()
plt.legend()
plt.xticks([100, 250, 500, 1000, 2000],["100", "250", "500", "1k", "2k"])
plt.minorticks_off()
# plt.tight_layout()

fig = plt.gcf()
fig.savefig("sample_impedance.pgf", bbox_inches="tight")
plt.show()