# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..\..\src")
from zpinn.impedance_models import miki_model, R_plus_2, RMK_plus_1

f = np.linspace(50, 500, 100)

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

plt.subplots(1, 1, figsize=(10, 10))
plt.plot(f, zr, label="Real pred", color="red")
plt.plot(f, zi, label="Imaginary pred", color="blue", linestyle="--")
plt.plot(f, zr_miki, label="Real Miki", color="green")
plt.plot(f, zi_miki, label="Imaginary Miki", color="orange", linestyle="--")
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Impedance [Ohm]")
plt.grid()
plt.title("RMK+1 model")
