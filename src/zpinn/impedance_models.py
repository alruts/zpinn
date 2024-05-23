import jax.numpy as jnp
from .constants import _c0, _rho0


def constant_impedance(coeffs, f, normalized=True):
    """Return the impedance of the singular model."""
    z = coeffs["alpha"] + 1j * coeffs["beta"]

    if normalized:
        z /= _rho0 * _c0

    return z.real, z.imag


def RMK_plus_1(coeffs, f, normalized=True):
    """Return the impedance of the RMK+1 model at a given frequency."""
    omega = 2 * jnp.pi * f
    z = coeffs["K"] * (1j * omega) ** -1
    z += coeffs["R_1"]
    z += coeffs["M"] * (1j * omega)
    z += coeffs["G"] * (1j * omega) ** coeffs["gamma"]

    if normalized:
        z /= _rho0 * _c0

    return z.real, z.imag
