import jax.numpy as jnp


def constant_impedance(coeffs, f):
    """Return the impedance of the singular model."""
    return coeffs["alpha"], coeffs["beta"]


def RMK_plus_1(coeffs, f):
    """Return the impedance of the RMK+1 model at a given frequency."""
    omega = 2 * jnp.pi * f
    z = coeffs["K"] * (1j * omega) ** -1
    z += coeffs["R_1"]
    z += coeffs["M"] * (1j * omega)
    z += coeffs["G"] * (1j * omega) ** coeffs["gamma"]
    return z.real, z.imag
