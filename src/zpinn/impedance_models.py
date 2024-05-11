import jax.numpy as jnp


def constant_impedance(coeffs, f):
    """Return the impedance of the singular model."""
    return coeffs["alpha"] + 1j * coeffs["beta"]


def RMK_plus_1(coeffs, f):
    """Return the impedance of the RMK+1 model at a given frequency."""
    w = 2 * jnp.pi * f
    z = coeffs["K"] * (1j * w) ** -1
    z += coeffs["R_1"]
    Z += coeffs["M"] * (1j * w)
    Z += coeffs["G"] * (1j * w) ** coeffs["gamma"]
    return z
