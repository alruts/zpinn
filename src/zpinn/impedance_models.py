import jax.numpy as jnp
from jax.numpy import pi, tanh
from .constants import _c0, _rho0


def constant_impedance(coeffs, f=None, normalized=True):
    """Return the impedance of the singular model."""
    # Calculate impedance
    z = coeffs["alpha"] + 1j * coeffs["beta"]

    if not normalized:
        z *= _rho0 * _c0

    return z.real, z.imag


def RMK_plus_1(coeffs, f, normalized=True):
    """Return the impedance of the RMK+1 model at a given frequency."""
    # Calculate the angular frequency
    omega = 2 * jnp.pi * f

    # Calculate the impedance
    z = coeffs["K"] * (1j * omega) ** -1
    z += coeffs["R_1"]
    z += coeffs["M"] * (1j * omega)
    z += coeffs["G"] * (1j * omega) ** coeffs["gamma"]

    if not normalized:
        z *= _rho0 * _c0

    return z.real, z.imag


def R_plus_2(coeffs, f, normalized=True):
    """Return the impedance of the R+2 model at a given frequency."""
    # Calculate the angular frequency
    omega = 2 * jnp.pi * f

    # Calculate the impedance
    z = coeffs["A"] * (1j * omega) ** -coeffs["alpha"]
    z += coeffs["R_2"]
    z += coeffs["B"] * (1j * omega) ** coeffs["beta"]

    if not normalized:
        z *= _rho0 * _c0

    return z.real, z.imag


def miki_model(
    flow_resistivity: float, frequency: float, thickness: float, normalized=True
) -> complex:
    """Calculate the normal surface impedance of a material using the Miki model.
    Assumes planar incidence and a rigid backing.

    [1] Y. Miki, "Acoustical properties of porous materials. Modifications of
    Delany-Bazley models.," J. Acoust. Soc. Jpn. (E), J Acoust Soc Jpn E, vol.
    11, no. 1, pp. 19-24, 1990, doi: 10.1250/ast.11.19.


    Args:
        flow_resistivity: float
            Flow resistivity of the material in Pa.s/m2.
        frequency: float
            Frequency of the sound in Hz.
        thickness: float
            Thickness of the material in meters.
        normalized: bool
            Whether to return the normalized impedance or not.
    Returns:
        float: specific impedance of the material.
    """
    # Calculate the angular frequency
    omega = 2 * pi * frequency

    # Calculate the specific impedance of the material
    R = 1 + 0.070 * (frequency / flow_resistivity) ** -0.632
    X = -0.107 * (frequency / flow_resistivity) ** -0.632
    Z = R + 1j * X

    # Calculate propagation constants
    alpha = omega / _c0 * (0.160 * (frequency / flow_resistivity) ** -0.618)
    beta = omega / _c0 * (1 + 0.109 * (frequency / flow_resistivity) ** -0.618)
    gamma = alpha + 1j * beta

    # Calculate the normalized impedance
    Z /= tanh(gamma * thickness)

    if not normalized:
        # unnormalize the impedance
        Z *= _rho0 * _c0

    return Z
