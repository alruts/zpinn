from math import pi, tanh
from ..zpinn.constants import _c0, _rho0


def miki(
    flow_resistivity: float, frequency: float, thickness: float, normalized=True
) -> complex:
    """Calculate the normal surface impedance of a material using the Miki model.
    Assumes planar incidence and a rigid backing.
    
    [1] Y. Miki, “Acoustical properties of porous materials. Modifications of
    Delany-Bazley models.,” J. Acoust. Soc. Jpn. (E), J Acoust Soc Jpn E, vol.
    11, no. 1, pp. 19–24, 1990, doi: 10.1250/ast.11.19.


    Args:
        flow_resistivity: float
            Flow resistivity of the material in Pa.s/m².
        frequency: float
            Frequency of the sound in Hz.
        thickness: float
            Thickness of the material in meters.
        normalized: bool
            Whether to return the normalized impedance or not.
    Returns:
        float: Normal surface impedance of the material.
    """
    # Calculate the angular frequency
    angular_frequency = 2 * pi * frequency

    # Calculate the specific impedance of the material
    R = 1 + 0.070 * (frequency / flow_resistivity) ** -0.632
    X = -0.107 * (frequency / flow_resistivity) ** -0.632
    impedance = R + 1j * X

    # Calculate propagation constants
    alpha = angular_frequency / _c0 * (0.160 * (frequency / flow_resistivity) ** -0.618)
    beta = (
        angular_frequency / _c0 * (1 + 0.109 * (frequency / flow_resistivity) ** -0.618)
    )
    gamma = alpha + 1j * beta

    # Calculate the normalized impedance
    impedance /= tanh(gamma * thickness)

    if not normalized:
        # unnormalize the impedance
        impedance *= _rho0 * _c0

    return impedance

