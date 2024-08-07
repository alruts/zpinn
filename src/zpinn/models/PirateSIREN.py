import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Union

from ..modules.pirate_block import PirateBlock
from ..modules.sine_layer import SineLayer


class PIKernel(eqx.Module):
    """Physics-Informed initialization kernel."""

    kernel: jnp.ndarray

    def __init__(self, pi_init_weights):
        self.kernel = pi_init_weights

    def __call__(self, x):
        return jnp.dot(x, self.kernel)


class PirateSIREN(eqx.Module):
    """Pirate Net with sinusoidal activation functions.

    This model is based on the Pirate Net architecture proposed by [1], combined
    with the idea of using sinusoidal activation functions, as proposed by [2].
    The model consists of a series of PirateBlock modules with a sinusoidal activation
    function, with the addition of the u and v encoding layers. The u and v layers
    are used to modulate the output of each layer, which helps to mitigate the gradient
    pathologies observed in MLP PINNs.

    [1] S. Wang, B. Li, Y. Chen, and P. Perdikaris, “PirateNets: Physics-informed
    Deep Learning with Residual Adaptive Networks.” arXiv, Feb. 11, 2024. Accessed:
    Jun. 05, 2024. [Online]. Available: http://arxiv.org/abs/2402.00326

    [2] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """

    pirate_blocks: list
    last_layer: Union[eqx.nn.Linear, jnp.ndarray]
    u: SineLayer
    v: SineLayer
    in_features: int
    hidden_features: int
    hidden_layers: int
    out_features: int

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        key=jrandom.PRNGKey(0),
        pi_init_weights=None,
        **kwargs,
    ):
        # Initialize the model
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        last_key, *keys = jax.random.split(key, hidden_layers + 5)
        keys_iter = iter(keys)

        self.pirate_blocks = []

        # u and v encoding layers
        self.u = SineLayer(
            omega_0=first_omega_0,
            is_first=True,
            in_features=in_features,
            out_features=hidden_features,
            key=next(keys_iter),
        )
        self.v = SineLayer(
            omega_0=first_omega_0,
            is_first=True,
            in_features=in_features,
            out_features=hidden_features,
            key=next(keys_iter),
        )

        # first hidden layer
        self.pirate_blocks.append(
            PirateBlock(
                omega_0=hidden_omega_0,
                in_features=in_features,
                hidden_features=hidden_features,
                key=next(keys_iter),
                is_first=True,
            )
        )
        # remaining hidden layers
        for _ in range(hidden_layers) - 1:
            self.pirate_blocks.append(
                PirateBlock(
                    omega_0=hidden_omega_0,
                    in_features=in_features,
                    hidden_features=hidden_features,
                    key=next(keys_iter),
                    is_first=False,
                )
            )

        # Last layer
        if outermost_linear:
            if pi_init_weights is not None:
                self.last_layer = PIKernel(pi_init_weights)

            else:
                init_key, last_key = jax.random.split(last_key)
                self.last_layer = eqx.nn.Linear(
                    in_features, out_features, use_bias=True, key=last_key
                )

                # Initialize the weights
                lim = jnp.sqrt(6.0 / in_features) / hidden_omega_0
                new_weights = jrandom.uniform(
                    init_key, (out_features, in_features), minval=-lim, maxval=lim
                )
                self.last_layer = eqx.tree_at(
                    lambda layer: layer.weight, self.last_layer, new_weights
                )

        else:
            self.last_layer = SineLayer(
                omega_0=hidden_omega_0,
                is_first=False,
                in_features=hidden_features,
                out_features=out_features,
                key=last_key,
            )

    def __call__(self, *args):
        """Forward pass."""
        x = jnp.array([*args])  # Stack the input variables
        u = self.u(x)
        v = self.v(x)

        for layer in self.pirate_blocks:
            x = layer(x, u, v)

        x = self.last_layer(x)  # Last layer

        return x
