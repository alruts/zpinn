import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from ..modules.sine_layer import SineLayer


class ModifiedSIREN(eqx.Module):
    """Modified MLP with sinusoidal activation functions.

    This model is based on the modified MLP proposed by [1], combined
    with the idea of using sinusoidal activation functions, as proposed by [2]. 
    The model consists of a series of SineLayer modules with a sinusoidal 
    activation function, with the addition of the u and v encoding
    layers. The u and v layers are used to modulate the output of each layer,
    which helps to mitigate the gradient pathologies observed in MLP models.
    
    [1] S. Wang, Y. Teng, and P. Perdikaris, "Understanding and mitigating
    gradient pathologies in physics-informed neural networks." arXiv, Jan.
    13, 2020. Accessed: Apr. 04, 2024. [Online].
    Available: http://arxiv.org/abs/2001.04536

    [2] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661

    """

    layers: list
    u: SineLayer
    v: SineLayer

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        key=jrandom.PRNGKey(0),
        **kwargs,
    ):
        last_key, *keys = jax.random.split(key, hidden_layers + 4)
        keys_iter = iter(keys)

        self.layers = []

        # First layer
        self.layers.append(
            SineLayer(
                omega_0=first_omega_0,
                is_first=True,
                in_features=in_features,
                out_features=hidden_features,
                key=next(keys_iter),
            )
        )

        # u and v layers
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

        # Hidden layers
        for _ in range(hidden_layers):
            self.layers.append(
                SineLayer(
                    omega_0=hidden_omega_0,
                    is_first=False,
                    in_features=hidden_features,
                    out_features=hidden_features,
                    key=next(keys_iter),
                )
            )

        # Last layer
        if outermost_linear:
            init_key, last_key = jax.random.split(last_key)
            last_layer = eqx.nn.Linear(
                hidden_features, out_features, use_bias=True, key=last_key
            )

            # Initialize the weights
            lim = jnp.sqrt(6.0 / hidden_features) / hidden_omega_0
            new_weights = jrandom.uniform(
                init_key, (out_features, hidden_features), minval=-lim, maxval=lim
            )
            last_layer = eqx.tree_at(
                lambda layer: layer.weight, last_layer, new_weights
            )

        else:
            last_layer = SineLayer(
                omega_0=hidden_omega_0,
                is_first=False,
                in_features=hidden_features,
                out_features=out_features,
                key=last_key,
            )
        self.layers.append(last_layer)

    def __call__(self, *args):
        """Forward pass."""
        num_ins = self.layers[0].in_features
        
        x = jnp.array([*args[:num_ins]])  # Stack the input variables
        u = self.u(x)
        v = self.v(x)

        for layer in self.layers[:-1]:
            x = layer(x)
            x = x * u + (1 - x) * v

        x = self.layers[-1](x)  # Last layer

        return x
