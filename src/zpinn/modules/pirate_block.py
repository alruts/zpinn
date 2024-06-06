import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from .sine_layer import SineLayer


class PirateBlock(eqx.Module):
    """ Implements an adaptive residual block as proposed in [1] combined
    with the idea of using sinusoidal activation functions from [2]. The block
    consists of a series of SineLayer modules with the addition of the u and v
    encoding layers. The u and v layers are used to modulate the output of each
    layer, which helps to mitigate the gradient pathologies observed in MLP models.
    In addition, the block also includes an adaptive skip connection via the
    parameter alpha, which is initialized to zero and learned during training.
    
    Based on the PIModifiedBottleneck class from repository provided in [1].
    
    Args:
    - in_features: Number of input features.
    - hidden_features: Number of hidden features.
    - key: Random key. Default is jrandom.PRNGKey(0).
    - omega_0: Frequency of the sine activation function. Default is 30.0.
    - is_first: Whether the block is the first in the network. Default is False.

    [1] S. Wang, B. Li, Y. Chen, and P. Perdikaris, “PirateNets: Physics-informed 
    Deep Learning with Residual Adaptive Networks.” arXiv, Feb. 11, 2024. Accessed: 
    Jun. 05, 2024. [Online]. Available: http://arxiv.org/abs/2402.00326
    
    [2] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """
    layers: list
    alpha: float

    def __init__(
        self,
        in_features,
        hidden_features,
        key=jrandom.PRNGKey(0),
        omega_0=30.0,
        is_first=False,
        **kwargs,
    ):
        last_key, *keys = jax.random.split(key, 3)
        keys_iter = iter(keys)

        self.layers = []

        # First layer
        self.layers.append(
            SineLayer(
                omega_0=omega_0,
                is_first=is_first,
                in_features=in_features,
                out_features=hidden_features,
                key=next(keys_iter),
            )
        )

        # Second layer
        self.layers.append(
            SineLayer(
                omega_0=omega_0,
                is_first=False,
                in_features=hidden_features,
                out_features=hidden_features,
                key=next(keys_iter),
            )
        )

        # Last layer
        self.layers.append(
            SineLayer(
                omega_0=omega_0,
                is_first=False,
                in_features=hidden_features,
                out_features=in_features,
                key=last_key,
            )
        )

        self.alpha = jnp.array(0.0)

    def __call__(self, x, u, v):
        """Forward pass."""
        identity = x
        for layer in self.layers[:-1]:
            x = layer(x)
            x = x * u + (1 - x) * v

        x = self.layers[-1](x)  # Last layer
        x = self.alpha * x + (1 - self.alpha) * identity
        return x
