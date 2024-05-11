import sys

import jax.numpy as jnp
import jax.random as jrandom
import pytest

sys.path.append("src")
from zpinn.modules.sine_layer import SineLayer

def test_sine_layer_forward_pass():
    omega_0 = 0.5
    is_first = True
    in_features = 10
    out_features = 5
    key = jrandom.PRNGKey(0)
    x = jnp.ones((in_features,))

    layer = SineLayer(omega_0, is_first, in_features, out_features, key=key)
    output = layer(x)

    assert output.shape == (out_features,)

def test_sine_layer_weight_initialization():
    omega_0 = 0.5
    is_first = True
    in_features = 10
    out_features = 5
    key = jrandom.PRNGKey(0)

    layer = SineLayer(omega_0, is_first, in_features, out_features, key=key)

    assert layer.weight.shape == (out_features, in_features)
    
if __name__ == "__main__":
    pytest.main([__file__])
