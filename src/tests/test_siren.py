import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.append("src")
from zpinn.models.SIREN import SIREN
from zpinn.modules.sine_layer import SineLayer


def test_siren_forward_pass():
    key = jax.random.PRNGKey(0)
    in_features = 10
    hidden_features = 20
    hidden_layers = 3
    out_features = 5

    model = SIREN(
        key=key,
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
    )

    output = model(*tuple([0.0] * in_features))

    assert output.shape == (out_features,)


def test_siren_initialization():
    key = jax.random.PRNGKey(0)
    in_features = 10
    hidden_features = 20
    hidden_layers = 3
    out_features = 5

    model = SIREN(
        key=key,
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
    )

    assert len(model.layers) == hidden_layers + 2

    assert isinstance(model.layers[0], SineLayer)

    assert model.layers[0].in_features == in_features
    assert model.layers[0].out_features == hidden_features

    assert isinstance(model.layers[-1], SineLayer)
    assert model.layers[-1].in_features == hidden_features
    assert model.layers[-1].out_features == out_features


def test_siren_output_range():
    key = jax.random.PRNGKey(0)
    in_features = 10
    hidden_features = 20
    hidden_layers = 3
    out_features = 5

    model = SIREN(
        key=key,
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
    )

    output = model(*tuple([1.0] * in_features))

    assert jnp.all(output >= -1.0) and jnp.all(output <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
