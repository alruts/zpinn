import sys

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

sys.path.append("src")
from zpinn.models.BVPModel import BVPModel
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.SIREN import SIREN
from tests.get_dataloaders import get_dataloaders


# create dummy config
config = OmegaConf.create(
    {
        "model": {
            "in_features": 4,
            "out_features": 2,
            "hidden_features": 32,
            "hidden_layers": 3,
            "outermost_linear": True,
        },
        "impedance_model": "single_freq",
        "criterion": "mse",
        "momentum": 0.9,
    }
)


models = [
    SIREN(**config.model, key=jax.random.PRNGKey(0)),
    ModifiedSIREN(**config.model, key=jax.random.PRNGKey(0)),
]

impedance_models = [
    "single_freq",
    "RMK+1",
]

# get dataloaders...
data_iterator, dom_iterator, bnd_iterator, transforms = get_dataloaders()


def test_fwd_pass_p_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            p = bvp.p_net(bvp.parameters(), *([0.0] * 4))

            assert len(p) == 2
            assert type(p) == tuple
            assert p[0] is not jnp.nan
            assert p[1] is not jnp.nan


def test_fwd_pass_r_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            r = bvp.r_net(bvp.parameters(), *([0.0] * 4))

            assert len(r) == 2
            assert type(r) == tuple
            assert r[0] is not jnp.nan
            assert r[1] is not jnp.nan


def test_fwd_pass_z_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            z = bvp.z_net(bvp.parameters(), *([0.0] * 4))

            assert len(z) == 2
            assert type(z) == tuple
            assert z[0] is not jnp.nan
            assert z[1] is not jnp.nan


def test_p_loss():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            l = bvp.p_loss(bvp.parameters(), next(data_iterator))

            assert len(l) == 2
            assert type(l) == tuple
            assert l[0] is not jnp.nan
            assert l[1] is not jnp.nan


def test_r_loss():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            l = bvp.r_loss(bvp.parameters(), next(dom_iterator))

            assert len(l) == 2
            assert type(l) == tuple
            assert l[0] is not jnp.nan
            assert l[1] is not jnp.nan


def test_z_loss():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            l = bvp.z_loss(bvp.parameters(), bvp.coefficients, next(bnd_iterator))

            assert len(l) == 2
            assert type(l) == tuple
            assert l[0] is not jnp.nan
            assert l[1] is not jnp.nan


def test_compute_loss():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            batches = dict(
                dat_batch=next(data_iterator),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )

            loss = bvp.compute_loss(
                bvp.parameters(), bvp.weights, bvp.coefficients, batches
            )
            assert loss.dtype == jnp.float32
            assert loss > 0


def test_compute_weights():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )
            dat_batch = next(data_iterator)
            dom_batch = next(dom_iterator)
            bnd_batch = next(bnd_iterator)

            weights = bvp.compute_weights(dat_batch, dom_batch, bnd_batch)
            assert len(weights) == 6  # Number of losses
            assert all(w.dtype == jnp.float32 for w in weights.values())
            assert all(w > 0 for w in weights.values())


def test_losses():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            losses = bvp.losses(
                bvp.parameters(),
                bvp.coefficients,
                next(data_iterator),
                next(dom_iterator),
                next(bnd_iterator),
            )

            assert len(losses) == 6  # Number of losses
            assert all(l.dtype == jnp.float32 for l in losses.values())
            assert all(l > 0 for l in losses.values())


def test_grad_coeffs():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model = impedance_model
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )
            coeffs = bvp.grad_coeffs(
                next(data_iterator), next(dom_iterator), next(bnd_iterator)
            )
            assert len(coeffs) == len(
                bvp.coefficients
            )  # Number of impedance coefficients
            assert all(c.dtype == jnp.float32 for c in coeffs.values())


if __name__ == "__main__":
    pytest.main([__file__])
