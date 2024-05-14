import sys

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf
import optax

sys.path.append("src")
from zpinn.models.BVPModel import BVPModel
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.SIREN import SIREN
from zpinn.get_loaders import get_loaders
from zpinn.utils import flatten_pytree


# create dummy config
config = OmegaConf.create(
    {
        "paths": {
            "dataset": "C:\\Users\\STNj\\dtu\\thesis\\zpinn\\data\\processed\\inf_baffle.pkl"
        },
        "model": {
            "in_features": 4,
            "out_features": 2,
            "hidden_features": 32,
            "hidden_layers": 3,
            "outermost_linear": True,
        },
        "training": {
            "criterion": "mse",
        },
        "impedance_model": {
            "normalized": True,
            "type": "single_freq",
            "initial_guess": dict(),
        },
        "weighting": {
            "momentum": 0.9,
            "initial_weights": {
                "data_re": 1.0,
                "data_im": 1.0,
                "pde_re": 1.0,
                "pde_im": 1.0,
                "bc_re": 1.0,
                "bc_im": 1.0,
            },
        },
        "batch": {
            "data": {
                "batch_size": 10,
                "shuffle": False,
            },
            "domain": {
                "batch_size": 10,
                "limits": {
                    "x": [-1, 1],
                    "y": [-1, 1],
                    "z": [-1, 1],
                    "f": [0, 1],
                },
                "distributions": {
                    "x": "uniform",
                    "y": "uniform",
                    "z": "uniform",
                    "f": "uniform",
                },
            },
            "boundary": {
                "batch_size": 10,
                "limits": {
                    "x": [-1, 1],
                    "y": [-1, 1],
                    "z": [-1, 1],
                    "f": [0, 1],
                },
                "distributions": {
                    "x": "uniform",
                    "y": "uniform",
                    "z": "uniform",
                    "f": "uniform",
                },
            },
        },
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

# initial guesses for impedance models
initial_guesses = {
    "single_freq": {
        "alpha": 0.0,
        "beta": 0.0,
    },
    "RMK+1": {
        "K": 0.0,
        "R_1": 0.0,
        "M": 0.0,
        "G": 0.0,
        "gamma": 0.0,
    },
}

# get dataloaders
data_loader, dom_loader, bnd_loader, transforms = get_loaders(config)
data_iterator = iter(data_loader)
dom_iterator = iter(dom_loader)
bnd_iterator = iter(bnd_loader)

def test_fwd_pass_psi_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            p = bvp.psi_net(bvp.parameters(), *([0.0] * 4))

            assert len(p) == 2
            assert type(p) == tuple
            assert p[0] is not jnp.nan
            assert p[1] is not jnp.nan


def test_fwd_pass_r_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
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


def test_fwd_pass_uz_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            uz = bvp.uz_net(bvp.parameters(), *([0.0] * 4))

            assert len(uz) == 2
            assert type(uz) == tuple
            assert uz[0] is not jnp.nan
            assert uz[1] is not jnp.nan


def test_fwd_pass_z_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
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
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
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
            assert l[0] > 0
            assert l[1] > 0
            assert l[0] < 100, "Real loss is too high"
            assert l[1] < 100, "Imaginary loss is too high"


def test_r_loss():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
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
            assert l[0] > 0
            assert l[1] > 0
            assert l[1] < 100, "Imaginary loss is too high"
            assert l[1] < 100, "Imaginary loss is too high"


def test_z_loss():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            l = bvp.z_loss(bvp.parameters(), bvp.init_coeffs, next(bnd_iterator))

            assert len(l) == 2
            assert type(l) == tuple
            assert l[0] is not jnp.nan
            assert l[1] is not jnp.nan
            assert l[0] > 0
            assert l[1] > 0
            assert l[0] < 100, "Real loss is too high"
            assert l[1] < 100, "Imaginary loss is too high"


def test_compute_loss():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
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
                bvp.parameters(), bvp.init_weights, bvp.init_coeffs, **batches
            )
            assert loss.dtype == jnp.float32
            assert loss > 0


def test_compute_weights():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )
            batch = dict(
                dat_batch=next(data_iterator),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )
            params, coeffs = bvp.parameters(), bvp.init_coeffs
            weights = bvp.compute_weights(params, coeffs, **batch)
            assert len(weights) == 6  # Number of losses
            assert all(w.dtype == jnp.float32 for w in weights.values())
            assert all(w > 0 for w in weights.values())


def test_update_weights():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )
            batch = dict(
                dat_batch=next(data_iterator),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )

            params, coeffs = bvp.parameters(), bvp.init_coeffs
            new_w = bvp.compute_weights(params, coeffs, **batch)
            old_w = bvp.init_weights
            updated_weights = bvp.update_weights(old_w, new_w)
            assert len(updated_weights) == 6  # Number of losses
            assert all(w.dtype == jnp.float32 for w in updated_weights.values())
            assert all(w > 0 for w in updated_weights.values())


def test_losses():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            losses = bvp.losses(
                bvp.parameters(),
                bvp.init_coeffs,
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
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )
            batch = dict(
                dat_batch=next(data_iterator),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )
            params, coeffs = bvp.parameters(), bvp.init_coeffs
            coeffs = bvp.grad_coeffs(params, coeffs, **batch)

            assert len(coeffs) == len(
                bvp.init_coeffs
            )  # Number of impedance coefficients
            assert all(c.dtype == jnp.float32 for c in coeffs.values())


def test_update():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )
            batch = dict(
                dat_batch=next(data_iterator),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )

            weights = bvp.init_weights
            params = bvp.parameters()
            coeffs = bvp.init_coeffs

            optimizers = dict(
                params=optax.adam(learning_rate=1e-4),
                coeffs=optax.adam(learning_rate=1e-1),
            )

            opt_states = dict(
                params=optimizers["params"].init(params),
                coeffs=optimizers["coeffs"].init(coeffs),
            )

            params, coeffs, opt_states = bvp.update(
                params, weights, coeffs, opt_states, optimizers, batch
            )

            assert len(params) == len(bvp.parameters())
            assert len(coeffs) == len(bvp.init_coeffs)

            assert type(coeffs) == dict
            assert all(c.dtype == jnp.float32 for c in coeffs.values())

            assert type(params) == list
            assert all(p.dtype == jnp.float32 for p in flatten_pytree(params))


if __name__ == "__main__":
    pytest.main([__file__])
