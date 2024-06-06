import os
os.environ["JAX_DISABLE_JIT"] = "1" # Disable JIT for faster testing

import sys
import warnings

import jax
import jax.numpy as jnp
import optax
import pytest
from omegaconf import OmegaConf

sys.path.append("src")
from zpinn.get_loaders import get_loaders
from zpinn.models.BVPModel import BVPModel
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.PirateSIREN import PirateSIREN
from zpinn.models.SIREN import SIREN
from zpinn.utils import flatten_pytree


# treat warnings as errors
# warnings.simplefilter("error")

# create dummy config
config = OmegaConf.create(
    {
        "paths": {
            "dataset": "./data/processed/inf_baffle.pkl",
            "initial_model": None,
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
            "use_boundary_loss": True,
            "scheme": "grad_norm",
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
                "restrict_to": {
                    "x": [-0.5, 0.5],
                    "y": [-0.5, 0.5],
                    "z": [-0.5, 0.5],
                    "f": [250],
                },
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
    PirateSIREN(**config.model, key=jax.random.PRNGKey(0)),
]

impedance_models = [
    "single_freq",
    "RMK+1",
    "R+2",
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
    "R+2": {
        "R_2": 0.0,
        "A": 0.0,
        "B": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
    },
}

# get dataloaders
data_loader, dom_loader, bnd_loader, _, _, transforms = get_loaders(config)
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

            p = bvp.psi_net(bvp.get_parameters(), *([0.0] * 4))

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

            r = bvp.r_net(bvp.get_parameters(), *([1.0] * 4))

            assert len(r) == 2
            assert type(r) == tuple
            assert r[0] is not jnp.nan
            assert r[1] is not jnp.nan


def test_fwd_pass_un_net():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )

            un = bvp.un_net(bvp.get_parameters(), *([1.0] * 4))

            assert len(un) == 2
            assert type(un) == tuple
            assert un[0] is not jnp.nan
            assert un[1] is not jnp.nan


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

            z = bvp.z_net(bvp.get_parameters(), *([1.0] * 4))

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

            l = bvp.p_loss(bvp.get_parameters(), next(iter(data_loader)))

            assert len(l) == 2
            assert type(l) == tuple
            assert l[0] is not jnp.nan
            assert l[1] is not jnp.nan
            assert l[0] > 0
            assert l[1] > 0
            if l[0] > 100:
                warnings.warn(f"Real loss is high: {l[0]}")
            if l[1] > 100:
                warnings.warn(f"Imaginary loss is high: {l[1]}")


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

            l = bvp.r_loss(bvp.get_parameters(), next(dom_iterator))

            assert len(l) == 2
            assert type(l) == tuple
            assert l[0] is not jnp.nan
            assert l[1] is not jnp.nan
            assert l[0] > 0
            assert l[1] > 0
            if l[0] > 100:
                warnings.warn(f"Real loss is high: {l[0]}")
            if l[1] > 100:
                warnings.warn(f"Imaginary loss is high: {l[1]}")


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

            l = bvp.z_loss(bvp.get_parameters(), bvp.coeffs, next(bnd_iterator))

            assert len(l) == 2
            assert type(l) == tuple
            assert l[0] is not jnp.nan
            assert l[1] is not jnp.nan
            assert l[0] > 0
            assert l[1] > 0
            if l[0] > 100:
                warnings.warn(f"Real loss is high: {l[0]}")
            if l[1] > 100:
                warnings.warn(f"Imaginary loss is high: {l[1]}")


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
                dat_batch=next(iter(data_loader)),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )

            loss = bvp.compute_loss(
                bvp.get_parameters(), bvp.weights, bvp.coeffs, **batches
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
                dat_batch=next(iter(data_loader)),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )
            params, coeffs = bvp.get_parameters(), bvp.coeffs
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
                dat_batch=next(iter(data_loader)),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )

            params, coeffs = bvp.get_parameters(), bvp.coeffs
            new_w = bvp.compute_weights(params, coeffs, **batch)
            old_w = bvp.weights
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
                bvp.get_parameters(),
                bvp.coeffs,
                next(iter(data_loader)),
                next(dom_iterator),
                next(bnd_iterator),
            )

            assert len(losses) == 6  # Number of losses
            assert all(l.dtype == jnp.float32 for l in losses.values())
            assert all(l > 0 for l in losses.values())


def test_compute_coeffs():
    for model in models:
        for impedance_model in impedance_models:
            config.impedance_model.type = impedance_model
            config.impedance_model.initial_guess = initial_guesses[impedance_model]
            bvp = BVPModel(
                model=model,
                transforms=transforms,
                config=config,
            )
            params, coeffs = bvp.get_parameters(), bvp.coeffs
            updates = bvp.compute_coeffs(params, coeffs, next(bnd_iterator))

            assert len(updates) == len(bvp.coeffs)  # Number of impedance coefficients
            assert all(c.dtype == jnp.float32 for c in updates.values())


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
                dat_batch=next(iter(data_loader)),
                dom_batch=next(dom_iterator),
                bnd_batch=next(bnd_iterator),
            )

            weights = bvp.weights
            params = bvp.get_parameters()
            coeffs = bvp.coeffs

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

            # TODO: change
            assert len(coeffs) == len(bvp.coeffs)
            assert type(coeffs) == dict
            assert all(c.dtype == jnp.float32 for c in coeffs.values())

            assert type(params) == model.__class__
            assert all(p.dtype == jnp.float32 for p in flatten_pytree(params))


if __name__ == "__main__":
    pytest.main([__file__])
