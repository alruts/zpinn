import logging
import sys

import equinox as eqx
import hydra
import jax.random as jrandom
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import jax.numpy as jnp

sys.path.append("src")
from experiments.domain_3d.utils import setup_loaders, setup_optimizers
from zpinn.models.BVPEvaluator import BVPEvaluator
from zpinn.models.BVPModel import BVPModel
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.PirateSIREN import PirateSIREN
from zpinn.models.SIREN import SIREN


def transition_to_boundary_loss(
    config, model, transforms, params, coeffs, evaluator, batch
):
    assert (
        config.weighting.scheme == "grad_norm"
    ), "Only grad_norm supported for transition to boundary loss."

    # switch to boundary loss
    config.weighting.use_boundary_loss = True
    bvp = BVPModel(model, transforms, config, params, None, coeffs)  # reset bvp

    # update evaluator
    evaluator.bvp = bvp

    # recalculate weights
    weights = bvp.weights
    new_w = bvp.compute_weights(params, coeffs, **batch, argnums=0)
    weights = bvp.update_weights(weights, new_w)

    return bvp, weights


def train_and_evaluate(config):
    writer_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(writer_path)

    logging.info(f"Logging to {writer_path}")

    # Set random seed
    key = jrandom.PRNGKey(config.random.seed)
    key, subkey = jrandom.split(key)

    # Initialize model architecture
    if config.architecture.name == "siren":
        model = SIREN(**config.architecture, key=subkey)
    elif config.architecture.name == "modified_siren":
        model = ModifiedSIREN(**config.architecture, key=subkey)
    elif config.architecture.name == "pirate_siren":
        model = PirateSIREN(**config.architecture, key=subkey)
    else:
        raise ValueError(f"Invalid architecture: {config.architecture}")

    # data iterators
    dataloader, dom_sampler, bnd_sampler, ref_coords, ref_gt, transforms = (
        setup_loaders(config)
    )

    # bvp
    bvp = BVPModel(model, config, transforms)
    evaluator = BVPEvaluator(bvp, writer, config)

    # load initial model if provided
    if config.paths.initial_model is not None:
        bvp = eqx.tree_deserialise_leaves(config.paths.initial_model, bvp)
        logging.info(f"Loaded model from {config.paths.initial_model}")

    # initial params, weights and coeffs
    params = bvp.params
    weights = bvp.weights
    coeffs = bvp.coeffs

    # optimizers
    optimizers = setup_optimizers(config)

    opt_states = dict(
        params=optimizers["params"].init(bvp.get_parameters()),
        coeffs=optimizers["coeffs"].init(bvp.coeffs),
    )
    if config.weighting.scheme == "mle":
        opt_states["weights"] = optimizers["weights"].init(bvp.weights)

    # ------------------- Training loop -------------------
    logging.info("Starting training, wait for JIT compilation...")
    for step in tqdm(range(config.training.steps)):

        if (
            step == config.weighting.transition_step
            and config.weighting.transition_step is not None
        ):
            bvp, weights = transition_to_boundary_loss(
                config, model, transforms, params, coeffs, evaluator, batch
            )
            
        batch = dict(
            dat_batch=next(iter(dataloader)),
            dom_batch=next(iter(dom_sampler)),
            bnd_batch=next(iter(bnd_sampler)),
        )

        if config.weighting.scheme == "grad_norm":
            params, coeffs, opt_states = bvp.update(
                params, weights, coeffs, opt_states, optimizers, batch
            )

            if step % config.weighting.update_every == 0:
                new_w = bvp.compute_weights(params, coeffs, **batch, argnums=0)
                weights = bvp.update_weights(weights, new_w)

        if config.weighting.scheme == "mle":
            params, weights, coeffs, opt_states = bvp.update(
                params, weights, coeffs, opt_states, optimizers, batch
            )
            # TODO: add this to config
            weights["bc_im"] = jnp.maximum(weights["bc_im"], 1.0)
            weights["bc_re"] = jnp.maximum(weights["bc_re"], 1.0)

        # logging
        if step % config.logging.log_interval == 0:
            evaluator(params, coeffs, weights, batch, step, ref_coords, ref_gt)
    # -----------------------------------------------

    # Save model in hydra output directory
    model_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/model.eqx"
    )
    bvp = BVPModel(model, config, transforms, params, None, coeffs)
    eqx.tree_serialise_leaves(model_path, bvp)
    logging.info(f"Model saved to {model_path}")

    writer.close()
