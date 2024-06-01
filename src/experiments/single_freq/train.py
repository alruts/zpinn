import logging
import os
import sys

# Block GPU from tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import equinox as eqx
import hydra
import jax.random as jrandom
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("src")
from experiments.single_freq.utils import setup_loaders, setup_optimizers
from zpinn.models.BVPEvaluator import BVPEvaluator
from zpinn.models.BVPModel import BVPModel
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.SIREN import SIREN


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
    else:
        raise ValueError(f"Invalid architecture: {config.architecture}")

    # data iterators
    dataloader, dom_sampler, bnd_sampler, ref_coords, ref_gt, transforms = (
        setup_loaders(config)
    )

    # bvp
    bvp = BVPModel(model, transforms, config)
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

    # ------------------- Training -------------------
    logging.info("Starting training, wait for JIT compilation...")
    for step in tqdm(range(config.training.steps)):
        # load batch
        batch = dict(
            dat_batch=next(iter(dataloader)),
            dom_batch=next(iter(dom_sampler)),
            bnd_batch=next(iter(bnd_sampler)),
        )
        
        # transition to boundary loss
        if (
            step == config.weighting.transition_step
            and config.weighting.transition_step is not None
        ):
            assert (
                config.weighting.scheme == "grad_norm"
            ), "Only grad_norm supported for transition to boundary loss."
            
            config.weighting.use_boundary_loss = True  # switch to boundary loss
            bvp = BVPModel(model, transforms, config, params, None, coeffs)  # reset bvp
            evaluator.bvp = bvp  # reset evaluator
            # reset weights
            weights = bvp.weights  
            new_w = bvp.compute_weights(params, coeffs, **batch)
            weights = bvp.update_weights(weights, new_w)


        # step
        if config.weighting.scheme == "grad_norm":
            params, coeffs, opt_states = bvp.update(
                params, weights, coeffs, opt_states, optimizers, batch
            )

            if step % config.weighting.update_every == 0:
                new_w = bvp.compute_weights(params, coeffs, **batch)
                weights = bvp.update_weights(weights, new_w)

        if config.weighting.scheme == "mle":
            params, weights, coeffs, opt_states = bvp.update(
                params, weights, coeffs, opt_states, optimizers, batch
            )

        # logging
        if step % config.logging.log_interval == 0:
            evaluator(params, coeffs, weights, batch, step, ref_coords, ref_gt)

    # -----------------------------------------------

    # Save model in hydra output directory
    model_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/model.eqx"
    )
    bvp = BVPModel(model, transforms, config, params, None, coeffs)
    eqx.tree_serialise_leaves(model_path, bvp)
    logging.info(f"Model saved to {model_path}")

    writer.close()
