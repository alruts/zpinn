import logging
import os
import sys

import equinox as eqx
import hydra
import jax.random as jrandom
from jax.numpy import save, load
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle

sys.path.append("src")
from experiments.single_freq.utils import setup_loaders, setup_optimizers
from zpinn.models.BVPModel import BVPModel
from zpinn.models.BVPEvaluator import BVPEvaluator
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.SIREN import SIREN


def train_and_evaluate(config):
    time_stamp = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    head, time = os.path.split(time_stamp)
    _, date = os.path.split(head)
    writer_path = os.path.join(config.paths.log_dir, date, time)
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

    # initial params, weights and coeffs
    params = bvp.get_parameters()
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

    print("Starting training...")
    for step in tqdm(range(config.training.steps)):
        # load batch
        batch = dict(
            dat_batch=next(iter(dataloader)),
            dom_batch=next(iter(dom_sampler)),
            bnd_batch=next(iter(bnd_sampler)),
        )

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

    # Save model in hydra output directory
    model_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/model.eqx"
    )
    logging.info(f"Saving model to {model_path}")

    # Create instance of optimised model
    model = BVPModel(model, transforms, config, params, weights, coeffs)
    eqx.tree_serialise_leaves(model_path, model)
    
    writer.close()
