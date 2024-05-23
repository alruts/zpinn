import logging
import os
import sys

import equinox as eqx
import hydra
import jax.random as jrandom
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("src")
from experiments.single_freq.utils import setup_loaders, setup_optimizers
from zpinn.models.BVPModel import BVPModel
from zpinn.models.BVPEvaluator import BVPEvaluator
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.SIREN import SIREN


def train_and_evaluate(config):
    time_stamp = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    date, time = time_stamp.split("\\")[-2], time_stamp.split("\\")[-1]
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
    lr, optimizers = setup_optimizers(config)

    opt_states = dict(
        params=optimizers["params"].init(bvp.get_parameters()),
        coeffs=optimizers["coeffs"].init(bvp.coeffs),
    )

    print("Starting training...")
    for step in tqdm(range(config.training.steps)):
        # load batch
        batch = dict(
            dat_batch=next(iter(dataloader)),
            dom_batch=next(iter(dom_sampler)),
            bnd_batch=next(iter(bnd_sampler)),
        )

        # step
        params, coeffs, opt_states = bvp.update(
            params, weights, coeffs, opt_states, optimizers, batch
        )

        # gradient based weighting schemes
        if step % config.weighting.update_every == 0:
            new_w = bvp.compute_weights(params, coeffs, **batch)
            weights = bvp.update_weights(weights, new_w)
            
        # logging
        if step % config.logging.log_interval == 0:
            evaluator(params, coeffs, new_w, batch, step, ref_coords, ref_gt)

    # TODO: Test if this works
    model_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        + "/model_params.eqx"
    )
    logging.info(f"Saving model to {model_path}")
    eqx.tree_serialise_leaves(model_path, params)

    writer.close()
