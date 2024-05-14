import os
import sys

import hydra
import jax
import jax.random as jrandom
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import optax

sys.path.append("src")
from zpinn.models.BVPModel import BVPModel, BVPEvaluator
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.SIREN import SIREN
from setup import setup_loaders, setup_optimizers


@hydra.main(config_path="./", config_name="config.yaml", version_base=hydra.__version__)
def main(config):
    time_stamp = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    date, time = time_stamp.split("\\")[-2], time_stamp.split("\\")[-1]
    writer_path = os.path.join(config.paths.log_dir, date, time)
    writer = SummaryWriter(writer_path)

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
    dataloader, dom_sampler, bnd_sampler, transforms = setup_loaders(
        config,
        restrict_to={"f": [250]},  ## HACK remove later
    )

    # bvp
    bvp = BVPModel(model, transforms, config)
    evaluator = BVPEvaluator(bvp, transforms, writer, config)

    # initial params, weights and coeffs
    params = bvp.parameters()
    weights = bvp.init_weights
    coeffs = bvp.init_coeffs

    # optimizers
    # lr, optimizers = setup_optimizers(config)
    
    optimizers = dict(
        params= optax.adam(learning_rate=1e-4),
        coeffs= optax.adam(learning_rate=1e-1),
    )
    
    opt_states = dict(
        params=optimizers["params"].init(bvp.parameters()),
        coeffs=optimizers["coeffs"].init(bvp.init_coeffs),
    )

    print("Starting training...")
    for step in tqdm(range(config.training.steps)):
        batch = dict(
            dat_batch=next(iter(dataloader)),
            dom_batch=next(iter(dom_sampler)),
            bnd_batch=next(iter(bnd_sampler)),
        )

        params, coeffs, opt_states = bvp.update(
            params, weights, coeffs, opt_states, optimizers, batch
        )

        if step % config.weighting.update_every == 0:
            new_w = bvp.compute_weights(params, coeffs, **batch)
            weights = bvp.update_weights(weights, new_w)

        if step % config.logging.log_interval == 0:
            ref_grid = dict(
                x=jax.numpy.linspace(-0.5, 0.5, 100),
                y=jax.numpy.linspace(-0.5, 0.5, 100),
                z=0.05,
                f=250.0,
            )
            evaluator(params, coeffs, new_w, batch, step, ref_grid, 1)


if __name__ == "__main__":
    main()
