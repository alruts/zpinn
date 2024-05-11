import logging
import os
import sys

import equinox as eqx
import hydra
import jax.random as jrandom
from torch.utils.tensorboard import SummaryWriter

sys.path.append("src")
from train_fn import train_fn
from setup import data_handlers, optimizers

from zpinn.utils import visualize
from zpinn.callbacks import log_weight_histograms_to_tensorboard
from zpinn.constants import _c0, _rho0
from zpinn.models import SIREN


@hydra.main(
    config_path="../../conf", config_name="inf_baffle", version_base=hydra.__version__
)
def main(config):

    # init tensorboard writer
    time_stamp = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    date, time = time_stamp.split("\\")[-2], time_stamp.split("\\")[-1]
    writer_path = os.path.join(config.tensorboard.log_dir, date, time)
    writer = SummaryWriter(writer_path)

    # Set random seed
    key = jrandom.PRNGKey(config.random.seed)
    key, subkey = jrandom.split(key)

    lambdas = dict(
        data=config.train.lambda_.data,
        pde=config.train.lambda_.pde,
        bc=config.train.lambda_.bc,
    )  # initial values

    coeffs = [
        config.train.impedance.real / (_rho0 * _c0),
        config.train.impedance.imag / (_rho0 * _c0),
    ]

    # Initialize model
    model = SIREN(
        key=subkey,
        in_features=4,
        out_features=2,
        hidden_features=config.model.hidden_features,
        hidden_layers=config.model.hidden_layers,
        first_omega_0=config.model.first_omega_0,
        hidden_omega_0=config.model.hidden_omega_0,
        outermost_linear=config.model.outermost_linear,
    )

    # Log weight histograms to tensorboard
    log_weight_histograms_to_tensorboard(model, writer, 0)

    (  # Get generators
        dataset,
        dom_gen,
        bnd_gen,
        eval_gen,
    ) = data_handlers(config, restrict_to=[250])

    (  # Get optimizers
        scheduler,
        optimizer_model,
        optimizer_lamdas,
        optimizer_coeffs,
    ) = optimizers(config)

    # Get dataloader
    dataloader = dataset.get_dataloader(
        batch_size=config.train.batch_size, shuffle=config.train.shuffle
    )

    # Visualize the model as a graph
    if config.model.plot_graph:
        print(type(model))
        visualize(model, (0.0, 0.0, 0.0, 0.0))

    # log scheduler to tensorboard
    for i in range(config.train.steps):
        writer.add_scalar("Learning Rate", scheduler(i).item(), i)

    # Train model
    model = train_fn(
        model=model,
        steps=config.train.steps,
        loss_type=config.train.loss_type,
        print_every=config.train.print_every,
        evaluate_every=config.train.evaluate_every,
        coeffs=coeffs,
        lambdas=lambdas,
        trainloader=dataloader,
        optim_model=optimizer_model,
        optim_eps=optimizer_lamdas,
        optim_Zn=optimizer_coeffs,
        transforms=dataset.transforms,
        dom_gen=dom_gen,
        bnd_gen=bnd_gen,
        eval_gen=eval_gen,
        writer=writer,
    )

    # Save model
    model_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/model.eqx"
    )
    logging.info(f"Saving model to {model_path}")
    eqx.tree_serialise_leaves(model_path, model)


if __name__ == "__main__":
    main()
