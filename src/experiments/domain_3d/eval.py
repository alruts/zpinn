import logging
import os
import sys

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import torch
from jax.tree_util import tree_map
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

sys.path.append("src")
from experiments.domain_3d.utils import setup_loaders, setup_optimizers
from zpinn.models.BVPEvaluator import BVPEvaluator
from zpinn.models.BVPModel import BVPModel
from zpinn.models.ModifiedSIREN import ModifiedSIREN
from zpinn.models.PirateSIREN import PirateSIREN
from zpinn.models.SIREN import SIREN
from zpinn.plot.fields import scalar_field
from zpinn.utils import transform


# parse log path
parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type=str, required=True)
args = parser.parse_args()
log_path = args.log_path


def evaluate(log_path=log_path):
    config = OmegaConf.load(os.path.join(log_path, ".hydra", "config.yaml"))

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

    # load the model
    bvp = eqx.tree_deserialise_leaves(
        os.path.join(log_path, "model.eqx"),
        BVPModel(model, config),
    )

    # initial params, weights and coeffs
    params = bvp.params
    weights = bvp.weights
    coeffs = bvp.coeffs

    # ground truth
    pr_gt, pi_gt = ref_gt["real_pressure"], ref_gt["imag_pressure"]
    ur_gt, ui_gt = ref_gt["real_velocity"], ref_gt["imag_velocity"]
    zr_gt, zi_gt = ref_gt["real_impedance"], ref_gt["imag_impedance"]

    # transform the data
    x, y, z, f = bvp.unpack_coords(ref_coords)
    x = transform(x, bvp.x0, bvp.xc)
    y = transform(y, bvp.y0, bvp.yc)
    z = transform(z, bvp.z0, bvp.zc)
    f = transform(f, bvp.f0, bvp.fc)

    # predictions
    pr_pred, pi_pred = bvp.p_pred_fn(params, *(x, y, z, f))
    ur_pred, ui_pred = bvp.un_pred_fn(params, *(x, y, z, f))
    zr_pred, zi_pred = bvp.z_pred_fn(params, *(x, y, z, f))

    # alternative methods
    z_pred_1 = ((pr_pred + 1j * pi_pred) / (ur_pred + 1j * ui_pred)).mean()
    z_pred_2 = (pr_pred.mean() + 1j * pi_pred.mean()) / (
        ur_pred.mean() + 1j * ui_pred.mean()
    )

    # ------------ plot histogram of impedance values ------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(zr_pred.flatten(), bins=100)
    ax[0].set_title("Impedance Re")
    ax[0].set_xlabel("Resistance")
    ax[0].set_ylabel("Frequency")
    ax[1].hist(zi_pred.flatten(), bins=100)
    ax[1].set_title("Impedance Im")
    ax[1].set_xlabel("Reactance")
    ax[1].set_ylabel("Frequency")

    # mark the ground truth
    ax[0].axvline(zr_gt, color="r", linestyle="--", label="Ground Truth")
    ax[1].axvline(zi_gt, color="r", linestyle="--", label="Ground Truth")

    # mark mean values
    ax[0].axvline(jnp.mean(zr_pred), color="g", linestyle="--", label="Mean")
    ax[1].axvline(jnp.mean(zi_pred), color="g", linestyle="--", label="Mean")

    # mark alternative methods
    ax[0].axvline(z_pred_1.real, color="b", linestyle="--", label="Method 1")
    ax[1].axvline(z_pred_1.imag, color="b", linestyle="--", label="Method 1")
    ax[0].axvline(z_pred_2.real, color="m", linestyle="--", label="Method 2")
    ax[1].axvline(z_pred_2.imag, color="m", linestyle="--", label="Method 2")

    # add legend
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("impedance_histogram.png")

    # ------------ plot diagonal fits ------------
    # get the diagonal of ur_pred
    pi_pred_diag = jnp.diag(pi_pred)
    pi_gt_diag = jnp.diag(pi_gt)
    pr_pred_diag = jnp.diag(pr_pred)
    pr_gt_diag = jnp.diag(pr_gt)
    
    # get the diagonal of ur_pred
    ui_pred_diag = jnp.diag(ui_pred)
    ui_gt_diag = jnp.diag(ui_gt)
    ur_pred_diag = jnp.diag(ur_pred)
    ur_gt_diag = jnp.diag(ur_gt)

    d = jnp.linspace(-jnp.sqrt(2), jnp.sqrt(2), len(pi_pred_diag))

    # plot the diagonal
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()
    
    ax[0].plot(d, pi_pred_diag, label="PINN prediction")
    ax[0].plot(d, pi_gt_diag, label="BEM Reference")
    ax[0].title.set_text("imaginary pressure")
    ax[0].legend()
    
    ax[1].plot(d, pr_pred_diag, label="PINN prediction")
    ax[1].plot(d, pr_gt_diag, label="BEM Reference")
    ax[1].title.set_text("real pressure")
    ax[1].legend()
    
    ax[2].plot(d, ui_pred_diag, label="PINN prediction")
    ax[2].plot(d, ui_gt_diag, label="BEM Reference")
    ax[2].title.set_text("imaginary velocity")
    ax[2].legend()
    
    ax[3].plot(d, ur_pred_diag, label="PINN prediction")
    ax[3].plot(d, ur_gt_diag, label="BEM Reference")
    ax[3].title.set_text("real velocity")
    ax[3].legend()

    plt.tight_layout()
    plt.savefig("diagonal_fits.png")


if __name__ == "__main__":
    evaluate()
