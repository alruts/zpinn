import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


sys.path.append("src")
from zpinn.constants import _c0, _rho0
from zpinn.plot.fields import scalar_field
from zpinn.utils import flatten_pytree, transform


class BVPEvaluator:
    def __init__(self, bvp, writer, config):
        self.config = config
        self.bvp = bvp
        self.writer = writer

    def log_losses(self, params, coeffs, batch, step):
        losses = self.bvp.losses(params, coeffs, **batch)

        for key, val in losses.items():
            self.writer.add_scalar("Loss/" + key, val.item(), step)

    def log_weights(self, weights, step):
        for key, val in weights.items():
            self.writer.add_scalar("Weights/" + key, val.item(), step)

    def log_coeffs(self, coeffs, step):
        for key, val in coeffs.items():
            self.writer.add_scalar("Coeffs/" + key, val.item(), step)

    def log_impedance(self, coeffs, ref_gt, step):
        zr, zi = self.bvp.impedance_model(coeffs, ref_gt["f"], True)
        zr_star, zi_star = ref_gt["real_impedance"] / (_rho0 * _c0), ref_gt[
            "imag_impedance"
        ] / (_rho0 * _c0)
        self.writer.add_scalar("Impedance/Real", zr.item(), step)
        self.writer.add_scalar("Impedance/Imag", zi.item(), step)
        self.writer.add_scalar("Impedance/RealStar", zr_star.item(), step)
        self.writer.add_scalar("Impedance/ImagStar", zi_star.item(), step)
        self.writer.add_scalar(
            "Impedance/RealL1Error", jnp.abs(zr - zr_star).item(), step
        )
        self.writer.add_scalar(
            "Impedance/ImagL1Error", jnp.abs(zi - zi_star).item(), step
        )

    def log_grads(self, params, coeffs, batch, step):
        grads = jax.jacrev(self.bvp.losses, argnums=0)(params, coeffs, **batch)

        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            self.writer.add_histogram("Grads/" + key, np.array(flattened_grad), step)

    def log_errors(self, params, coords, ref, step):
        # Compute the L2 errors
        errors = self.bvp.compute_l2_error(params, coords, ref)
        for key, val in errors.items():
            self.writer.add_scalar("PercentL2Errors/" + key, val.item(), step)

        # # Compute the L2 errors
        # errors_grid = self.bvp.compute_relative_error(params, coords, ref)
        # x, y, z, f = self.bvp.unpack_coords(coords)
        # pl_kwargs = dict(cbar_label="Relative Error")

        # # Log the relative errors as figures
        # for key, val in errors_grid.items():
        #     _, ax = plt.subplots(figsize=(5, 5))
        #     ax = scalar_field(val, x, y, ax=ax, **pl_kwargs)
        #     self.writer.add_figure("RelErrors/" + key, plt.gcf(), step)
        #     plt.close()

    def log_preds(self, params, grid, step):
        x, y, z, f = self.bvp.unpack_coords(grid)

        pr_pred, pi_pred = self.bvp.p_pred_fn(params, *(x, y, z, f))
        ur_pred, ui_pred = self.bvp.un_pred_fn(params, *(x, y, z, f))
        zr_pred, zi_pred = self.bvp.z_pred_fn(params, *(x, y, z, f))

        preds = dict(
            pr=pr_pred,
            pi=pi_pred,
            ur=ur_pred,
            ui=ui_pred,
            zr=zr_pred,
            zi=zi_pred,
        )

        for key, val in preds.items():
            _, ax = plt.subplots(figsize=(5, 5))
            ax = scalar_field(
                val, x, y, ax=ax, cbar_label=key, cmap="jet", balanced_cmap=False
            )
            self.writer.add_figure("Predictions/" + key, plt.gcf(), step)
            plt.close()

    def log_alphas(self, params, step):
        alphas = []
        for block in params.pirate_blocks:
            alphas.append(block.alpha)

        for idx, alpha in enumerate(alphas):
            self.writer.add_scalar(f"Alphas/{idx}", alpha.item(), step)

    def log_casual_weights(self, params, batch, step):
        lr, li, wr, wi = self.bvp.res_and_w(params, batch)
        _, ax = plt.subplots(figsize=(5, 5))
        ax.plot(wr, label="Real")
        ax.plot(wi, label="Imag")
        ax.legend()
        self.writer.add_figure("CasualWeights", plt.gcf(), step)
        plt.close()

    def __call__(
        self, params, coeffs, weights, batch, step, ref_coords, ref_gt, **kwargs
    ):
        ref_coords = dict(
            x=transform(ref_coords["x"], self.bvp.x0, self.bvp.xc),
            y=transform(ref_coords["y"], self.bvp.y0, self.bvp.yc),
            z=transform(ref_coords["z"], self.bvp.z0, self.bvp.zc),
            f=transform(ref_coords["f"], self.bvp.f0, self.bvp.fc),
        )

        if self.config.logging.log_losses:
            self.log_losses(params, coeffs, batch, step)

        if self.config.logging.log_weights:
            self.log_weights(weights, step)

        if self.config.logging.log_coeffs:
            self.log_coeffs(coeffs, step)

        if self.config.logging.log_impedance:
            self.log_impedance(coeffs, ref_gt, step)

        if self.config.logging.log_grads:
            self.log_grads(params, coeffs, batch, step)

        if self.config.logging.log_errors:
            self.log_errors(params, ref_coords, ref_gt, step)

        if self.config.logging.log_preds:
            self.log_preds(params, ref_coords, step)

        if self.config.architecture.name == "pirate_siren":
            self.log_alphas(params, step)

        if self.config.weighting.use_causal:
            self.log_casual_weights(params, batch["dom_batch"], step)

        return self.writer
