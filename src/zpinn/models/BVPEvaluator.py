import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("src")
from zpinn.utils import flatten_pytree, transform
from zpinn.plot.fields import scalar_field
from zpinn.constants import _c0, _rho0


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
        zr, zi = self.bvp.impedance_model(coeffs, ref_gt["f"])
        zr_star, zi_star = ref_gt["real_impedance"] / (_rho0 * _c0), ref_gt[
            "imag_impedance"
        ] / (_rho0 * _c0)
        self.writer.add_scalar("Impedance/Real", zr.item(), step)
        self.writer.add_scalar("Impedance/Imag", zi.item(), step)
        self.writer.add_scalar("Impedance/RealStar", zr_star.item(), step)
        self.writer.add_scalar("Impedance/ImagStar", zi_star.item(), step)
        self.writer.add_scalar(
            "Impedance/RealError", jnp.abs(zr - zr_star).item(), step
        )
        self.writer.add_scalar(
            "Impedance/ImagError", jnp.abs(zi - zi_star).item(), step
        )

    def log_grads(self, params, coeffs, batch, step):
        grads = jax.jacrev(self.bvp.losses, argnums=0)(params, coeffs, **batch)

        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            self.writer.add_histogram("Grads/" + key, np.array(flattened_grad), step)

    def log_errors(self, params, coords, ref, step):
        # Compute the L2 errors
        errors_grid = self.bvp.compute_l2_error_grid(params, coords, ref)
        x, y, z, f = self.bvp.unpack_coords(coords)
        pl_kwargs = dict(
            cbar_label="Relative L2 Error",
            balanced_cmap=False,
            cmap="viridis",
        )

        # Log the L2 errors as figures
        for key, val in errors_grid.items():
            _, ax = plt.subplots(figsize=(5, 5))
            ax = scalar_field(val, x, y, ax=ax, **pl_kwargs)
            self.writer.add_figure("L2Errors/" + key, plt.gcf(), step)
            plt.close()

        # Compute the total  L2 errors as scalars
        errors = self.bvp.compute_l2_error(params, coords, ref)
        for key, val in errors.items():
            self.writer.add_scalar("PercentErrors/" + key, val.item(), step)

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
            ax = scalar_field(val, x, y, ax=ax)
            self.writer.add_figure("Predictions/" + key, plt.gcf(), step)
            plt.close()

    def __call__(self, params, coeffs, weights, batch, step, ref_coords, ref_gt):
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

        return self.writer
