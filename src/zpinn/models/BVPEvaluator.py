import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("src")
from zpinn.utils import flatten_pytree, transform
from zpinn.plot.fields import scalar_field


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

    def log_grads(self, params, coeffs, batch, step):
        grads = jax.jacrev(self.bvp.losses, argnums=0)(params, coeffs, **batch)

        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            self.writer.add_histogram("Grads/" + key, np.array(flattened_grad), step)

    def log_errors(self, params, coords, ref, step):
        # Compute the L2 errors
        error_grids = self.bvp.compute_l2_error_grid(params, coords, ref)
        x, y, z, f = self.bvp.unpack_coords(coords)

        pl_kwargs = dict(
            cbar_label="Relative L2 Error",
            balanced_cmap=False,
            cmap="viridis",
        )
        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(error_grids["pr"], x, y, ax=ax, **pl_kwargs)
        self.writer.add_figure("Errors/pr", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(error_grids["pi"], x, y, ax=ax, **pl_kwargs)
        self.writer.add_figure("Errors/pi", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(error_grids["unr"], x, y, ax=ax, **pl_kwargs)
        self.writer.add_figure("Errors/unr", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(error_grids["uni"], x, y, ax=ax, **pl_kwargs)
        self.writer.add_figure("Errors/uni", plt.gcf(), step)
        plt.close()

        self.writer.add_figure("Errors/zi", plt.gcf(), step)
        plt.close()
        
        # Compute the L2 errors
        error_grids = self.bvp.compute_l2_error(params, coords, ref)
        for key, val in error_grids.items():
            self.writer.add_scalar("PercentErrors/" + key, val.item(), step)

    def log_preds(self, params, grid, step):

        x, y, z, f = self.bvp.unpack_coords(grid)

        pr_pred, pi_pred = self.bvp.p_pred_fn(params, *(x, y, z, f))
        ur_pred, ui_pred = self.bvp.un_pred_fn(params, *(x, y, z, f))
        zr_pred, zi_pred = self.bvp.z_pred_fn(params, *(x, y, z, f))

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(pr_pred, x, y, ax=ax)
        self.writer.add_figure("Predictions/pr", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(pi_pred, x, y, ax=ax)
        self.writer.add_figure("Predictions/pi", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(ur_pred, x, y, ax=ax, cbar_label="Particle Velocity (m/s)")
        self.writer.add_figure("Predictions/uzr", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(ui_pred, x, y, ax=ax, cbar_label="Particle Velocity (m/s)")
        self.writer.add_figure("Predictions/uzi", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(zr_pred, x, y, ax=ax, cbar_label="Impedance (Pa.s/m)")
        self.writer.add_figure("Predictions/zr", plt.gcf(), step)
        plt.close()

        _, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(zi_pred, x, y, ax=ax, cbar_label="Impedance (Pa.s/m)")
        self.writer.add_figure("Predictions/zi", plt.gcf(), step)
        plt.close()

    def __call__(self, params, coeffs, weights, batch, step, ref_coords, ref):
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

        if self.config.logging.log_grads:
            self.log_grads(params, coeffs, batch, step)

        if self.config.logging.log_errors:
            self.log_errors(params, ref_coords, ref, step)

        if self.config.logging.log_preds:
            self.log_preds(params, ref_coords, step)

        return self.writer
