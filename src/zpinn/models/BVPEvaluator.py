import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("src")
from zpinn.constants import _c0, _rho0
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
        # Unpack the reference data
        pr_star = ref["real_pressure"]
        pi_star = ref["imag_pressure"]
        ur_star = ref["real_velocity"]
        ui_star = ref["imag_velocity"]

        # Compute the impedance
        z_star = (pr_star + 1j * pi_star) / (ur_star + 1j * ui_star)
        const = _c0 * _rho0 if self.bvp.is_normalized else 1.0
        zi_star = jnp.imag(z_star) / const
        zr_star = jnp.real(z_star) / const

        # make predictions
        x, y, z, f = self.bvp.unpack_coords(coords)
        pr_pred, pi_pred = self.bvp.p_pred_fn(params, *(x, y, z, f))
        ur_pred, ui_pred = self.bvp.u_pred_fn(params, *(x, y, z, f))
        zr_pred, zi_pred = self.bvp.z_pred_fn(params, *(x, y, z, f))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(
            jnp.abs(pr_star - pr_pred) / jnp.linalg.norm(pr_star), x, y, ax=ax
        )
        self.writer.add_figure("Errors/pr", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(
            jnp.abs(pi_star - pi_pred) / jnp.linalg.norm(pi_star), x, y, ax=ax
        )
        self.writer.add_figure("Errors/pi", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(
            jnp.abs(ur_star - ur_pred) / jnp.linalg.norm(ur_star), x, y, ax=ax
        )
        self.writer.add_figure("Errors/uzr", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(
            jnp.abs(ui_star - ui_pred) / jnp.linalg.norm(ui_star), x, y, ax=ax
        )
        self.writer.add_figure("Errors/uzi", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(
            jnp.abs(zr_star - zr_pred) / jnp.linalg.norm(zr_star), x, y, ax=ax
        )
        self.writer.add_figure("Errors/zr", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(
            jnp.abs(zi_star - zi_pred) / jnp.linalg.norm(zi_star), x, y, ax=ax
        )
        self.writer.add_figure("Errors/zi", plt.gcf(), step)
        plt.close()

    def log_preds(self, params, grid, step):

        x, y, z, f = self.bvp.unpack_coords(grid)

        pr_pred, pi_pred = self.bvp.p_pred_fn(params, *(x, y, z, f))
        ur_pred, ui_pred = self.bvp.u_pred_fn(params, *(x, y, z, f))
        zr_pred, zi_pred = self.bvp.z_pred_fn(params, *(x, y, z, f))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(pr_pred, x, y, ax=ax)
        self.writer.add_figure("Predictions/pr", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(pi_pred, x, y, ax=ax)
        self.writer.add_figure("Predictions/pi", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(ur_pred, x, y, ax=ax)
        self.writer.add_figure("Predictions/uzr", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(ui_pred, x, y, ax=ax)
        self.writer.add_figure("Predictions/uzi", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(zr_pred, x, y, ax=ax)
        self.writer.add_figure("Predictions/zr", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = scalar_field(zi_pred, x, y, ax=ax)
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
