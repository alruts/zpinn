import sys
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import grad, lax, vmap
from jax.tree_util import tree_leaves, tree_map, tree_reduce
from matplotlib import pyplot as plt

sys.path.append("src")
from zpinn.constants import _c0, _rho0
from zpinn.impedance_models import RMK_plus_1, constant_impedance
from zpinn.utils import flatten_pytree, transform

criteria = {
    "mse": lambda x, y: jnp.mean((x - y) ** 2),
    "mae": lambda x, y: jnp.mean(jnp.abs(x - y)),
}


# TODO: Add epsilon method,
# TODO: add exact imposing method,
# TODO: (EXTRA) Add extra class to handle optimizers.
# TODO: (EXTRA) Add "custom scalars" to tensorboard logger.


class BVPModel:
    """Base class for the boundary value problem models."""

    model: eqx.Module
    criterion: Callable
    momentum: float
    impedance_model: Callable
    is_normalized: bool
    init_coeffs: dict
    init_weights: dict
    x0: float
    xc: float
    y0: float
    yc: float
    z0: float
    zc: float
    f0: float
    fc: float
    a0: float
    ac: float
    b0: float
    bc: float

    def __init__(self, model, transforms, config):
        self.model = model
        self.momentum = config.weighting.momentum
        self.init_weights = dict(config.weighting.initial_weights)
        self.init_coeffs = dict(config.impedance_model.initial_guess)

        self.impedance_model = self._init_z_model(config)
        (
            self.x0,
            self.xc,
            self.y0,
            self.yc,
            self.z0,
            self.zc,
            self.f0,
            self.fc,
            self.a0,
            self.ac,
            self.b0,
            self.bc,
        ) = self._init_transforms(transforms)

        self.is_normalized = config.impedance_model.normalized

        # Initialize the loss criterion
        self.criterion = criteria[config.training.criterion]

        # predict over grid
        self.p_pred_fn = vmap(
            vmap(self.p_net, (None, None, 0, None, None)), (None, 0, None, None, None)
        )
        self.uz_pred_fn = vmap(
            vmap(self.uz_net, (None, None, 0, None, None)), (None, 0, None, None, None)
        )
        self.z_pred_fn = vmap(
            vmap(self.z_net, (None, None, 0, None, None)), (None, 0, None, None, None)
        )

    def _init_transforms(self, tfs):
        """Unpack the transformation parameters."""
        x0, xc = tfs["x0"], tfs["xc"]
        y0, yc = tfs["y0"], tfs["yc"]
        z0, zc = tfs["z0"], tfs["zc"]
        f0, fc = tfs["f0"], tfs["fc"]
        a0, ac = tfs["a0"], tfs["ac"]
        b0, bc = tfs["b0"], tfs["bc"]
        return x0, xc, y0, yc, z0, zc, f0, fc, a0, ac, b0, bc

    def _init_z_model(self, config):
        # Initialize the coefficients based on the impedance model
        if config.impedance_model.type == "single_freq":
            z_model = constant_impedance

        elif config.impedance_model.type == "RMK+1":
            z_model = RMK_plus_1
        else:
            raise NotImplementedError(
                "Impedance model not implemented. Choose from ['single_freq', 'RMK+1']"
            )

        return z_model

    def apply_model(self, params, *args):
        """Trick to enable gradient with respect to weights."""
        get_params = lambda m: m.params()
        model = eqx.tree_at(get_params, self.model, params)
        return model(*args)

    def parameters(self):
        """Returns the parameters of the model."""
        is_eqx_linear = lambda x: isinstance(x, eqx.nn.Linear)
        params = [
            x.weight
            for x in jax.tree_util.tree_leaves(self.model, is_leaf=is_eqx_linear)
            if is_eqx_linear(x)
        ]
        return params

    def psi_net(self, params, *args, part=None):
        """Nondimensionalized pressure network."""
        p = self.apply_model(params, *args)

        if part == "real":
            return p[0]
        elif part == "imag":
            return p[1]
        else:
            return p[0], p[1]

    def p_net(self, params, *args):
        """Pressure network."""
        x, y, z, f = args
        pr, pi = self.psi_net(params, *args)
        pr = pr * self.ac + self.a0
        pi = pi * self.bc + self.b0
        return pr, pi

    def r_net(self, params, *args):
        """PDE residual network."""
        x, y, z, f = args
        k = (2 * jnp.pi * (f * self.fc + self.f0)) / (_c0)
        pr_pred, pi_pred = self.psi_net(params, *args)

        # compute real part
        p_xx = grad(grad(self.psi_net, argnums=1), argnums=1)(
            params, *args, part="real"
        )
        p_xx *= self.yc**2 * self.zc**2
        p_yy = grad(grad(self.psi_net, argnums=2), argnums=2)(
            params, *args, part="real"
        )
        p_yy *= self.xc**2 * self.zc**2
        p_zz = grad(grad(self.psi_net, argnums=3), argnums=3)(
            params, *args, part="real"
        )
        p_zz *= self.xc**2 * self.yc**2
        Lr = (p_xx + p_yy + p_zz) * self.ac
        rr = Lr + (k * self.xc * self.yc * self.zc) ** 2 * (pr_pred * self.ac + self.a0)

        # compute imaginary part
        p_xx = grad(grad(self.psi_net, argnums=1), argnums=1)(
            params, *args, part="imag"
        )
        p_xx *= self.yc**2 * self.zc**2
        p_yy = grad(grad(self.psi_net, argnums=2), argnums=2)(
            params, *args, part="imag"
        )
        p_yy *= self.xc**2 * self.zc**2
        p_zz = grad(grad(self.psi_net, argnums=3), argnums=3)(
            params, *args, part="imag"
        )
        p_zz *= self.xc**2 * self.yc**2
        Li = (p_xx + p_yy + p_zz) * self.bc
        ri = Li + (k * self.xc * self.yc * self.zc) ** 2 * (pi_pred * self.bc + self.b0)

        return rr, ri

    def uz_net(self, params, *args):
        """Normal particle velocity network."""
        x, y, z, f = args

        # compute the gradient of the pressure w.r.t. z
        p_zr = grad(self.psi_net, argnums=3)(params, *args, part="real")
        p_zi = grad(self.psi_net, argnums=3)(params, *args, part="imag")

        # apply euler's equation
        uz = 1 / (1j * 2 * jnp.pi * (f * self.fc + self.f0) * _rho0)
        uz /= self.zc
        uz *= self.ac * p_zr + 1j * self.bc * p_zi

        return uz.real, uz.imag

    def z_net(self, params, *args):
        """Boundary condition network."""
        # compute the particle velocity
        uzr, uzi = self.uz_net(params, *args)
        u_cplx = uzr + 1j * uzi

        # compute the pressure
        pr, pi = self.psi_net(params, *args)
        p_cplx = (pr * self.ac + self.a0) + 1j * (pi * self.bc + self.b0)

        z = p_cplx / u_cplx

        if self.is_normalized:
            z /= _rho0 * _c0
        return z.real, z.imag

    @eqx.filter_jit
    def compute_weights(self, params, coeffs, dat_batch, dom_batch, bnd_batch):
        """Computes the lambda weights for each loss."""

        # Compute the gradient of each loss w.r.t. the parameters
        grads = jax.jacrev(self.losses, argnums=0)(
            params, coeffs, dat_batch, dom_batch, bnd_batch
        )

        # Compute the grad norm of each loss
        grad_norm_dict = {}
        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

        # Compute the mean of grad norms over all losses
        mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))

        # Grad Norm Weighting
        w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)

        return w

    @eqx.filter_jit
    def update_weights(self, old_w, new_w, **kwargs):
        """Updates `weights` using running average with momentum."""
        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, old_w, new_w)
        weights = lax.stop_gradient(weights)

        return weights

    @eqx.filter_jit
    def grad_coeffs(self, params, coeffs, dat_batch, dom_batch, bnd_batch):
        """Computes the gradient of the loss w.r.t. the coefficients."""
        grads = jax.jacrev(self.losses, argnums=1)(
            params, coeffs, dat_batch, dom_batch, bnd_batch
        )

        coeff_keys = self.init_coeffs.keys()
        subdicts = [grads[key] for key in grads.keys()]

        sum_grad_dict = {}
        for key in coeff_keys:
            sum_grad_dict[key] = jnp.sum(jnp.stack([d[key] for d in subdicts]))

        return sum_grad_dict

    @eqx.filter_jit
    def losses(self, params, coeffs, dat_batch, dom_batch, bnd_batch):
        """Returns the losses of the model."""
        data_loss_re, data_loss_im = self.p_loss(params, dat_batch)
        pde_loss_re, pde_loss_im = self.r_loss(params, dom_batch)
        bc_loss_re, bc_loss_im = self.z_loss(params, coeffs, bnd_batch)
        return {
            "data_re": data_loss_re,
            "data_im": data_loss_im,
            "pde_re": pde_loss_re,
            "pde_im": pde_loss_im,
            "bc_re": bc_loss_re,
            "bc_im": bc_loss_im,
        }

    def p_loss(self, params, batch):
        """Data loss."""
        coords, gt = batch  # unpack the data batch
        f, x, y, z = coords.values()
        pr_pred, pi_pred = vmap(self.psi_net, in_axes=(None, *[0] * 4))(
            params, *(x, y, z, f)
        )
        pr_target, pi_target = gt["real_pressure"], gt["imag_pressure"]

        return self.criterion(pr_pred, pr_target), self.criterion(pi_pred, pi_target)

    def r_loss(self, params, batch):
        """PDE residual loss."""
        coords = batch
        f, x, y, z = coords.values()
        rr, ri = vmap(self.r_net, in_axes=(None, *[0] * 4))(params, *(x, y, z, f))
        return self.criterion(rr, 0.0), self.criterion(ri, 0.0)

    def z_loss(self, params, coeffs, batch):
        """Boundary loss."""
        coords = batch
        f, x, y, z = coords.values()
        zpr, zpi = vmap(self.z_net, in_axes=(None, *[0] * 4))(params, *(x, y, z, f))
        zmr, zmi = self.impedance_model(
            coeffs, f * self.fc + self.f0, normalized=self.is_normalized
        )

        return self.criterion(zpr, zmr), self.criterion(zpi, zmi)

    @eqx.filter_jit
    def compute_loss(self, params, weights, coeffs, dat_batch, dom_batch, bnd_batch):
        # Compute losses
        losses = self.losses(params, coeffs, dat_batch, dom_batch, bnd_batch)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    @eqx.filter_jit
    def update(self, params, weights, coeffs, opt_states, optimizers, batch):

        # --- Update the parameters ---
        grads = jax.grad(self.compute_loss)(params, weights, coeffs, **batch)
        updates, opt_states["params"] = optimizers["params"].update(
            grads, opt_states["params"]
        )
        params = eqx.apply_updates(params, updates)

        # --- Update the coefficients ---
        grads = self.grad_coeffs(params, coeffs, **batch)
        updates, opt_states["coeffs"] = optimizers["coeffs"].update(
            grads, opt_states["coeffs"]
        )
        coeffs = eqx.apply_updates(self.init_coeffs, updates)

        # Clip the coefficients if using RMK+1 model
        if self.impedance_model == RMK_plus_1:
            coeffs = {
                "K": jnp.clip(coeffs["K"], 0.0, 1.0),
                "R_1": jnp.clip(coeffs["R_1"], 0.0, 1.0),
                "M": jnp.clip(coeffs["M"], 0.0, 1.0),
                "G": jnp.clip(coeffs["G"], 0.0, 1.0),
                "gamma": jnp.clip(coeffs["gamma"], -1.0, 1.0),
            }

        return params, coeffs, opt_states


class BVPEvaluator:
    def __init__(self, bvp, transforms, writer, config):
        self.config = config
        self.bvp = bvp
        self.transforms = transforms
        self.writer = writer

        # Initialize the layout
        # self._init_layout()

    def log_losses(self, params, coeffs, batch, step):
        losses = self.bvp.losses(params, coeffs, **batch)

        for key, val in losses.items():
            self.writer.add_scalar("Loss/" + key, val.item(), step)

        # log the total loss
        self.writer.add_scalar("Loss/Total", sum(losses.values()).item(), step)

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
            grad_norm = jnp.linalg.norm(flattened_grad)
            self.writer.add_histogram(key, grad_norm.item(), step)

    # def log_errors(self, params, ref, step):
    #     p_pred = self.bvp.p_pred_fn(params, *(grid.values()))
    #     uz_pred = self.bvp.uz_pred_fn(params, *(grid.values()))
    #     z_pred = self.bvp.z_pred_fn(params, *(grid.values()))

    def log_preds(self, params, grid, step):
        pr_pred, pi_pred = self.bvp.p_pred_fn(params, *(grid.values()))
        uzr_pred, uzi_pred = self.bvp.uz_pred_fn(params, *(grid.values()))
        zr_pred, zi_pred = self.bvp.z_pred_fn(params, *(grid.values()))

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(pr_pred.T, cmap="jet")
        self.writer.add_figure("Predictions/pr", plt.gcf(), step)
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(pi_pred.T, cmap="jet")
        self.writer.add_figure("Predictions/pi", plt.gcf(), step)
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(uzr_pred.T, cmap="jet")
        self.writer.add_figure("Predictions/uzr", plt.gcf(), step)
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(uzi_pred.T, cmap="jet")
        self.writer.add_figure("Predictions/uzi", plt.gcf(), step)
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(zr_pred.T, cmap="jet")
        self.writer.add_figure("Predictions/zr", plt.gcf(), step)
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(zi_pred.T, cmap="jet")
        self.writer.add_figure("Predictions/zi", plt.gcf(), step)
        plt.close()

    def __call__(self, params, coeffs, weights, batch, step, ref_grid, ref):

        # TODO: add u_ref, p_ref, z_ref and compute l2 loss
        # alternatively, the ref could include coordinates and the eval grid could be computed from the ref grid

        if self.config.logging.log_losses:
            self.log_losses(params, coeffs, batch, step)

        if self.config.logging.log_weights:
            self.log_weights(weights, step)

        if self.config.logging.log_coeffs:
            self.log_coeffs(coeffs, step)

        if self.config.logging.log_grads:
            self.log_grads(params, coeffs, batch, step)

        if self.config.logging.log_errors:
            self.log_errors(params, ref, step)

        if self.config.logging.log_preds:
            # Transform the grid
            for axis in ref_grid:
                t = {k: v for k, v in self.transforms.items() if k.startswith(axis)}
                ref_grid[axis] = transform(ref_grid[axis], *t.values())

            self.log_preds(params, ref_grid, step)

        return self.writer

    # def _init_layout(self):
    #     layout = dict(FIGS=dict())
    #     # Losses
    #     layout["FIGS"]["Loss/PDE"] = [
    #         "Multiline",
    #         ["Loss/PDE/" + key for key in self.bvp.weights.keys() if "pde" in key],
    #     ]
    #     layout["FIGS"]["Loss/BC"] = [
    #         "Multiline",
    #         ["Loss/BC/" + key for key in self.bvp.weights.keys() if "bc" in key],
    #     ]
    #     layout["FIGS"]["Loss/Data"] = [
    #         "Multiline",
    #         ["Loss/Data/" + key for key in self.bvp.weights.keys() if "dat" in key],
    #     ]

    #     # Weights
    #     layout["FIGS"]["Weights/PDE"] = [
    #         "Multiline",
    #         ["Weights/PDE/" + key for key in self.bvp.weights.keys() if "pde" in key],
    #     ]
    #     layout["FIGS"]["Weights/BC"] = [
    #         "Multiline",
    #         ["Weights/BC/" + key for key in self.bvp.weights.keys() if "bc" in key],
    #     ]
    #     layout["FIGS"]["Weights/Data"] = [
    #         "Multiline",
    #         ["Weights/Data/" + key for key in self.bvp.weights.keys() if "dat" in key],
    #     ]

    #     # Coefficients
    #     layout["FIGS"]["Coeffs"] = [
    #         "Multiline",
    #         ["Coeffs/" + key for key in self.bvp.coefficients.keys()],
    #     ]
    #     self.writer.add_custom_scalars(layout)
