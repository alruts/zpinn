import sys

import equinox as eqx
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

from jax.tree_util import tree_map, tree_leaves, tree_reduce
from jax import vmap, grad

sys.path.append("src")
from zpinn.constants import _c0, _rho0
from zpinn.impedance_models import RMK_plus_1, constant_impedance
from zpinn.utils import flatten_pytree

criteria = {
    "mse": lambda x, y: jnp.mean((x - y) ** 2),
    "mae": lambda x, y: jnp.mean(jnp.abs(x - y)),
}

unity_tf = {
    "x0": 0.0,
    "xc": 1.0,
    "y0": 0.0,
    "yc": 1.0,
    "z0": 0.0,
    "zc": 1.0,
    "a0": 0.0,
    "ac": 1.0,
    "b0": 0.0,
    "bc": 1.0,
}


@dataclass
class BVPModel:
    """Base class for the boundary value problem models."""

    criterion: Callable
    coefficients: dict
    model: eqx.Module
    x0: float
    xc: float
    y0: float
    yc: float
    z0: float
    zc: float
    a0: float
    ac: float
    b0: float
    bc: float
    impedance_model: Callable

    def __init__(
        self,
        model,
        transforms,
        impedance_model,
        criterion="mse",
    ):
        self.model = model
        self.coefficients = {}

        # Initialize the coefficients based on the impedance model
        if impedance_model == "single_freq":
            self.coefficients = {"alpha": 0.0, "beta": 0.0}
            self.impedance_model = constant_impedance

        elif impedance_model == "RMK+1":
            self.coefficients = {
                "K": 0.0,
                "R_1": 0.0,
                "M": 0.0,
                "G": 0.0,
                "gamma": 0.0,
            }
            self.impedance_model = RMK_plus_1

        else:
            raise NotImplementedError(
                "Impedance model not implemented. Choose from ['single_freq', 'RMK+1']"
            )

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
        ) = self._transforms(transforms)

        # Initialize the loss criterion
        self.criterion = criteria[criterion]

    def _transforms(self, tfs):
        """Unpack the transformation parameters."""
        x0, xc = tfs["x0"], tfs["xc"]
        y0, yc = tfs["y0"], tfs["yc"]
        z0, zc = tfs["z0"], tfs["zc"]
        f0, fc = tfs["f0"], tfs["fc"]
        a0, ac = tfs["a0"], tfs["ac"]
        b0, bc = tfs["b0"], tfs["bc"]
        return x0, xc, y0, yc, z0, zc, f0, fc, a0, ac, b0, bc

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

    def p_net(self, params, *args, part=None):
        """Pressure network."""
        p = self.apply_model(params, *args)

        if part == "real":
            return p[0]
        elif part == "imag":
            return p[1]
        else:
            return p[0], p[1]

    def r_net(self, params, *args, part="real"):
        """PDE residual network."""
        x, y, z, f = args

        p_xx = grad(grad(self.p_net, argnums=1), argnums=1)(params, *args, part=part)
        p_xx /= self.xc**2
        p_yy = grad(grad(self.p_net, argnums=2), argnums=2)(params, *args, part=part)
        p_yy /= self.yc**2
        p_zz = grad(grad(self.p_net, argnums=3), argnums=3)(params, *args, part=part)
        p_zz /= self.zc**2

        k = (2 * jnp.pi * (f * self.fc + self.f0)) / (_c0)

        if part == "real":
            p = self.p_net(params, *args, part=part) * self.ac + self.a0
        elif part == "imag":
            p = self.p_net(params, *args, part=part) * self.bc + self.b0

        # Todo: make hnet more efficient by not computing pressure twice (once for real and imag)
        return p_xx + p_yy + p_zz + k**2 * p
        

    def z_net(self, params, *args, part=None):
        """Boundary condition network."""
        x, y, z, f = args

        p_zr = grad(self.p_net, argnums=1)(params, *args, part="real")
        p_zr = p_zr * self.ac / self.xc
        p_zi = grad(self.p_net, argnums=2)(params, *args, part="imag")
        p_zi = p_zi * self.bc / self.yc

        u_cplx = 1j * 2 * jnp.pi * (f * self.fc + self.f0) * _rho0
        u_cplx *= p_zr + 1j * p_zi

        pr, pi = self.p_net(params, *args)  #! check if this is possible
        pr = pr * self.ac + self.a0
        pi = pi * self.bc + self.b0
        p_cplx = pr + 1j * pi

        z = p_cplx / u_cplx
        return z.real, z.imag

    def compute_weights(self, params, batches):
        """Computes the lambda weights for each loss."""

        # Compute the gradient of each loss w.r.t. the parameters
        grads = jax.jacrev(self.losses, argnums=0)(params, self.coeffs, **batches)

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

    def compute_coeffs(self, batch, *args):
        """Computes the coefficient updates for the impedance model."""
        pass

    #! @eqx.filter_jit
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
        pr_pred, pi_pred = vmap(self.p_net, in_axes=(None, *[0] * 4))(
            params, *(x, y, z, f)
        )
        pr_target, pi_target = gt["real_pressure"], gt["imag_pressure"]

        return self.criterion(pr_pred, pr_target), self.criterion(pi_pred, pi_target)

    def r_loss(self, params, batch):
        """PDE residual loss."""
        coords = batch
        f, x, y, z = coords.values()
        rr = vmap(self.r_net, in_axes=(None, *[0] * 4))(
            params, *(x, y, z, f), part="real"
        )
        ri = vmap(self.r_net, in_axes=(None, *[0] * 4))(
            params, *(x, y, z, f), part="imag"
        )

        return self.criterion(rr, 0.0), self.criterion(ri, 0.0)

    def z_loss(self, params, coeffs, batch):
        """Boundary loss."""
        coords = batch
        f, x, y, z = coords.values()
        z_pred = vmap(self.z_net, in_axes=(None, *[0] * 4))(params, *(x, y, z, f))
        z_est = self.impedance_model(coeffs, f)

        return self.criterion(z_pred, z_est)

    #! @eqx.filter_jit
    def loss(self, params, weights, coeffs, batches, *args):
        # Compute losses
        losses = self.losses(params, coeffs, **batches)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss
