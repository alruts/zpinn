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


@dataclass
class BVPModel:
    """Base class for the boundary value problem models."""

    model: eqx.Module
    criterion: Callable
    coefficients: dict
    weights: dict
    momentum: float
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
        momentum=0.9,
    ):
        self.model = model
        self.momentum = momentum
        self.weights = self._init_weights()
        self.coefficients, self.impedance_model = self._init_z(impedance_model)
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

        # Initialize the loss criterion
        self.criterion = criteria[criterion]

    def _init_transforms(self, tfs):
        """Unpack the transformation parameters."""
        x0, xc = tfs["x0"], tfs["xc"]
        y0, yc = tfs["y0"], tfs["yc"]
        z0, zc = tfs["z0"], tfs["zc"]
        f0, fc = tfs["f0"], tfs["fc"]
        a0, ac = tfs["a0"], tfs["ac"]
        b0, bc = tfs["b0"], tfs["bc"]
        return x0, xc, y0, yc, z0, zc, f0, fc, a0, ac, b0, bc

    def _init_weights(self):
        """Initialize the weights for each loss."""
        return {
            "data_re": 1.0,
            "data_im": 1.0,
            "pde_re": 1.0,
            "pde_im": 1.0,
            "bc_re": 1.0,
            "bc_im": 1.0,
        }  #! add config for this

    def _init_z(self, impedance_model):
        # Initialize the coefficients based on the impedance model
        if impedance_model == "single_freq":
            coefficients = {"alpha": 1.0, "beta": -1.0}
            z_model = constant_impedance

        elif impedance_model == "RMK+1":
            coefficients = {
                "K": 0.0,
                "R_1": 0.0,
                "M": 0.0,
                "G": 0.0,
                "gamma": 0.0,
            }
            z_model = RMK_plus_1
        else:
            raise NotImplementedError(
                "Impedance model not implemented. Choose from ['single_freq', 'RMK+1']"
            )

        return coefficients, z_model

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

    def r_net(self, params, *args):
        """PDE residual network."""
        x, y, z, f = args
        k = (2 * jnp.pi * (f * self.fc + self.f0)) / (_c0)
        pr_pred, pi_pred = self.p_net(params, *args)

        pr = pr_pred * self.ac + self.a0
        pi = pi_pred * self.bc + self.b0

        norm = self.xc * self.yc * self.zc

        # compute real part of the Laplacian
        p_xx = grad(grad(self.p_net, argnums=1), argnums=1)(params, *args, part="real")
        p_xx /= self.xc**2
        p_yy = grad(grad(self.p_net, argnums=2), argnums=2)(params, *args, part="real")
        p_yy /= self.yc**2
        p_zz = grad(grad(self.p_net, argnums=3), argnums=3)(params, *args, part="real")
        p_zz /= self.zc**2
        Lr = p_xx + p_yy + p_zz

        # compute imaginry part of the Laplacian
        p_xx = grad(grad(self.p_net, argnums=1), argnums=1)(params, *args, part="imag")
        p_xx /= self.xc**2
        p_yy = grad(grad(self.p_net, argnums=2), argnums=2)(params, *args, part="imag")
        p_yy /= self.yc**2
        p_zz = grad(grad(self.p_net, argnums=3), argnums=3)(params, *args, part="imag")
        p_zz /= self.zc**2
        Li = p_xx + p_yy + p_zz

        return norm * (Lr + k**2 * pr), norm * (Li + k**2 * pi)

    def z_net(self, params, *args):
        """Boundary condition network."""
        x, y, z, f = args

        p_zr = grad(self.p_net, argnums=1)(params, *args, part="real")
        p_zr = p_zr * self.ac / self.xc
        p_zi = grad(self.p_net, argnums=2)(params, *args, part="imag")
        p_zi = p_zi * self.bc / self.yc

        u_cplx = 1j * 2 * jnp.pi * (f * self.fc + self.f0) * _rho0
        u_cplx *= p_zr + 1j * p_zi

        pr, pi = self.p_net(params, *args)
        pr = pr * self.ac + self.a0
        pi = pi * self.bc + self.b0
        p_cplx = pr + 1j * pi

        z = p_cplx / u_cplx
        return z.real, z.imag

    def compute_weights(self, dat_batch, dom_batch, bnd_batch):
        """Computes the lambda weights for each loss."""

        # Compute the gradient of each loss w.r.t. the parameters
        grads = jax.jacrev(self.losses, argnums=0)(
            self.parameters(), self.coefficients, dat_batch, dom_batch, bnd_batch
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

    def update_weights(self, weights):
        """Updates `self.weights` using running average."""

        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        self.weights = tree_map(running_average, self.weights, weights)

    def grad_coeffs(self, dat_batch, dom_batch, bnd_batch):
        """Computes the gradient of the loss w.r.t. the coefficients."""
        grads = jax.jacrev(self.losses, argnums=1)(
            self.parameters(), self.coefficients, dat_batch, dom_batch, bnd_batch
        )

        coeff_keys = self.coefficients.keys()
        subdicts = [grads[key] for key in grads.keys()]

        sum_grad_dict = {}
        for key in coeff_keys:
            sum_grad_dict[key] = jnp.sum(jnp.stack([d[key] for d in subdicts]))

        return sum_grad_dict

    def update_coeffs(self, sum_grad_dict, opt, opt_state):
        """Updates the coefficients using the gradient."""
        updates, opt_state = opt.update(sum_grad_dict, opt_state)
        coeffs = eqx.apply_updates(self.coefficients, updates)

        # Clip the coefficients if using RMK+1 model
        if self.impedance_model == RMK_plus_1:
            coeffs = {
                "K": jnp.clip(coeffs["K"], 0.0, 1.0),
                "R_1": jnp.clip(coeffs["R_1"], 0.0, 1.0),
                "M": jnp.clip(coeffs["M"], 0.0, 1.0),
                "G": jnp.clip(coeffs["G"], 0.0, 1.0),
                "gamma": jnp.clip(coeffs["gamma"], -1.0, 1.0),
            }

        return coeffs, opt_state

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
        rr, ri = vmap(self.r_net, in_axes=(None, *[0] * 4))(params, *(x, y, z, f))
        return self.criterion(rr, 0.0), self.criterion(ri, 0.0)

    def z_loss(self, params, coeffs, batch):
        """Boundary loss."""
        coords = batch
        f, x, y, z = coords.values()
        zpr, zpi = vmap(self.z_net, in_axes=(None, *[0] * 4))(params, *(x, y, z, f))
        zmr, zmi = self.impedance_model(coeffs, f * self.fc + self.f0)

        return self.criterion(zpr, zmr), self.criterion(zpi, zmi)

    def compute_loss(self, params, weights, coeffs, batches, *args):
        # Compute losses
        losses = self.losses(params, coeffs, **batches)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss
