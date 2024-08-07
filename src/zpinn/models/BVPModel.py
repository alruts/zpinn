import sys
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import grad, lax, vmap
from jax.tree_util import tree_leaves, tree_map, tree_reduce

sys.path.append("src")
from zpinn.constants import _c0, _rho0
from zpinn.impedance_models import RMK_plus_1, constant_impedance, R_plus_2
from zpinn.utils import flatten_pytree, cat_batches

criteria = {
    "mse": lambda x, y, axis=None: jnp.mean((x - y) ** 2, axis),
    "mae": lambda x, y, axis=None: jnp.mean(jnp.abs(x - y), axis),
}


class BVPModel(eqx.Module):
    """Base class for the boundary value problem models."""

    model: eqx.Module
    criterion: Callable
    momentum: float
    impedance_model: Callable
    is_normalized: bool
    coeffs: dict
    weights: dict
    weighting_scheme: str
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
    use_locality: bool
    num_chunks: int
    tol: float
    M: jnp.ndarray
    use_boundary_loss: bool
    use_std_loss: bool
    use_gpinn: bool

    def __init__(
        self, model, transforms, config, params=None, weights=None, coeffs=None
    ):
        self.model = model
        self.coeffs = (
            dict(config.impedance_model.initial_guess) if coeffs is None else coeffs
        )
        self.momentum = config.weighting.momentum
        self.weighting_scheme = config.weighting.scheme
        self.impedance_model = self._init_impedance_model(config)
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

        self.use_locality = config.weighting.use_locality
        self.tol = config.weighting.tol
        self.num_chunks = config.weighting.num_chunks
        self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T
        self.use_boundary_loss = config.weighting.use_boundary_loss
        self.use_std_loss = config.weighting.use_std_loss
        self.use_gpinn = config.weighting.use_gpinn

        self.criterion = criteria[config.training.criterion]
        self.weights = self._init_weights() if weights is None else weights

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
        """Initialize the weights."""
        weights = {}
        keys = ["data_re", "data_im", "pde_re", "pde_im"]

        if self.use_boundary_loss:
            keys += ["bc_re", "bc_im"]

        if self.use_gpinn:
            keys += ["gx_re", "gx_im", "gy_re", "gy_im", "gz_re", "gz_im"]

        if self.use_std_loss:
            keys += ["std_re", "std_im"]

        for key in keys:
            weights[key] = jnp.array(1.0)

        return weights

    def _init_impedance_model(self, config):
        # Initialize the coefficients based on the impedance model
        if config.impedance_model.type == "single_freq":
            z_model = constant_impedance
        elif config.impedance_model.type == "RMK+1":
            z_model = RMK_plus_1
        elif config.impedance_model.type == "R+2":
            z_model = R_plus_2

        else:
            raise NotImplementedError(
                "Impedance model not implemented. Choose from ['single_freq', 'RMK+1', 'R+2']"
            )

        return z_model

    def unpack_coords(self, coords):
        """Unpack the coordinates and ground truth."""
        return coords["x"], coords["y"], coords["z"], coords["f"]

    def apply_model(self, params, *args):
        """Trick to enable gradient with respect to weights."""
        _, static = eqx.partition(self.model, eqx.is_inexact_array)
        model = eqx.combine(params, static)
        return model(*args[: model.in_features])

    def get_parameters(self):
        """Returns the parameters of the model."""
        params, _ = eqx.partition(self.model, eqx.is_inexact_array)
        return params

    def p_pred_fn(self, params, *args):
        """Predict pressure over a grid."""
        return vmap(
            vmap(self.p_net, (None, None, 0, None, None)), (None, 0, None, None, None)
        )(params, *args)

    def un_pred_fn(self, params, *args):
        """Predict particle velocity over a grid."""
        return vmap(
            vmap(self.un_net, (None, None, 0, None, None)), (None, 0, None, None, None)
        )(params, *args)

    def z_pred_fn(self, params, *args):
        """Predict impedance over a grid."""
        return vmap(
            vmap(self.z_net, (None, None, 0, None, None)), (None, 0, None, None, None)
        )(params, *args)

    def psi_net(self, params, *args, part=None):
        """Nondimensionalized pressure network."""
        psi = self.apply_model(params, *args)

        if part == "real":
            return psi[0]
        elif part == "imag":
            return psi[1]
        else:
            return psi[0], psi[1]

    def p_net(self, params, *args):
        """Pressure network."""
        pr, pi = self.psi_net(params, *args)
        pr = pr * self.ac + self.a0
        pi = pi * self.bc + self.b0
        return pr, pi

    def r_net(self, params, *args, part=None):
        """PDE residual network."""
        x, y, z, f = args
        k = 2 * jnp.pi * (f * self.fc + self.f0) / _c0
        pr, pi = self.psi_net(params, *args)

        # compute real part
        pr_xx = grad(grad(self.psi_net, argnums=1), argnums=1)(
            params, *args, part="real"
        )
        pr_yy = grad(grad(self.psi_net, argnums=2), argnums=2)(
            params, *args, part="real"
        )
        pr_zz = grad(grad(self.psi_net, argnums=3), argnums=3)(
            params, *args, part="real"
        )

        res_re = self.ac * (
            (self.yc * self.zc) ** 2 * pr_xx
            + (self.xc * self.zc) ** 2 * pr_yy
            + (self.xc * self.yc) ** 2 * pr_zz
        ) + (self.xc * self.yc * self.zc * k) ** 2 * (pr * self.ac + self.a0)

        # compute imag part
        pi_xx = grad(grad(self.psi_net, argnums=1), argnums=1)(
            params, *args, part="imag"
        )
        pi_yy = grad(grad(self.psi_net, argnums=2), argnums=2)(
            params, *args, part="imag"
        )
        pi_zz = grad(grad(self.psi_net, argnums=3), argnums=3)(
            params, *args, part="imag"
        )
        res_im = self.bc * (
            (self.yc * self.zc) ** 2 * pi_xx
            + (self.xc * self.zc) ** 2 * pi_yy
            + (self.xc * self.yc) ** 2 * pi_zz
        ) + (self.xc * self.yc * self.zc * k) ** 2 * (pi * self.bc + self.b0)

        if part == "real":
            return res_re
        elif part == "imag":
            return res_im
        else:
            return res_re, res_im

    def un_net(self, params, *args):
        """Normal particle velocity network."""
        x, y, z, f = args

        # compute the gradient of the pressure w.r.t. z
        pr_z = grad(self.psi_net, argnums=3)(params, *args, part="real")
        pi_z = grad(self.psi_net, argnums=3)(params, *args, part="imag")

        # apply euler's equation
        uz = 1 / (1j * 2 * jnp.pi * (f * self.fc + self.f0) * _rho0)
        uz /= self.zc
        uz *= self.ac * pr_z + 1j * self.bc * pi_z

        return uz.real, uz.imag

    def z_net(self, params, *args):
        """Boundary condition network."""
        # compute the particle velocity
        unr, uni = self.un_net(params, *args)
        u_cplx = unr + 1j * uni

        # compute the pressure
        pr, pi = self.p_net(params, *args)
        p_cplx = pr + 1j * pi

        z = p_cplx / u_cplx

        return z.real, z.imag

    def g_net(self, params, *args):
        """Gradient of the residual."""
        # take spatial derivatives of the r net
        # gr, gi = jax.jacfwd(self.r_net, argnums=(1, 2, 3))(params, *args)
        # gr_x, gr_y, gr_z = gr
        # gi_x, gi_y, gi_z = gi

        gr_x = grad(self.r_net, argnums=1)(params, *args, part="real")
        gr_y = grad(self.r_net, argnums=2)(params, *args, part="real")
        gr_z = grad(self.r_net, argnums=3)(params, *args, part="real")
        gi_x = grad(self.r_net, argnums=1)(params, *args, part="imag")
        gi_y = grad(self.r_net, argnums=2)(params, *args, part="imag")
        gi_z = grad(self.r_net, argnums=3)(params, *args, part="imag")

        return gr_x, gr_y, gr_z, gi_x, gi_y, gi_z

    def p_loss(self, params, batch):
        """Data (pressure) loss."""
        coords, gt = batch  # unpack the data batch
        x, y, z, f = self.unpack_coords(coords)
        pr_pred, pi_pred = vmap(self.psi_net, in_axes=(None, *[0] * 4))(
            params, *(x, y, z, f)
        )
        pr_target, pi_target = gt["real_pressure"], gt["imag_pressure"]
        return self.criterion(pr_pred, pr_target), self.criterion(pi_pred, pi_target)

    def r_loss(self, params, batch):
        """PDE residual loss."""
        coords = batch
        x, y, z, f = self.unpack_coords(coords)
        rr, ri = vmap(self.r_net, in_axes=(None, *[0] * 4))(params, *(x, y, z, f))
        return self.criterion(rr, 0.0), self.criterion(ri, 0.0)

    def z_loss(self, params, coeffs, batch):
        """Boundary loss."""
        coords = batch
        x, y, z, f = self.unpack_coords(coords)
        zr_pred, zi_pred = vmap(self.z_net, in_axes=(None, *[0] * 4))(
            params, *(x, y, z, f)
        )
        zr_mdl, zi_mdl = self.impedance_model(
            coeffs, f * self.fc + self.f0, self.is_normalized
        )

        # if is_normalized, model will fit normalized impedance (zeta = z / rho0 * c0)
        if self.is_normalized:
            zi_pred /= _rho0 * _c0
            zr_pred /= _rho0 * _c0

        return self.criterion(zr_pred, zr_mdl), self.criterion(zi_pred, zi_mdl)

    def std_loss(self, params, batch):
        """Boundary loss."""
        coords = batch
        x, y, z, f = self.unpack_coords(coords)
        zr_pred, zi_pred = vmap(self.z_net, in_axes=(None, *[0] * 4))(
            params, *(x, y, z, f)
        )

        # if is_normalized, model will fit normalized impedance (zeta = z / rho0 * c0)
        if self.is_normalized:
            zi_pred /= _rho0 * _c0
            zr_pred /= _rho0 * _c0

        return jnp.std(zr_pred), jnp.std(zi_pred)

    def g_loss(self, params, batch):
        """gPINNS loss."""
        coords = self.unpack_coords(batch)
        gr_x, gr_y, gr_z, gi_x, gi_y, gi_z = vmap(self.g_net, in_axes=(None, *[0] * 4))(
            params, *(coords)
        )
        return (
            self.criterion(gr_x, 0.0),
            self.criterion(gr_y, 0.0),
            self.criterion(gr_z, 0.0),
            self.criterion(gi_x, 0.0),
            self.criterion(gi_y, 0.0),
            self.criterion(gi_z, 0.0),
        )

    def res_and_w(self, params, batch):
        """Respecting locality in z-direction."""
        # unpack the data
        x, y, z, f = self.unpack_coords(batch)
        z = -jnp.sort(-z)  # sort in descending order

        # compute the residual
        rr, ri = vmap(self.r_net, in_axes=(None, *[0] * 4))(params, *(x, y, z, f))

        # split into chunks
        rr, ri = rr.reshape(self.num_chunks, -1), ri.reshape(self.num_chunks, -1)
        lr, li = self.criterion(rr, 0.0, axis=1), self.criterion(ri, 0.0, axis=1)

        # compute the weights
        wr = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ lr)))
        wi = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ li)))

        return lr, li, wr, wi

    @eqx.filter_jit
    def compute_weights(
        self, params, coeffs, dat_batch, dom_batch, bnd_batch, **kwargs
    ):
        """Computes the lambda weights for each loss."""

        # Compute the gradient of each loss w.r.t. the parameters
        grads = jax.jacrev(self.losses, **kwargs)(
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
    def update_weights(self, old_w, new_w):
        """Updates `weights` using running average with momentum."""
        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, old_w, new_w)

        weights = lax.stop_gradient(weights)

        return weights

    @eqx.filter_jit
    def compute_coeffs(self, params, coeffs, batch):
        """Computes the gradient of the loss w.r.t. the coefficients."""
        grad_re, grad_im = jax.jacrev(self.z_loss, argnums=1)(params, coeffs, batch)

        # add real and imag parts to form one gradient
        updates = tree_map(lambda x, y: x + y, grad_re, grad_im)

        return updates

    @eqx.filter_jit
    def losses(self, params, coeffs, dat_batch, dom_batch, bnd_batch):
        """Returns the losses of the model."""
        loss_dict = {}

        # data loss
        data_re, data_im = self.p_loss(params, dat_batch)
        loss_dict["data_re"], loss_dict["data_im"] = data_re, data_im

        # pde loss
        if self.use_locality:
            lr, li, wr, wi = self.res_and_w(params, dom_batch)
            pde_re, pde_im = jnp.mean(wr * lr), jnp.mean(wi * li)

        else:
            pde_re, pde_im = self.r_loss(params, dom_batch)

        loss_dict["pde_re"], loss_dict["pde_im"] = pde_re, pde_im

        # boundary loss
        if self.use_boundary_loss:
            bc_re, bc_im = self.z_loss(params, coeffs, bnd_batch)
            loss_dict["bc_re"], loss_dict["bc_im"] = bc_re, bc_im

        # std loss
        if self.use_std_loss:
            std_re, std_im = self.std_loss(params, bnd_batch)
            loss_dict["std_re"], loss_dict["std_im"] = std_re, std_im

        # gradient residual loss
        if self.use_gpinn:
            gr_x, gr_y, gr_z, gi_x, gi_y, gi_z = self.g_loss(params, dom_batch)
            loss_dict["gx_re"], loss_dict["gx_im"] = gr_x, gi_x
            loss_dict["gy_re"], loss_dict["gy_im"] = gr_y, gi_y
            loss_dict["gz_re"], loss_dict["gz_im"] = gr_z, gi_z

        return loss_dict

    @eqx.filter_jit
    def bc_strategy(self, losses):
        """Boundary condition balancing strategy."""
        a = 1.0
        b = None
        C = 1e-3
        
        # save first time step losses
        if b is None:
            b = {}
            b["re"] = losses["data_re"] + losses["pde_re"]
            b["im"] = losses["data_im"] + losses["pde_im"]

        exp_decay = lambda x, a, b: jnp.exp(-a / b * x)

        w_re = C * exp_decay(losses["data_re"] + losses["pde_re"], a, b["re"])
        w_im = C * exp_decay(losses["data_im"] + losses["pde_im"], a, b["im"])

        losses["bc_re"] = w_re * losses["bc_re"]
        losses["bc_im"] = w_im * losses["bc_im"]

        return losses

    @eqx.filter_jit
    def compute_loss(self, params, weights, coeffs, dat_batch, dom_batch, bnd_batch):
        # Compute losses
        losses = self.losses(params, coeffs, dat_batch, dom_batch, bnd_batch)

        if self.weighting_scheme == "grad_norm":
            if self.use_boundary_loss:
                # Apply boundary condition balancing strategy
                losses = self.bc_strategy(losses)

            # Compute weighted loss
            weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
            # Sum weighted losses
            loss = tree_reduce(lambda x, y: x + y, weighted_losses)

        elif self.weighting_scheme == "mle":
            # Compute log regularization term
            loss = jnp.log(tree_reduce(lambda x, y: x * y, weights))
            # convert to lambdas
            lambdas = jax.tree_map(lambda x: 1 / (2 * x**2), weights)
            # Compute weighted loss
            weighted_losses = tree_map(lambda x, y: x * y, losses, lambdas)
            # Sum weighted losses
            loss += tree_reduce(lambda x, y: x + y, weighted_losses)

        return loss

    @eqx.filter_jit
    def update(self, params, weights, coeffs, opt_states, optimizers, batches):
        """Update the model parameters."""
        # --- Update the coefficients ---
        grads = self.compute_coeffs(params, coeffs, batches["bnd_batch"])
        updates, opt_states["coeffs"] = optimizers["coeffs"].update(
            grads, opt_states["coeffs"]
        )
        coeffs = eqx.apply_updates(coeffs, updates)

        # Constraint RMK+1 model
        if self.impedance_model == RMK_plus_1:
            coeffs = {
                "K": jnp.maximum(coeffs["K"], 0.0),
                "R_1": jnp.maximum(coeffs["R_1"], 0.0),
                "M": jnp.maximum(coeffs["M"], 0.0),
                "G": jnp.maximum(coeffs["G"], 0.0),
                "gamma": jnp.clip(coeffs["gamma"], -1.0, 1.0),
            }

        if self.impedance_model == R_plus_2:
            coeffs = {
                "A": jnp.maximum(coeffs["A"], 0.0),
                "R_2": jnp.maximum(coeffs["R_2"], 0.0),
                "B": jnp.maximum(coeffs["B"], 0.0),
                "alpha": jnp.clip(coeffs["alpha"], -1.0, 1.0),
                "beta": jnp.clip(coeffs["beta"], -1.0, 1.0),
            }

        # --- Update the parameters and return ---
        if self.weighting_scheme == "grad_norm":
            grads = jax.grad(self.compute_loss)(params, weights, coeffs, **batches)
            updates, opt_states["params"] = optimizers["params"].update(
                grads, opt_states["params"]
            )
            params = eqx.apply_updates(params, updates)
            return params, coeffs, opt_states

        elif self.weighting_scheme == "mle":
            # params
            grads = jax.grad(self.compute_loss)(params, weights, coeffs, **batches)
            updates, opt_states["params"] = optimizers["params"].update(
                grads, opt_states["params"]
            )
            params = eqx.apply_updates(params, updates)

            # weights
            grads = jax.jacrev(self.compute_loss, argnums=1)(
                params, weights, coeffs, **batches
            )
            updates, opt_states["weights"] = optimizers["weights"].update(
                grads, opt_states["weights"]
            )
            weights = eqx.apply_updates(weights, updates)
            return params, weights, coeffs, opt_states

    @eqx.filter_jit
    def compute_l2_error(self, params, coords, ref):
        """Compute relative L2 error."""
        pr_star = ref["real_pressure"]
        pi_star = ref["imag_pressure"]
        unr_star = ref["real_velocity"]
        uzi_star = ref["imag_velocity"]

        z_cmplx = ref["real_pressure"] + 1j * ref["imag_pressure"]
        z_cmplx /= ref["real_velocity"] + 1j * ref["imag_velocity"]
        zr_star, zi_star = z_cmplx.real, z_cmplx.imag

        x, y, z, f = self.unpack_coords(coords)
        pr_pred, pi_pred = self.p_pred_fn(params, *(x, y, z, f))
        unr_pred, uni_pred = self.un_pred_fn(params, *(x, y, z, f))
        zr_pred, zi_pred = self.z_pred_fn(params, *(x, y, z, f))

        error_fn = lambda x, y: jnp.linalg.norm(x - y) / (jnp.linalg.norm(y) + 1e-12)

        errors = dict(
            pr=error_fn(pr_star, pr_pred),
            pi=error_fn(pi_star, pi_pred),
            unr=error_fn(unr_star, unr_pred),
            uni=error_fn(uzi_star, uni_pred),
            zr=error_fn(zr_star, zr_pred),
            zi=error_fn(zi_star, zi_pred),
        )
        return errors

    @eqx.filter_jit
    def compute_relative_error(self, params, coords, ref):
        """Compute relative error."""
        pr_star = ref["real_pressure"]
        pi_star = ref["imag_pressure"]
        unr_star = ref["real_velocity"]
        uzi_star = ref["imag_velocity"]

        z_cmplx = ref["real_pressure"] + 1j * ref["imag_pressure"]
        z_cmplx /= ref["real_velocity"] + 1j * ref["imag_velocity"]
        zr_star, zi_star = z_cmplx.real, z_cmplx.imag

        x, y, z, f = self.unpack_coords(coords)
        pr_pred, pi_pred = self.p_pred_fn(params, *(x, y, z, f))
        unr_pred, uni_pred = self.un_pred_fn(params, *(x, y, z, f))
        zr_pred, zi_pred = self.z_pred_fn(params, *(x, y, z, f))

        error_fn = lambda x, y: (x - y) / (jnp.abs(x) + jnp.abs(y))

        errors = dict(
            pr=error_fn(pr_star, pr_pred),
            pi=error_fn(pi_star, pi_pred),
            unr=error_fn(unr_star, unr_pred),
            uni=error_fn(uzi_star, uni_pred),
            zr=error_fn(zr_star, zr_pred),
            zi=error_fn(zi_star, zi_pred),
        )
        return errors
