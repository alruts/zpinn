import logging
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
from jax.numpy import log
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

sys.path.append("src")
from zpinn.loss_fns import (
    data_loss_fn,
    helmholtz_residual,
    bc_residual_single_val,
    criteria
)

from zpinn.callbacks import ModelEvaluator
from zpinn.plot.fields import scalar_field


def train_model(
    model,
    lambdas,
    coeffs,
    loss_type,
    trainloader,
    optim_model,
    optim_lambdas,
    optim_coeffs,
    steps,
    print_every,
    transforms,
    writer,
    evaluate_every=100,
    dom_gen=None,
    bnd_gen=None,
    eval_gen=None,
    key=jrandom.PRNGKey(0),
):
    @eqx.filter_jit
    def update(
        model,
        lambdas,
        coeffs,
        state_model,
        state_lambdas,
        state_coeffs,
        batch,
        dom_gen,
        bnd_gen,
        transforms,
        key,
    ):
        """Update the model using the optimizer and the batch."""

        def compute_loss(
            model,
            epsilons,
            Zn,
            batch,
            loss_type,
            domain_collocation_generator=None,
            transforms=None,
            key=None,
        ):
            """Compute the combined loss function."""
            # bookkeeping
            pde_residuals_re = []
            pde_residuals_im = []
            losses = []
            criterion = criteria[loss_type]

            # Compute data loss of type 'loss_type'
            dat_loss_re, dat_loss_im = data_loss_fn(model, loss_type, batch)
            data_loss = dat_loss_re + dat_loss_im

            # Compute boundary loss and pde loss on boundary
            if bnd_gen is not None:

                # get key for collocation points
                key, collocation_key = jrandom.split(key)
                bnd_coords = bnd_gen.gen_data(collocation_key)

                # compute boundary loss
                res_re_bnd_, res_im_bnd_, std_re, std_im = bc_residual_single_val(
                    model, bnd_coords, transforms, Zn
                )
                
                bnd_loss_re = criterion(res_re_bnd_, 0)
                bnd_loss_im = criterion(res_im_bnd_, 0)
                boundary_loss = bnd_loss_re + bnd_loss_im + jnp.abs(std_re + std_im)

                # bookkeeping
                aux["boundary_loss"] = boundary_loss
                aux["boundary_loss_re"] = bnd_loss_re
                aux["boundary_loss_im"] = bnd_loss_im

                #  also compute pde loss on boundary
                res_re, res_im = helmholtz_residual(model, bnd_coords, transforms)

                # append to pde residuals
                pde_residuals_re.append(res_re)
                pde_residuals_im.append(res_im)

            # PDE loss
            if domain_collocation_generator is not None:
                # domain collocation points
                key, collocation_key = jrandom.split(key)
                domain_coords = domain_collocation_generator.gen_data(collocation_key)

                # compute pde residuals
                res_re, res_im = helmholtz_residual(model, domain_coords, transforms)

                # append to pde residuals
                pde_residuals_re.append(res_re)
                pde_residuals_im.append(res_im)

                # compute pde loss on batch (data points)
                res_re, res_im = helmholtz_residual(model, batch, transforms)

                # append to pde residuals
                pde_residuals_re.append(res_re)
                pde_residuals_im.append(res_im)

                # compute total pde loss
                pde_residuals_re = jnp.concatenate(pde_residuals_re, axis=0)
                pde_residuals_im = jnp.concatenate(pde_residuals_im, axis=0)
                pde_loss_re = criterion(pde_residuals_re, 0)
                pde_loss_im = criterion(pde_residuals_im, 0)
                pde_loss = pde_loss_re + pde_loss_im

                # bookkeeping
                aux["pde_loss"] = pde_loss
                aux["pde_loss_re"] = pde_loss_re
                aux["pde_loss_im"] = pde_loss_im

            else:
                # otherwise just compute on batch
                res_re, res_im = helmholtz_residual(model, batch, transforms)
                pde_loss_re = criterion(res_re, 0)
                pde_loss_im = criterion(res_im, 0)
                pde_loss = pde_loss_re + pde_loss_im

                aux["pde_loss"] = pde_loss
                aux["pde_loss_re"] = pde_loss_re
                aux["pde_loss_im"] = pde_loss_im

            losses.append(data_loss)
            losses.append(pde_loss)
            losses.append(boundary_loss)

            # hard constrain epsilon to be 1.0 or greater
            # epsilons["bc"] = jnp.maximum(epsilons["bc"], 1.0)

            # Combine losses and weights
            total_loss = 0.0
            for loss_, eps_ in zip(
                losses,
                [epsilons["data"], epsilons["pde"], epsilons["bc"]],
            ):
                total_loss += 1 / (2 * eps_**2) * loss_
                if use_adaptive_eps:
                    total_loss += log(eps_)

            # bookkeeping
            aux["loss"] = total_loss
            aux["key"] = key

            return total_loss, aux

        # Model update
        model_loss_grad = eqx.filter_value_and_grad(compute_loss, has_aux=True)
        (loss, aux), grads = model_loss_grad(
            model,
            lambdas,
            coeffs,
            batch,
            loss_type,
            dom_gen,
            transforms,
            key=key,
        )
        updates, state_model = optim_model.update(grads, state_model, model)
        model = eqx.apply_updates(model, updates)

        if bnd_gen is not None:
            # Impedance update
            grads, _ = jax.grad(compute_loss, argnums=2, has_aux=True)(
                model,
                lambdas,
                coeffs,
                batch,
                loss_type,
                bnd_gen,
                transforms,
                key=key,
            )
            updates, state_coeffs = optim_coeffs.update(grads, state_coeffs)
            coeffs = eqx.apply_updates(coeffs, updates)

        # Epsilon update
        if use_adaptive_eps:
            grads, _ = jax.grad(compute_loss, argnums=1, has_aux=True)(
                model,
                lambdas,
                coeffs,
                batch,
                loss_type,
                dom_gen,
                transforms,
                key=key,
            )

            updates, state_lambdas = optim_lambdas.update(grads, state_lambdas)
            lambdas = eqx.apply_updates(lambdas, updates)

        return (
            model,
            lambdas,
            coeffs,
            state_model,
            state_lambdas,
            state_coeffs,
            loss,
            aux,
        )

    # Initialize optimizer states
    state_model = optim_model.init(eqx.filter(model, eqx.is_array))
    state_Zn = optim_coeffs.init(coeffs)

    # Train model
    for step in tqdm(range(steps)):
        batch = next(iter(trainloader))

        assert len(batch) == 2, "Batch should contain coordinates and ground truth"

        # Update model, epsilons and Zn
        model, lambdas, coeffs, state_model, state_eps, state_Zn, loss, aux = update(
            model,
            lambdas,
            coeffs,
            state_model,
            state_eps,
            state_Zn,
            batch,
            dom_gen,
            bnd_gen,
            transforms,
            key=key,
        )

        # update key for collocation points
        key = aux["key"]

        if step % print_every == 0:

    return model
