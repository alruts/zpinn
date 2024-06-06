from zpinn.get_loaders import get_loaders
import optax


def setup_loaders(config):
    return get_loaders(config, restrict_to=config.batch.data.restrict_to)


def setup_optimizers(config):
    # Learning rate schedule for the parameters
    lr = optax.join_schedules(
        schedules=[
            optax.exponential_decay(
                init_value=config.training.optim.params.lr,
                transition_steps=config.training.optim.params.transition_steps,
                decay_rate=config.training.optim.params.decay_rate,
            ),
            optax.linear_schedule(0, config.training.optim.params.lr, 5000),
        ],
        boundaries=[5000],
    )

    optimizer_params = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=lr),
    )

    # schedule for the coefficients
    lr = optax.linear_schedule(0, config.training.optim.coeffs.lr, 5000)
    
    optimizer_coeffs = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=lr),
    )

    optimizers = dict(
        params=optimizer_params,
        coeffs=optimizer_coeffs,
    )

    if config.weighting.scheme == "mle":
        optimizer_weights = optax.chain(
            optax.adam(learning_rate=config.training.optim.weights.lr),
        )
        optimizers["weights"] = optimizer_weights

    return optimizers
