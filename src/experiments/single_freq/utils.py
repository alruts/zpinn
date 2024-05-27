from zpinn.get_loaders import get_loaders
import optax


def setup_loaders(config):
    return get_loaders(config, restrict_to=config.batch.data.restrict_to)


def setup_optimizers(config):
    lr = optax.exponential_decay(
        init_value=config.training.optim.params.lr,
        transition_steps=config.training.optim.params.transition_steps,
        decay_rate=config.training.optim.params.decay_rate,
    )

    optimizer_params = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=lr),
    )

    # lr = optax.cosine_onecycle_schedule(
    #     transition_steps=config.training.optim.coeffs.transition_steps,
    #     peak_value=config.training.optim.coeffs.lr,
    #     pct_start=config.training.optim.coeffs.pct_start,
    # )
    
    lr = config.training.optim.coeffs.lr
    optimizer_coeffs = optax.chain(
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
