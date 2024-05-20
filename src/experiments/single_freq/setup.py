from zpinn.get_loaders import get_loaders
import optax


def setup_loaders(config):
    return get_loaders(config, restrict_to=config.batch.data.restrict_to)


def setup_optimizers(config):
    # lr schedule
    lr = optax.join_schedules(
        schedules=[
            optax.constant_schedule(config.training.learning_rate.params),
            optax.cosine_decay_schedule(
                config.training.learning_rate.params, config.training.steps // 2
            ),
        ],
        boundaries=[config.training.steps // 2],
    )
    optimizer_params = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=lr),
    )
    optimizer_coeffs = optax.chain(
        optax.adam(learning_rate=config.training.learning_rate.coeffs),
    )
    optimizers = dict(
        params=optimizer_params,
        coeffs=optimizer_coeffs,
    )

    return lr, optimizers
