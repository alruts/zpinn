import hydra
import jax
import train
import logging

@hydra.main(
    config_path="configs",
    config_name="f_sweep",
    version_base=hydra.__version__,
)
def main(config):
    # Run the training and evaluation loop
    print(config._idx) 
    config.batch.data.restrict_to.f = [config.batch.data.restrict_to.f[0][config._idx]]
    config.batch.domain.limits.f = config.batch.domain.limits.f[config._idx]
    config.batch.boundary.limits.f = config.batch.boundary.limits.f[config._idx]
    logging.info(f"Running sweep for f={config.batch.data.restrict_to.f}")
    logging.info(f"Domain limits: {config.batch.domain.limits.f}")
    logging.info(f"Boundary limits: {config.batch.boundary.limits.f}")

    train.train_and_evaluate(config)


if __name__ == "__main__":
    main()
