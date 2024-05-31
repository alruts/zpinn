import hydra
import jax
import train


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base=hydra.__version__,
)
def main(config):
    # Run the training and evaluation loop
    train.train_and_evaluate(config)


if __name__ == "__main__":
    main()
