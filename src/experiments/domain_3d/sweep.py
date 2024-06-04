import hydra
import jax
import train
import logging


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base=hydra.__version__,
)
def run_sweep(config):
    param_dict = {
        "hidden_features": [32, 64, 128],
        "hidden_layers": [3, 4, 5],
        "architectures": ["modified_siren", "siren"],
    }

    # Sweep over the hyperparameters
    for hidden_features in param_dict["hidden_features"]:
        for hidden_layers in param_dict["hidden_layers"]:
            for architecture in param_dict["architectures"]:
                config.architecture.hidden_features = hidden_features
                config.architecture.hidden_layers = hidden_layers
                config.architecture.name = architecture
                config.experiment.name = f"{hidden_features}_{hidden_layers}_{architecture}"
                logging.info(f"Running experiment with {config.experiment.name}")
                train.train_and_evaluate(config)


if __name__ == "__main__":
    run_sweep()
