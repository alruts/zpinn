import argparse
import logging

import hydra
import jax
import train

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config_file",
    type=str,
    default="default.yaml",
    help="Path to the config file",
)
args = parser.parse_args()

@hydra.main(
    config_path=".\configs", config_name="default", version_base=hydra.__version__
)
def main(config):

    # Run the training and evaluation loop
    train.train_and_evaluate(config)


if __name__ == "__main__":
    
    main()
