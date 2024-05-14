import argparse
import sys

import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

sys.path.append("src")
from zpinn.plot.fields import plot_scalar_field_grid, scalar_field

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_config",
    type=str,
    default=None,
    help="path to the experiment config file",
)
parser.add_argument(
    "--dataset_config",
    type=str,
    default=None,
    help="path to the experiment config file",
)
args = parser.parse_args()

if None in [args.experiment_config, args.dataset_config]:
    raise ValueError("Please provide a path to the config files")

CONFIG = OmegaConf.load(args.experiment_config)
CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load(args.dataset_config))


def main(cfg=CONFIG):
    # Load dataset
    dataset = pd.read_pickle(CONFIG.paths.dataset)

    num_images = len(dataset.keys())
    rows = int(num_images**0.5)  # Adjust as needed for desired grid layout
    cols = num_images // rows + (1 if num_images % rows else 0)

    first_key = list(dataset.keys())[0]

    # # print metadata
    print(f"Dataset contains {num_images} images")
    print(f"Pressure field shape: {dataset[first_key]['real_pressure'].shape}")

    # Visualize the dataset
    data_to_plot = []
    for i, f in enumerate(dataset.keys()):
        grid = dataset.attrs["grid"]
        pressure = dataset[f]["real_pressure"].real[:, :, 0]
        data_to_plot.append((pressure, grid[0], grid[1], f))

    fig, axes = plot_scalar_field_grid(
        scalar_field,
        data_to_plot,
        nrows=rows,
        ncols=cols,
        figsize=(20, 20),
        xlabel="x [m]",
        ylabel="y [m]",
        cbar_label="Pressure [Pa]",
    )
    fig.suptitle("Pressure field at different frequencies", fontsize=20)
    plt.show()


if __name__ == "__main__":
    main()
