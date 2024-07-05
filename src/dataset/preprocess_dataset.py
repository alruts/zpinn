import logging
import os

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import argparse

logging.basicConfig(level=logging.INFO)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default="src\dataset\configs\inf_baffle.yaml",
    help="path to the config file",
)
args = parser.parse_args()

# Load the config
CONFIG = OmegaConf.load(os.path.join(args.config_path))

# Helper function
downsample_fn = lambda data, factor: data[::factor, ::factor, :]


def preprocess(config=CONFIG):
    raw_path = os.path.join(
        config.paths.data,
        "raw",
        f"{config.dataset.name}.pkl",
    )
    processed_path = os.path.join(
        config.paths.data,
        "processed",
        f"{config.dataset.name}.pkl",
    )

    # Load the raw data
    raw_df = pd.read_pickle(raw_path)

    # Initialize processed DataFrame
    processed_df = pd.DataFrame()

    # Get frequencies
    f = np.array(config.dataset.frequencies, dtype=np.float32)

    # fetch and downsample the grid
    x, y, z = raw_df.attrs["grid"]
    x, y, z = [downsample_fn(arr, config.downsampling) for arr in (x, y, z)]

    # initialize transforms
    logging.info("Using non-dimensionalization")
    transforms = {
        "x0": 0,
        "xc": 1,
        "y0": 0,
        "yc": 1,
        "z0": 0,
        "zc": 1,
        "f0": 0,
        "fc": 1,
        "a0": 0,
        "ac": 1,
        "b0": 0,
        "bc": 1,
    }

    # loop over the frequencies and save the ground truth values
    for idx, frequency in enumerate(config.dataset.frequencies):
        # Load the data
        data = raw_df[frequency]

        ref = data.ref
        ref["real_velocity"] = -ref["real_velocity"]
        ref["imag_velocity"] = -ref["imag_velocity"]

        # save ground truth and transforms to the processed dataframe
        gt = {
            "real_pressure": downsample_fn(data["real_pressure"], config.downsampling),
            "imag_pressure": downsample_fn(data["imag_pressure"], config.downsampling),
            "real_impedance": np.array(data["real_impedance"]),
            "imag_impedance": np.array(data["imag_impedance"]),
            "ref": data.ref,
        }

        # save gt and transforms to the processed dataframe
        processed_df[f[idx]] = gt

    # save attrs
    processed_df.attrs = {
        "name": config.dataset.name,
        "downsample_factor": config.downsampling,
        "transforms": transforms,
        "grid": (x, y, z, f),
        "ref_grid": raw_df.attrs["ref_grid"],
        "thickness": raw_df.attrs["thickness"],
        "flow_resistivity": raw_df.attrs["flow_resistivity"],
    }

    # Save the DataFrame
    logging.info(f"Saving the dataframe to {processed_path}")
    processed_df.to_pickle(processed_path)

    # log output shape
    logging.info(f"Output shape: {processed_df[frequency]['real_pressure'].shape}")

    print("Transforms: ", transforms)


if __name__ == "__main__":
    preprocess()
