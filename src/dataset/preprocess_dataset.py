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

    # concatenate all pressure values to compute mean and max
    _p_re = np.concatenate(
        [
            raw_df[frequency]["real_pressure"]
            for frequency in config.dataset.frequencies
        ],
        axis=-1,
    )
    _p_im = np.concatenate(
        [
            raw_df[frequency]["imag_pressure"]
            for frequency in config.dataset.frequencies
        ],
        axis=-1,
    )
    x, y, z = raw_df.attrs["grid"]

    # Get frequencies
    f = np.array(config.dataset.frequencies, dtype=np.float32)

    # compute shift values
    a0 = np.mean(_p_re) if config.nondim.p.shift else 0
    b0 = np.mean(_p_im) if config.nondim.p.shift else 0
    x0 = np.mean(x) if config.nondim.x.shift else 0
    y0 = np.mean(y) if config.nondim.y.shift else 0
    z0 = np.mean(z) if config.nondim.z.shift else 0
    f0 = np.mean(f) if config.nondim.f.shift else 0

    # compute the scale values
    ac = np.max(abs(_p_re - a0)) if config.nondim.p.scale else 1
    bc = np.max(abs(_p_im - b0)) if config.nondim.p.scale else 1
    xc = np.max(abs(x - x0)) if config.nondim.x.scale else 1
    yc = np.max(abs(y - y0)) if config.nondim.y.scale else 1
    zc = np.max(abs(z - z0)) if config.nondim.z.scale else 1
    fc = np.max(abs(f - f0)) if config.nondim.f.scale else 1

    # downsample spatial coordinates
    x, y, z = [
        downsample_fn(arr, config.downsampling) for arr in (x, y, z)
    ]

    logging.info("Using non-dimensionalization")
    transforms = {
        "x0": x0,
        "xc": xc,
        "y0": y0,
        "yc": yc,
        "z0": z0,
        "zc": zc,
        "f0": f0,
        "fc": fc,
        "a0": a0,
        "ac": ac,
        "b0": b0,
        "bc": bc,
    }

    # loop over the frequencies and save the ground truth values
    for idx, frequency in enumerate(config.dataset.frequencies):
        # Load the data
        data = raw_df[frequency]

        # save ground truth and transforms to the processed dataframe
        gt = {
            "real_pressure": downsample_fn(
                data["real_pressure"], config.downsampling
            ),
            "imag_pressure": downsample_fn(
                data["imag_pressure"], config.downsampling
            ),
            "real_impedance": data["real_impedance"],
            "imag_impedance": data["imag_impedance"],
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
