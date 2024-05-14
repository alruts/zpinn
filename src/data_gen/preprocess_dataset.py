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
    default="src\data_gen\conf\inf_baffle.yaml",
    help="path to the config file",
)
parser.add_argument(
    "-d", "--downsample_factor", type=int, default=1, help="downsample factor"
)
args = parser.parse_args()

# Load the config
CONFIG = OmegaConf.load(os.path.join(args.config_path))

# Helper function
downsample_fn = lambda data, factor: data[::factor, ::factor, :]

def preprocess(cfg=CONFIG):
    raw_path = os.path.join(
        cfg.paths.data,
        "raw",
        f"{cfg.dataset.name}.pkl",
    )
    processed_path = os.path.join(
        cfg.paths.data,
        "processed",
        f"{cfg.dataset.name}.pkl",
    )

    # Load the raw data
    raw_df = pd.read_pickle(raw_path)

    # Initialize processed DataFrame
    processed_df = pd.DataFrame()

    # concatenate all pressure values to compute mean and max
    _p_re = np.concatenate(
        [raw_df[frequency]["real_pressure"] for frequency in cfg.dataset.frequencies],
        axis=-1,
    )
    _p_im = np.concatenate(
        [raw_df[frequency]["imag_pressure"] for frequency in cfg.dataset.frequencies],
        axis=-1,
    )

    # Inference grid from the first frequency
    x, y, z = raw_df.attrs["grid"]

    # Get frequencies
    f = np.array(cfg.dataset.frequencies, dtype=np.float32)

    # find constants for transforming the data
    a0 = np.mean(_p_re)
    b0 = np.mean(_p_im)
    x0 = np.mean([cfg.postprocessing.x.min, cfg.postprocessing.x.max])
    y0 = np.mean([cfg.postprocessing.y.min, cfg.postprocessing.y.max])
    z0 = np.mean([cfg.postprocessing.z.min, cfg.postprocessing.z.max])
    f0 = np.mean([cfg.postprocessing.f.min, cfg.postprocessing.f.max])

    # find the max difference frvom the mean to norm in [-1, 1]
    ac = np.std(np.abs(_p_re - a0))
    bc = np.std(np.abs(_p_im - b0))
    xc = np.max(np.abs(x - x0))
    yc = np.max(np.abs(y - y0))
    zc = np.max(np.abs(z - z0))
    fc = np.max(np.abs(f - f0))

    # downsample spatial coordinates
    x, y, z = [downsample_fn(arr, args.downsample_factor) for arr in (x, y, z)]

    # save the transforms
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
        "b0": bc,
        "bc": b0,
    }

    # loop over the frequencies and save the ground truth values
    for idx, frequency in enumerate(cfg.dataset.frequencies):
        # Load the data
        data = raw_df[frequency]

        # Unpack the data
        p_re = data["real_pressure"]
        p_im = data["imag_pressure"]

        # downsample the data
        p_re = downsample_fn(p_re, args.downsample_factor)
        p_im = downsample_fn(p_im, args.downsample_factor)

        # save ground truth and transforms to the processed dataframe
        gt = {
            "real_pressure": p_re,
            "imag_pressure": p_im,
            "real_impedance": data["real_impedance"],
            "imag_impedance": data["imag_impedance"],
            "ref": data.ref,
        }

        # save gt and transforms to the processed dataframe
        processed_df[f[idx]] = gt

    # save attrs
    processed_df.attrs = {
        "name": cfg.dataset.name,
        "downsample_factor": args.downsample_factor,
        "transforms": transforms,
        "grid": (x, y, z, f),
    }

    # Save the DataFrame
    logging.info(f"Saving the dataframe to {processed_path}")
    processed_df.to_pickle(processed_path)

    # log output shape
    logging.info(f"Output shape: {p_re.shape}")

    print("Transforms: ", transforms)

if __name__ == "__main__":
    preprocess()
