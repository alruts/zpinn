import logging
import os
import sys

import hydra
import pandas as pd
import dill
import numpy as np

from ..zpinn.constants import _c0, _rho0
logging.basicConfig(level=logging.INFO)


def nondim(x, x0=None, xc=None):
    """Normalize the data"""
    if x0:
        x -= x0
    if xc:
        x /= xc
    return x


def downsample(data, factor):
    """Downsample the data"""
    return data[::factor, ::factor, :]


@hydra.main(
    config_path="../../conf", config_name="inf_baffle", version_base=hydra.__version__
)
def preprocess(cfg):
    raw_path = os.path.join(
        cfg.dataset.out_dir,
        "raw",
        f"{cfg.dataset.name}.pkl",
    )
    processed_path = os.path.join(
        cfg.dataset.out_dir,
        "processed",
        f"{cfg.dataset.name}.pkl",
    )

    # Load the raw data
    raw_df = pd.read_pickle(raw_path)

    # Initialize processed DataFrame
    processed_df = pd.DataFrame()

    downsample_factor = 19

    # concatenate all pressure values to compute mean and max
    _p_re = np.concatenate(
        [raw_df[frequency].pressure.real for frequency in cfg.dataset.frequencies],
        axis=-1,
    )
    _p_im = np.concatenate(
        [raw_df[frequency].pressure.imag for frequency in cfg.dataset.frequencies],
        axis=-1,
    )

    # Inference grid from the first frequency
    x, y, z = raw_df[cfg.dataset.frequencies[0]].grid

    # Get frequencies
    f = np.array(cfg.dataset.frequencies, dtype=np.float32)

    # find constants for transforming the data
    p_re0 = np.mean(_p_re)
    p_im0 = np.mean(_p_im)
    x0 = np.mean([cfg.postprocessing.x.min, cfg.postprocessing.x.max])
    y0 = np.mean([cfg.postprocessing.y.min, cfg.postprocessing.y.max])
    z0 = np.mean([cfg.postprocessing.z.min, cfg.postprocessing.z.max])
    f0 = np.mean([cfg.postprocessing.f.min, cfg.postprocessing.f.max])

    # find the max difference frvom the mean to norm in [-1, 1]
    p_rec = np.max(np.abs(_p_re - p_re0))
    p_imc = np.max(np.abs(_p_im - p_im0))
    xc = np.max(np.abs(x - x0))
    yc = np.max(np.abs(y - y0))
    zc = np.max(np.abs(z - z0))
    fc = np.max(np.abs(f - f0))

    # non dimensionalize the 4D grid
    f = nondim(f, f0, fc)
    x = nondim(x, x0, xc)
    y = nondim(y, y0, yc)

    if not cfg.postprocessing.z.nondim:
        z0, zc = 0.0, 1.0
        z = nondim(z, z0, zc)
    else:
        z = nondim(z, z0, zc)

    # downsample x, y
    x, y, z = [downsample(arr, downsample_factor) for arr in (x, y, z)]

    # save the transforms
    transforms = {
        "x": (x0, xc),
        "y": (y0, yc),
        "z": (z0, zc),
        "f": (f0, fc),
        "real_pressure": (p_re0, p_rec),
        "imag_pressure": (p_im0, p_imc),
    }

    # loop over the frequencies and save the ground truth values
    for ii, frequency in enumerate(cfg.dataset.frequencies):
        # Load the data
        data = raw_df[frequency]

        # Unpack the data
        p_re = data.pressure.real
        p_im = data.pressure.imag
        Zn = data.impedance

        # transform the data
        p_re = nondim(p_re, p_re0, p_rec)
        p_im = nondim(p_im, p_im0, p_imc)

        # downsample the data
        p_re = downsample(p_re, downsample_factor)
        p_im = downsample(p_im, downsample_factor)

        # save ground truth and transforms to the processed dataframe
        gt = {
            "real_pressure": p_re,
            "imag_pressure": p_im,
            "real_impedance": Zn.real,
            "imag_impedance": Zn.imag,
        }

        # save gt and transforms to the processed dataframe
        processed_df[f[ii]] = gt

    logging.info(f"Saving the dataframe to {processed_path}")

    # save attrs
    processed_df.attrs = {
        "name": cfg.dataset.name,
        "downsample_factor": downsample_factor,
        "transforms": transforms,
        "grid": (x, y, z, f),
    }

    # Save the DataFrame
    processed_df.to_pickle(processed_path)

    # print f bar
    logging.log(logging.INFO, f"f bar: {f}")

    # log output shapes
    logging.log(logging.INFO, f"p_re shape: {p_re.shape}")
    logging.log(logging.INFO, f"p_im shape: {p_im.shape}")
    logging.log(logging.INFO, f"x shape: {x.shape}")
    logging.log(logging.INFO, f"y shape: {y.shape}")
    logging.log(logging.INFO, f"z shape: {z.shape}")

    # log mean and variance of the processed data
    logging.log(
        logging.INFO, f"Mean real pressure: {np.mean(p_re)}, var: {np.var(p_re)}"
    )
    logging.log(
        logging.INFO, f"Mean imag pressure: {np.mean(p_im)}, var: {np.var(p_im)}"
    )
    logging.log(logging.INFO, f"Mean x: {np.mean(x)}, var: {np.var(x)}")
    logging.log(logging.INFO, f"Mean y: {np.mean(y)}, var: {np.var(y)}")
    logging.log(logging.INFO, f"Mean z: {np.mean(z)}, var: {np.var(z)}")
    logging.log(logging.INFO, f"Mean f: {np.mean(f)}, var: {np.var(f)}")

    # log min and max of the processed data
    logging.log(logging.INFO, f"Min real pressure: {np.min(p_re)}, max: {np.max(p_re)}")
    logging.log(logging.INFO, f"Min imag pressure: {np.min(p_im)}, max: {np.max(p_im)}")
    logging.log(logging.INFO, f"Min x: {np.min(x)}, max: {np.max(x)}")
    logging.log(logging.INFO, f"Min y: {np.min(y)}, max: {np.max(y)}")
    logging.log(logging.INFO, f"Min z: {np.min(z)}, max: {np.max(z)}")
    logging.log(logging.INFO, f"Min f: {np.min(f)}, max: {np.max(f)}")

    print(processed_df.head())


if __name__ == "__main__":
    preprocess()
