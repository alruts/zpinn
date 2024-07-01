import logging
import pickle
import sys

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import torch
from jax.tree_util import tree_map
from torch.utils import data as torchdata
from torch.utils.data import DataLoader, Dataset

logging.log(logging.INFO, "Setting torch seed to 0")
torch.manual_seed(0)

sys.path.append("src")
from zpinn.utils import transform


def numpy_collate(batch):
    return tree_map(np.asarray, torchdata.default_collate(batch))


class PressureDataset(Dataset):
    """Dataset class for 4D data."""

    def __init__(self, path, snr=None, non_dim=True):
        self.path = path
        self.data = self._load_data()

        self.grid = self.data.attrs["grid"]
        self.transforms = self.data.attrs["transforms"]

        # Get linspaces of x, y, z
        self._x, self._y, self._z, self._f = self.grid
        self._x = self._x[0, :, 0]
        self._y = self._y[:, 0, 0]
        self._z = self._z[0, 0, :]

        # infer sizes from first example
        self.n_x = len(self._x)
        self.n_y = len(self._y)
        self.n_z = len(self._z)
        self.n_f = len(self._f)

        self.snr = snr
        if self.snr is not None:
            self._add_noise(snr=self.snr)
        if non_dim:
            self._non_dim()
        else:
            self.transforms = {
                "x0": 0.0,
                "y0": 0.0,
                "z0": 0.0,
                "f0": 0.0,
                "a0": 0.0,
                "b0": 0.0,
                "xc": 1.0,
                "yc": 1.0,
                "zc": 1.0,
                "fc": 1.0,
                "ac": 1.0,
                "bc": 1.0,
            }

    def __len__(self):
        return self.n_f * self.n_x * self.n_y * self.n_z  # Total number of voxels

    def __getitem__(self, idx):
        frame_idx = idx // (self.n_x * self.n_y * self.n_z)  # Get frame index
        pixel_idx = idx % (self.n_x * self.n_y)  # Get x,y pixel within the z-slice
        z_idx = idx // (self.n_x * self.n_y) % self.n_z  # Get z index

        f = self._f[frame_idx]  # convert frame index to frequency
        data = self.data[f]  # get the data at the frequency

        x_idx, y_idx = np.unravel_index(
            pixel_idx, (self.n_x, self.n_y)
        )  # Unravel pixel index for x and y

        # Get the pressure at the pixel
        pressure_re = data["real_pressure"][x_idx, y_idx, z_idx]
        pressure_im = data["imag_pressure"][x_idx, y_idx, z_idx]

        x, y, z = (
            self._x[x_idx],
            self._y[y_idx],
            self._z[z_idx],
        )  # Convert to (x, y, z) coordinates

        coords = {
            "x": transform(x, self.transforms["x0"], self.transforms["xc"]),
            "y": transform(y, self.transforms["y0"], self.transforms["yc"]),
            "z": transform(z, self.transforms["z0"], self.transforms["zc"]),
            "f": transform(f, self.transforms["f0"], self.transforms["fc"]),
        }
        gt = {
            "real_pressure": transform(
                pressure_re, self.transforms["a0"], self.transforms["ac"]
            ),
            "imag_pressure": transform(
                pressure_im, self.transforms["b0"], self.transforms["bc"]
            ),
            "real_impedance": data["real_impedance"],
            "imag_impedance": data["imag_impedance"],
        }  # ground truth

        return (coords, gt)

    def _load_data(self):
        """Loads the dataset from the path."""
        with open(self.path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def _non_dim(self):
        """Non-dimensionalizes the data."""
        get_scale = lambda x: (np.max(x) - np.min(x))
        get_shift = lambda x: np.min(x)
        get_max = lambda x: np.max(x)

        # Non-dimensionalize the coordinates
        self.transforms["x0"] = get_shift(self._x)
        self.transforms["y0"] = get_shift(self._y)
        self.transforms["z0"] = 0

        self.transforms["xc"] = get_scale(self._x)
        self.transforms["yc"] = get_scale(self._y)
        self.transforms["zc"] = get_max(self._z)

        # Non-dimensionalize the frequencies
        self.transforms["f0"] = 0
        self.transforms["fc"] = get_max(self._f)

        # Non-dimensionalize the pressure data
        self.transforms["a0"] = get_shift(self.data[self._f[0]]["real_pressure"])
        self.transforms["ac"] = get_scale(self.data[self._f[0]]["real_pressure"])
        self.transforms["b0"] = get_shift(self.data[self._f[0]]["imag_pressure"])
        self.transforms["bc"] = get_scale(self.data[self._f[0]]["imag_pressure"])

        logging.info("Non-dimensionalizing the data")
        logging.info(f"x0: {self.transforms['x0']}, xc: {self.transforms['xc']}")
        logging.info(f"y0: {self.transforms['y0']}, yc: {self.transforms['yc']}")
        logging.info(f"z0: {self.transforms['z0']}, zc: {self.transforms['zc']}")
        logging.info(f"f0: {self.transforms['f0']}, fc: {self.transforms['fc']}")
        logging.info(f"a0: {self.transforms['a0']}, ac: {self.transforms['ac']}")
        logging.info(f"b0: {self.transforms['b0']}, bc: {self.transforms['bc']}")

    def _add_noise(self, snr):
        """Adds noise to the dataset."""
        # convert snr to linear scale
        snr_linear = 10 ** (snr / 10)

        # get all data points
        data = []
        for f in self._f:
            data.append(self.data[f]["real_pressure"])
            data.append(self.data[f]["imag_pressure"])
        data = np.stack(data, axis=-1)

        # calculate signal power
        signal_power = np.mean(data**2)

        # calculate noise power
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, 1, data.shape) * np.sqrt(noise_power)

        print(f"Adding noise with SNR: {snr} dB")
        print(f"Signal power: {signal_power}, Noise power: {noise_power}")
        print(f"Percentage of noise: {noise_power / signal_power * 100}%")

        # add noise to the data
        for i, f in enumerate(self._f):
            self.data[f]["real_pressure"] += noise[..., 2 * i]
            self.data[f]["imag_pressure"] += noise[..., 2 * i + 1]

    def restrict_to(self, x=None, y=None, z=None, f=None):
        """Restricts the dataset to a specific x, y, z, f."""

        def get_lohi(arr, lb, ub):
            idxs = np.where((arr <= ub) & (arr >= lb))
            return np.amin(idxs), np.amax(idxs) + 1

        def filter_data(arr, bounds):
            if bounds is not None:
                lo, hi = get_lohi(arr, *bounds)
                print(f"Filtering data from {lo} to {hi}")
                return arr[lo:hi]
            else:
                return arr

        # get the slice indices before filtering
        slice_x = slice(*get_lohi(self._x, *x)) if x is not None else slice(None)
        slice_y = slice(*get_lohi(self._y, *y)) if y is not None else slice(None)
        slice_z = slice(*get_lohi(self._z, *z)) if z is not None else slice(None)

        # filter the dataset
        self._f = f if f is not None else self._f
        self._x = filter_data(self._x, x)
        self._y = filter_data(self._y, y)
        self._z = filter_data(self._z, z)

        # update the number of points
        self.n_x = len(self._x)
        self.n_y = len(self._y)
        self.n_z = len(self._z)
        self.n_f = len(self._f)

        # filter the pressure data
        for f in self._f:
            for key in ["real_pressure", "imag_pressure"]:
                self.data[f][key] = self.data[f][key][slice_x, slice_y, slice_z]

        # filter the reference
        ref_x, ref_y = self.data.attrs["ref_grid"]
        _lin_x, _lin_y = ref_x[0, :], ref_y[:, 0]

        # filter the reference data
        for f in self._f:
            for key, val in self.data[f]["ref"].items():
                self.data[f]["ref"][key] = self.data[f]["ref"][key][
                    slice(*get_lohi(_lin_x, *x)) if x is not None else slice(None),
                    slice(*get_lohi(_lin_y, *y)) if y is not None else slice(None),
                ]

        # filter the reference grid
        self.data.attrs["ref_grid"] = (
            ref_x[
                slice(*get_lohi(_lin_y, *y)) if y is not None else slice(None),
                slice(*get_lohi(_lin_x, *x)) if x is not None else slice(None),
            ],
            ref_y[
                slice(*get_lohi(_lin_y, *y)) if y is not None else slice(None),
                slice(*get_lohi(_lin_x, *x)) if x is not None else slice(None),
            ],
        )
        self._non_dim()

    def get_reference(self, f):
        """Returns the reference data for a specific frequency."""
        assert f in self.data, f"Frequency {f} not in dataset"

        x, y = self.data.attrs["ref_grid"]
        x_lin = np.linspace(x.min(), x.max(), len(x))
        y_lin = np.linspace(y.min(), y.max(), len(y))

        coords = dict(
            x=x_lin,
            y=y_lin,
            z=self.data.attrs["thickness"],
            f=f,
        )

        gt = self.data[f]["ref"]

        assert gt["real_pressure"].shape == (
            len(x),
            len(y),
        ), f"Reference grid shape mismatch: {gt['real_pressure'].shape}, expected: {len(x), len(y)}"

        gt["real_impedance"] = self.data[f]["real_impedance"]
        gt["imag_impedance"] = self.data[f]["imag_impedance"]
        gt["f"] = f

        return (coords, gt)

    def get_dataloader(self, batch_size=32, shuffle=False):
        """Returns a dataloader for the dataset."""
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate
        )


class BaseSampler(Dataset):
    """Base class for samplers."""

    def __init__(self, batch_size, rng_key=jrandom.PRNGKey(0)):
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = jrandom.split(self.key)
        batch = self.gen_data(subkey)
        return batch

    def gen_data(self, key):
        raise NotImplementedError("Subclasses should implement this!")


class DomainSampler(BaseSampler):
    """Generator for collocation points.

    Args:
        num_points (int):
            Number of points to generate.
        limits (dict):
            Dictionary containing the limits of the points in each axis (min, max)
        transforms (dict):
            Dictionary containing the transforms for the points (shift, scale)
        distributions (dict):
            Dictionary containing the distributions for the points 'uniform' or 'grid'.
    """

    def __init__(self, batch_size, limits, transforms, distributions, rng_key=0):
        super().__init__(batch_size, rng_key=jrandom.PRNGKey(rng_key))

        self.limits = limits
        self.transforms = transforms
        self.distributions = distributions

    def _sample_axis(self, key, axis_min, axis_max, distribution="uniform"):
        if distribution == "uniform":
            return jrandom.uniform(
                key, (self.batch_size,), minval=axis_min, maxval=axis_max
            )
        else:
            raise ValueError(f"Invalid distribution: {distribution}")

    def gen_data(self, key):
        key, *subkeys = jrandom.split(key, num=len(self.limits) + 1)
        data = {}

        for axis, (axis_min, axis_max) in self.limits.items():
            data[axis] = self._sample_axis(
                subkeys.pop(0), axis_min, axis_max, self.distributions[axis]
            )
            t = {k: v for k, v in self.transforms.items() if k.startswith(axis)}
            data[axis] = transform(data[axis], *t.values())

        return data


class BoundarySampler(BaseSampler):
    """Generator for boundary points.

    Args:
        num_points (int):
            Number of points to generate.
        limits (dict):
            Dictionary containing the limits of the points in each axis (min, max)
        transforms (dict):
            Dictionary containing the transforms for the points
        distributions (dict):
            Dictionary containing the distributions for the points 'uniform' or 'grid'.
    """

    def __init__(self, batch_size, limits, transforms, distributions, rng_key=0):
        super().__init__(batch_size, rng_key=jrandom.PRNGKey(rng_key))

        self.limits = limits
        self.transforms = transforms
        self.distributions = distributions
        self.sqrt_batch_size = int(np.sqrt(self.batch_size))

    def _sample_axis(self, key, axis_min, axis_max, distribution="uniform"):
        if distribution == "uniform":
            return jrandom.uniform(
                key, (self.batch_size,), minval=axis_min, maxval=axis_max
            )
        if distribution == "grid":
            return jnp.linspace(axis_min, axis_max, self.sqrt_batch_size)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")

    def gen_data(self, key):
        key, *subkeys = jrandom.split(key, num=len(self.limits) + 1)
        data = {}

        for axis, (axis_min, axis_max) in self.limits.items():
            data[axis] = self._sample_axis(
                subkeys.pop(0), axis_min, axis_max, self.distributions[axis]
            )
            t = {k: v for k, v in self.transforms.items() if k.startswith(axis)}
            data[axis] = transform(data[axis], *t.values())

        # check if x and y are grid
        if self.distributions["x"] == "grid" and self.distributions["y"] == "grid":
            assert (
                self.batch_size == self.sqrt_batch_size**2
            ), "Batch size must be a square number for grid distribution"
            data["x"], data["y"] = jnp.meshgrid(data["x"], data["y"])
            data["x"] = data["x"].flatten().ravel()
            data["y"] = data["y"].flatten().ravel()

        return data
