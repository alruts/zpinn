import os
import shutil

import h5py
import numpy as np
import pandas as pd
import equinox as eqx

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from typing import Callable


def load_comsol_data(path: str, variables: list = ["x", "y", "z", "p.real", "p.imag"]):
    df = pd.read_csv(
        path,
        comment="%",
        sep=",",
        names=variables,
    )
    return df


def get_gt_meshgrid(df: pd.DataFrame) -> np.ndarray:
    return np.meshgrid(np.unique(df["x"]), np.unique(df["y"]), np.unique(df["z"]))


def get_val_meshgrid(df: pd.DataFrame) -> np.ndarray:
    return np.meshgrid(np.unique(df["x"]), np.unique(df["y"]))


def get_pressure(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    p = df["p.real"].values + 1j * df["p.imag"].values
    p = p.reshape(grid[0].shape)
    return p


def get_pvel(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    uz = df["uz.real"].values + 1j * df["uz.imag"].values
    uz = uz.reshape(grid[0].shape)
    return uz


def get_gt_data(file_path: str, variables: list = ["x", "y", "z", "p.real", "p.imag"]):
    df = load_comsol_data(file_path, variables=variables)
    grid = get_gt_meshgrid(df)
    p = get_pressure(df, grid)
    return grid, p


def get_val_data(
    file_path: str,
    variables: list = ["x", "y", "p.real", "p.imag", "uz.real", "uz.imag"],
):
    df = load_comsol_data(file_path, variables=variables)
    grid = get_val_meshgrid(df)
    p = get_pressure(df, grid)
    u = get_pvel(df, grid)
    return grid, p, u


def load_data(filename, frequency):
    with h5py.File(filename, "r") as f:
        # Extract data from the first group (assuming only one group)
        group = f[str(frequency)]
        grid = group["grid"][()]
        pressure = group["pressure"][()]
        impedance = group["impedance"][()]

    return grid, pressure, impedance


def downsample_grid(grid, factor):
    """Downsample a grid.

    Args:
        grid (np.ndarray): grid to downsample
        factor (int): downsampling factor

    Returns:
        np.ndarray: downsampled grid
    """
    return grid[::factor, ::factor]


def create_tmp_dir(path: str):
    """Create a temporary directory.

    Args:
        path (str): path to the directory
    """
    tmp_path = os.path.join(path, "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    return os.path.abspath(tmp_path)


def delete_dir(path: str):
    """Delete a directory.

    Args:
        path (str): path to the directory
    """
    path = os.path.join(path)
    if os.path.exists(path):
        shutil.rmtree(path)


import logging

from omegaconf import DictConfig


def set_param(model, param_name, value):
    """Sets a parameter in a COMSOL model.

    Args:
        model (any): comsol model object
        param_name (str): name of the parameter
        value (any): value of the parameter
    """
    logging.log(logging.INFO, f"Setting {param_name} to {value}")
    model.parameter(param_name, str(value))


def get_all_leaf_nodes(config: DictConfig, path="", skip_nodes=[]):
    """
    Retrieves all leaf nodes from a Hydra configuration (DictConfig) in the format '...grandparent.parent.child'.

    Args:
        config: The Hydra configuration (DictConfig) object.
        path: The current path in the config structure.
        nodes: The list of nodes to skip.

    Returns:
        A list of tuples, where each tuple contains (path, value) for a leaf node.
    """

    leaf_nodes = []

    def _traverse_and_collect_leaf_nodes(config_node, current_path, skip_nodes=[]):
        """Recursively traverses a configuration and collects leaf nodes."""

        for key, value in config_node.items():
            if key in skip_nodes:
                continue

            new_path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, DictConfig):
                _traverse_and_collect_leaf_nodes(value, new_path)
            elif isinstance(value, list):
                pass
            else:
                leaf_nodes.append((new_path, value))

    _traverse_and_collect_leaf_nodes(
        config, path, skip_nodes
    )  # Start the recursion with the initial config
    return leaf_nodes


def set_all_config_params(model: any, cfg: DictConfig, skip_nodes: list[str] = []):
    """Sets all parameters in the config to the COMSOL model.

    Args:
        model (any): comsol model object
        cfg (DictConfig): Hydra configuration object
        nodes (list[str]): list of nodes to use in the configuration
    """
    leaf_nodes = get_all_leaf_nodes(cfg, skip_nodes=skip_nodes)
    for param, value in leaf_nodes:
        set_param(model, param, value)


def load_model(eval_model: str, out_dir: str, model_skeleton: eqx.Module) -> eqx.Module:
    """Loads a model from the file system based on provided parameters.

    Args:
        eval_model (str): Can be "latest" to load the most recent model or
                           a specific filename within the output directory.
        out_dir (str): The output directory where models are saved.
        model_skeleton (eqx.Module): The base model architecture without weights.
        subkey (jnp.ndarray): A JAX subkey for potential randomness during loading.

    Returns:
        eqx.Module: The loaded model with weights.

    Raises:
        FileNotFoundError: If the specified model file is not found.
    """

    if eval_model == "latest":
        # Get all files in the output directory
        files = os.listdir(out_dir)

        # Sort files (assuming filenames have timestamps or versioning)
        files.sort()

        # Choose the latest file
        latest_file = files[-1]
        model_path = os.path.join(out_dir, latest_file)
    else:
        # Use the specified filename
        model_path = os.path.join(out_dir, eval_model)

    try:
        # Load the model weights using Equinox tree deserialization
        model = eqx.tree_deserialise_leaves(model_path, model_skeleton)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model


def visualize(fn: Callable[[jnp.ndarray], jnp.ndarray], args: jnp.ndarray) -> None:
    """Visualizes the computation graph of a function `f` with input batch.

    Args:
        f (callable[[jnp.ndarray], jnp.ndarray]):
            The function for which to visualize the computation graph.
        batch (jnp.ndarray):
            The input batch to be used with `f`.

    Raises:
        RuntimeError:
            If there's an error generating the visualization.
    """

    try:
        # Leverage JAX's XLA capabilities for efficient HLO graph generation
        hlo_graph = jax.xla_computation(fn)(*args).as_hlo_dot_graph()

        # Create temporary DOT file and write HLO representation for visualization
        with open("t2.dot", "w") as fn:
            fn.write(hlo_graph)

        # Generate PNG image from DOT representation (handle potential errors)
        result = os.system("dot -Tpng ./t2.dot -o ./t2.png")
        if result != 0:
            raise RuntimeError(
                "Error generating visualization image (dot command failed)."
            )

        # Read the generated image and display it using Matplotlib
        img = plt.imread("t2.png")
        plt.imshow(img)

        # Gracefully remove temporary files after visualization
        os.remove("t2.dot")
        os.remove("t2.png")

        plt.show()
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error encountered while visualizing: {e}")
    except RuntimeError as e:
        print(f"Visualization failed: {e}")


def transform(data, shift, scale):
    return (data - shift) / scale


def flatten_pytree(pytree):
    """Flattens a pytree."""
    return ravel_pytree(pytree)[0]

def cat_batches(batches):
    """Concatenates a list of batches."""
    result = dict(
        x=jnp.array([]), y=jnp.array([]), z=jnp.array([]), f=jnp.array([])
    )
    for batch in batches:
        for key in batch.keys():
            result[key] = jnp.concatenate([result[key], batch[key]], axis=0)
    return result
