import logging
import os
import sys

import hydra
import mph
import pandas as pd

sys.path.append("zpinn")
from miki import miki

from zpinn.utils import (
    create_tmp_dir,
    delete_dir,
    get_gt_data,
    get_val_data,
    set_all_config_params,
    set_param,
)


@hydra.main(
    config_path="conf", config_name="inf_baffle.yaml", version_base=hydra.__version__
)
def gen_data(cfg):
    # create a temporary directory
    tmp_dir = create_tmp_dir(cfg.paths.data)

    # Load config parameters
    frequencies = cfg.dataset.frequencies
    model_path = cfg.paths.mph
    name = cfg.dataset.name
    thickness = cfg.sample.dimensions.lz
    flow_resistivity = 41000  # Pa.s/m²

    # Initialize the dataset
    df = pd.DataFrame()
    filename = os.path.join(cfg.paths.data, "raw", cfg.dataset.name + ".pkl")

    df.attrs = {
        "frequencies": frequencies,
        "name": name,
    }

    client = mph.start()
    model = client.load(model_path)

    # Set the parameters
    set_all_config_params(
        model,
        cfg,
        nodes=["sample", "source", "grid", "mesh"],
    )

    # Loop over the frequencies and run the simulations
    for frequency in frequencies:
        print(f"Running simulation for {frequency} Hz")

        impedance = miki(
            flow_resistivity, frequency, thickness=thickness, normalized=False
        )

        # Set the parameters
        set_param(model, "frequency", frequency)
        set_param(model, "Z.real", impedance.real)
        set_param(model, "Z.imag", impedance.imag)

        # Build, mesh and solve the model
        model.build()
        model.mesh()
        model.solve()

        # Export the measurement grid
        save_name = f"{name}_{frequency}_Hz"
        save_name = os.path.join(tmp_dir, save_name)
        logging.log(logging.INFO, f"Exporting pressure grid to {save_name}")
        model.export("grid", save_name + "_measurement.txt")
        model.export("surf", save_name + "_surface.txt")

        # Get the grid and pressure from .txt file
        grid, pressure = get_gt_data(save_name + "_measurement.txt")
        ref_grid, ref_pressure, ref_uz = get_val_data(save_name + "_surface.txt")

        ref = {
            "real_pressure": ref_pressure.real,
            "imag_pressure": ref_pressure.imag,
            "real_velocity": ref_uz.real,
            "imag_velocity": ref_uz.imag,
        }

        # ground truth
        gt = {
            "real_pressure": pressure.real,
            "imag_pressure": pressure.imag,
            "real_impedance": impedance.real,
            "imag_impedance": impedance.imag,
            "ref": ref,
        }

        # Add the data to the dataframe
        df[frequency] = gt

        logging.log(logging.INFO, f"Simulation for {frequency} Hz completed")

    # add metadata to the dataframe
    df.attrs["grid"] = grid
    df.attrs["ref_grid"] = ref_grid
    df.attrs["thickness"] = thickness
    df.attrs["flow_resistivity"] = flow_resistivity
    df.attrs["name"] = name
    df.attrs["frequencies"] = frequencies

    df.to_pickle(filename)
    logging.log(logging.INFO, f"All simulations completed, dataset saved to {filename}")

    delete_dir(tmp_dir)
    logging.log(logging.INFO, f"Temporary directory {tmp_dir} deleted")


if __name__ == "__main__":
    gen_data()
