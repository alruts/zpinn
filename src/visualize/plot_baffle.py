import sys

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse
from numpy import linspace

sys.path.append("src")

from zpinn.plot.rooms import (
    draw_shoebox,
    draw_receiver,
    draw_rectangle,
    draw_source,
    mark_side_lengths,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
    type=str,
    default=None,
    help="path to the experiment config file",
)
args = parser.parse_args()

if args.dataset_config is None:
    raise ValueError("Please provide a path to the config files")

CONFIG = OmegaConf.load(args.dataset_config)

def main(config=CONFIG):

    # sample dimensions
    sample_dimensions = {
        "lx": config.sample.dimensions.lx,
        "ly": config.sample.dimensions.ly,
        "lz": config.sample.dimensions.lz,
    }
    sample_center = (
        config.sample.center.x,
        config.sample.center.y,
        sample_dimensions["lz"] / 2,
    )
    # initialize 3D plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # draw the sample
    ax = draw_shoebox(ax, **sample_dimensions, center=sample_center)

    dz = config.grid.dz
    z_start = config.sample.dimensions.lz + config.grid.delta_height
    z_stop = z_start + config.grid.domain_height

    step = int((z_stop - z_start) / dz)

    # draw the receiver aperture
    for idx, z in enumerate(linspace(z_start, z_stop, step + 1, endpoint=True)):
        ax = draw_rectangle(
            ax,
            config.grid.domain_side,
            config.grid.domain_side,
            center=(
                config.sample.center.x,
                config.sample.center.y,
                z,
            ),
            draw_height_lines=True,
            color="b",
            label=f"Measurement aperture {idx} @ {z:.4f} m",
        )

    # draw the sample surface
    ax = draw_rectangle(
        ax,
        1,
        1,
        center=(config.sample.center.x, config.sample.center.y, config.sample.dimensions.lz),
        color="grey",
        alpha=0.5,
        label="Impedance surface",
    )

    ax = draw_source(
        ax, config.source.center.x, config.source.center.y, config.source.center.z, color="r"
    )

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_zlabel("$z$ (m)")
    ax.axis("equal")
    ax.grid(True)

    # change grid style
    ax.xaxis._axinfo["grid"].update(color="lightgrey", linestyle="-", linewidth=0.5)
    ax.yaxis._axinfo["grid"].update(color="lightgrey", linestyle="-", linewidth=0.5)
    ax.zaxis._axinfo["grid"].update(color="lightgrey", linestyle="-", linewidth=0.5)
    ax.zaxis._axinfo["juggled"] = (1, 2, 2)

    # remove the z-axis
    ax.axis("off")

    # modify the view angle
    ax.view_init(elev=19, azim=150)

    # set legend
    plt.tight_layout()

    # export as as pgf and show
    ax.legend(loc="upper right", fontsize="small")
    plt.savefig("room_setup.pgf", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
