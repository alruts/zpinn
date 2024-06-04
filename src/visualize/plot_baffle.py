import sys
import os

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
    raise ValueError("Please provide a path to the config")

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
            draw_height_lines=False,
            color="b",
            label=f"Measurement aperture {idx} @ {z*1000:.0f} mm",
        )

    # draw the sample surface
    ax = draw_rectangle(
        ax,
        1,
        1,
        center=(
            config.sample.center.x,
            config.sample.center.y,
            config.sample.dimensions.lz / 2,
        ),
        color="grey",
        alpha=0.5,
        label="Impedance surface",
    )

    ax = draw_source(
        ax,
        config.source.center.x,
        config.source.center.y,
        config.source.center.z,
        color="r",
    )

    if hasattr(config, "room"):
        lx, ly, lz = (
            config.room.dimensions.lx,
            config.room.dimensions.ly,
            config.room.dimensions.lz,
        )
        c = config.room.center.x, config.room.center.y, lz / 2
        ax = draw_shoebox(
            ax,
            lx,
            ly,
            lz,
            center=c,
            label="Room",
        )
        # ax = mark_side_lengths(ax, **config.room.dimensions, center=c)

    else:
        draw_rectangle(
            ax,
            1.5,
            1.5,
            color="purple",
            alpha=0.1,
            label="Infinite Baffle",
            linestyle="--",
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
    # ax.axis("off")

    # modify the view angle
    ax.view_init(elev=12, azim=155)

    # set legend
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.1, 1),
        fontsize="small",
    )

    plt.tight_layout()

    # export as as pgf and show
    name = os.path.basename(args.dataset_config).split(".")[0]
    plt.savefig(f"{name}.pgf", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
