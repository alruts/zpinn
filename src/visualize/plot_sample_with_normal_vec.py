import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from omegaconf import OmegaConf

sys.path.append("src")
from zpinn.plot.rooms import (
    draw_rectangle,
    draw_shoebox,
    mark_side_lengths,
    draw_source,
)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
    type=str,
    default=None,
    help="path to the dataset config file",
)
args = parser.parse_args()

if None in [args.dataset_config]:
    raise ValueError("Please provide a path to the config file")

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
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(4, 3))

    # draw the sample
    ax = draw_shoebox(ax, **sample_dimensions, center=sample_center)

    # draw the sample surface
    ax = draw_rectangle(
        ax,
        1,
        1,
        center=(
            config.sample.center.x,
            config.sample.center.y,
            config.sample.dimensions.lz,
        ),
        color="grey",
        alpha=0.5,
        label="Impedance surface",
    )

    ax = mark_side_lengths(ax, **sample_dimensions, center=sample_center)

    # draw the normal vector
    start = (0, 0, sample_dimensions["lz"])
    end = (0, 0, sample_dimensions["lz"] + 0.5)

    # draw source
    ax = draw_source(
        ax,
        config.source.center.x,
        config.source.center.y,
        config.source.center.z,
        color="red",
    )

    # draw arrow
    a = Arrow3D(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="black",
        zorder=10,
    )
    ax.add_artist(a)

    # mark the arrow head
    ax.text(end[0]+0.1, end[1]+0.1, end[2]-0.14, "$\mathbf{n}$", color="black", fontsize=12)

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_zlabel("$z$ (m)")
    ax.grid(True)

    # change grid style
    ax.xaxis._axinfo["grid"].update(color="lightgrey", linestyle="-", linewidth=0.5)
    ax.yaxis._axinfo["grid"].update(color="lightgrey", linestyle="-", linewidth=0.5)
    ax.zaxis._axinfo["grid"].update(color="lightgrey", linestyle="-", linewidth=0.5)
    ax.zaxis._axinfo["juggled"] = (1, 2, 2)


    # legend
    ax.legend(loc='center left', bbox_to_anchor=(0.6, 0.6), frameon=False)

    # modify the view angle
    ax.view_init(elev=19, azim=135)

    # set z limit
    ax.set_zlim(0, config.source.center.z + 0.2)

    # set aspect ratio
    # ax.set_box_aspect([1, 1, 0.1])

    # custom ticks
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_zticks([0, 0.05])

    # remove axis
    ax.axis("off")

    # remove whitespace under the plot
    plt.tight_layout()

    plt.savefig("sample_sketch.pgf")
    plt.show()


if __name__ == "__main__":
    main()
