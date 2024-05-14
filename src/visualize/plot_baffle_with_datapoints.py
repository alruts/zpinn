import argparse
import sys

import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.append("src")
from zpinn.plot.rooms import draw_rectangle, draw_shoebox
from zpinn.get_loaders import get_loaders

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_config",
    type=str,
    default=None,
    help="path to the experiment config file",
)
parser.add_argument(
    "--dataset_config",
    type=str,
    default=None,
    help="path to the experiment config file",
)
args = parser.parse_args()

if None in [args.experiment_config, args.dataset_config]:
    raise ValueError("Please provide a path to the config files")

CONFIG = OmegaConf.load(args.experiment_config)
CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load(args.dataset_config))

# use unit tranforms
transforms = dict(
    x0=0.0,
    y0=0.0,
    z0=0.0,
    f0=0.0,
    a0=0.0,
    b0=0.0,
    xc=1.0,
    yc=1.0,
    zc=1.0,
    fc=1.0,
    ac=1.0,
    bc=1.0,
)
data_loader, dom_loader, bnd_loader, transforms = get_loaders(
    CONFIG, custom_transforms=transforms
)

def main(cfg=CONFIG):

    # sample dimensions
    sample_dimensions = {
        "lx": cfg.sample.dimensions.lx,
        "ly": cfg.sample.dimensions.ly,
        "lz": cfg.sample.dimensions.lz,
    }
    sample_center = (
        cfg.sample.center.x,
        cfg.sample.center.y,
        sample_dimensions["lz"] / 2,
    )
    # initialize 3D plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # draw the sample
    ax = draw_shoebox(ax, **sample_dimensions, center=sample_center)

    dz = cfg.grid.dz
    z_start = cfg.sample.dimensions.lz + cfg.grid.delta_height
    z_stop = z_start + cfg.grid.domain_height

    step = int((z_stop - z_start) / dz)

    # draw the sample surface
    ax = draw_rectangle(
        ax,
        1,
        1,
        center=(cfg.sample.center.x, cfg.sample.center.y, cfg.sample.dimensions.lz),
        color="grey",
        alpha=0.5,
        label="Impedance surface",
    )

    key, subkey = jrandom.split(jrandom.PRNGKey(0))

    # draw the collocation points
    dom_data = next(iter(dom_loader))
    bnd_data = next(iter(bnd_loader))
    mes_data, _ = next(iter(data_loader))

    # plot the collocation points
    x, y, z = dom_data["x"], dom_data["y"], dom_data["z"]
    ax.scatter(x, y, z, c="g", marker="*", label="Domain points")

    # plot the boundary points
    x, y, z = bnd_data["x"], bnd_data["y"], bnd_data["z"]
    ax.scatter(x, y, z, c="b", marker="x", label="Boundary points")

    # plot the measurement points
    x, y, z = mes_data["x"], mes_data["y"], mes_data["z"]
    ax.scatter(x, y, z, c="k", marker="o", label="Data points")

    print(x.shape)

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

    # modify the view angle
    ax.view_init(elev=19, azim=150)

    # set z limit
    ax.set_zlim(0, jnp.max(z))

    # set legend
    plt.tight_layout()

    # export as as pgf and show
    ax.legend(loc="upper right", fontsize="small")
    plt.savefig("baffle_with_datapoints.pgf", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
