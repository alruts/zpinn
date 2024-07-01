import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from zpinn.constants import _c0


def get_center(lx, ly, lz, center=(0, 0, 0)):
    x0, y0, z0 = center
    x0 -= lx / 2
    y0 -= ly / 2
    z0 -= lz / 2
    return x0, y0, z0


def calculate_room_modes(lx, ly, lz, speed_of_sound=_c0, num_modes=5):
    """Calculates the first `num_modes` room modes and returns them as a list of tuples.

    Args:
        lx: Length of the room in meters.
        ly: Width of the room in meters.
        lz: Height of the room in meters.
        speed_of_sound: Speed of sound in meters per second (default: 343).
        num_modes: Number of modes to calculate (default: 5).

    Returns:
        A list of tuples containing (p, q, r, frequency) for each mode.
    """

    modes = []
    for nx in range(0, num_modes + 1):
        for ny in range(0, num_modes + 1):
            for nz in range(0, num_modes + 1):
                frequency = (
                    0.5
                    * speed_of_sound
                    * math.sqrt((nx / lx) ** 2 + (ny / ly) ** 2 + (nz / lz) ** 2)
                )
                modes.append((nx, ny, nz, frequency))
    return modes


def create_room_modes_dataframe(lx, ly, lz, speed_of_sound=_c0, num_modes=5):
    """Calculates room modes and creates a pandas DataFrame containing the results.

    Args:
        lx: Length of the room in meters.
        ly: Width of the room in meters.
        lz: Height of the room in meters.
        speed_of_sound: Speed of sound in meters per second (default: 343).
        num_modes: Number of modes to calculate (default: 5).

    Returns:
        A pandas DataFrame containing the calculated room modes.
    """

    modes = calculate_room_modes(lx, ly, lz, speed_of_sound, num_modes)
    df = pd.DataFrame(
        modes, columns=["Mode (nx)", "Mode (ny)", "Mode (nz)", "Frequency (Hz)"]
    )
    return df


def draw_shoebox(ax_3d, lx, ly, lz, center=(0, 0, 0), label=None):
    """Draws a 3D rectangular room in a matplotlib 3D axis.

    Args:
        ax: Matplotlib 3D axis.
        lx: Length of the room in meters.
        ly: Width of the room in meters.
        lz: Height of the room in meters.

    Returns:
        The modified matplotlib 3D axis.
    """
    x0, y0, z0 = get_center(lx, ly, lz, center)
    # Coordinates for the vertices of the rectangular room
    vertices = np.array(
        [
            [0, 0, 0],
            [0, ly, 0],
            [lx, ly, 0],
            [lx, 0, 0],
            [0, 0, lz],
            [0, ly, lz],
            [lx, ly, lz],
            [lx, 0, lz],
        ]
    )

    # Define the 6 faces of the rectangular room
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
    ]

    # Plot the rectangular room wireframe
    for face in faces:
        x = [vertex[0] + x0 for vertex in face]
        y = [vertex[1] + y0 for vertex in face]
        z = [vertex[2] + z0 for vertex in face]
        
        
        if np.all(np.array(face).flatten() == np.array(faces[0]).flatten()) and label is not None:
            ax_3d.plot(x, y, z, color="k", label=label)
        else:
            ax_3d.plot(x, y, z, color="k")
            ax_3d.plot([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]], color="k")

    return ax_3d


def draw_line_with_flat_ends(
    ax_3d, x0, y0, z0, x1, y1, z1, color="k", linestyle="-", orientation="horizontal"
):
    """Draws a line with flat ends in a matplotlib 3D axis.

    Args:
        ax: Matplotlib 3D axis.
        x0: x-coordinate of the start of the line.
        y0: y-coordinate of the start of the line.
        z0: z-coordinate of the start of the line.
        x1: x-coordinate of the end of the line.
        y1: y-coordinate of the end of the line.
        z1: z-coordinate of the end of the line.
        color: Color of the line (default: black).
        linestyle: Linestyle of the line (default: dashed).

    Returns:
        The modified matplotlib 3D axis.
    """
    d = 0.01

    ax_3d.plot(
        [x0 - d, x1 - d],
        [y0 - d, y1 - d],
        [z0 - d, z1 - d],
        color=color,
        linestyle=linestyle,
    )
    if orientation == "horizontal":
        ax_3d.plot([x0 - d, x0 - d], [y0 - d, y0 - d], [z0 - 2 * d, z0], color=color)
        ax_3d.plot([x1 - d, x1 - d], [y1 - d, y1 - d], [z1 - 2 * d, z1], color=color)
    else:
        ax_3d.plot([x0 - d, x0 - d], [y0 - 2 * d, y0], [z0 - d, z0 - d], color=color)
        ax_3d.plot([x1 - d, x1 - d], [y1 - 2 * d, y1], [z1 - d, z1 - d], color=color)

    return ax_3d


def mark_side_lengths(ax_3d, lx, ly, lz, center=(0, 0, 0), d=0.15):
    x0, y0, z0 = get_center(lx, ly, lz, center)

    # mark the side lengths a
    ax_3d.text(
        -(x0 + lx / 2),
        -(y0 - d*2),
        0,
        "$l_x=1$ m",
        color="k",
        horizontalalignment="center",
    )

    ax_3d.text(
        x0 - d*2,
        y0 + ly / 2,
        0,
        "$l_y=1$ m",
        color="k",
        horizontalalignment="center",
    )

    ax_3d.text(
        x0 - d*1.4,
        y0 - d*1.4,
        lz / 2,
        "$l_z=0.05$ m",
        color="k",
        horizontalalignment="center",
    )
    
    # draw lines
    ax_3d.plot(
        [-x0, -(x0 + lx)],
        [-(y0 - d/2), -(y0 - d/2)],
        [0, 0],
        color="k",
        linestyle="--",
    )
    
    ax_3d.plot(
        [(x0 - d/2), (x0 - d/2)],
        [y0, (y0 + ly)],
        [0, 0],
        color="k",
        linestyle="--",
    )
    
    ax_3d.plot(
        [(x0 - d/4), (x0 - d/4)],
        [(y0 - d/4), (y0 - d/4)],
        [0, lz],
        color="k",
        linestyle="--",
    )
    
    return ax_3d


def draw_receiver(ax_3d, rx, ry, rz, color="b", label=""):
    """Draws a red receiver in a 3D plot.

    Args:
        ax_3d: Matplotlib 3D axis.
        rx: x-coordinate of the receiver.
        ry: y-coordinate of the receiver.
        rz: z-coordinate of the receiver.
    Returns:
        The modified matplotlib 3D axis.
    """
    ax_3d.scatter(rx, ry, rz, color=color, s=5, label=label)
    # draw line from floor to receiver
    ax_3d.plot([rx, rx], [ry, ry], [0, rz], color=color, linestyle="--")
    # draw 'x' marker at floor
    ax_3d.scatter(rx, ry, 0, color=color, marker="x", s=20)
    return ax_3d


def draw_source(ax_3d, sx, sy, sz, color="r"):
    """Draws a blue source in a 3D plot.

    Args:
        ax_3d: Matplotlib 3D axis.
        rx: x-coordinate of the receiver.
        ry: y-coordinate of the receiver.
        rz: z-coordinate of the receiver.
    Returns:
        The modified matplotlib 3D axis.
    """
    ax_3d.scatter(sx, sy, sz, color=color, s=20, label="Source")
    # draw line from floor to receiver
    ax_3d.plot([sx, sx], [sy, sy], [0, sz], color=color, linestyle="--")
    # draw 'x' marker at floor
    ax_3d.scatter(sx, sy, 0, color=color, marker="x", s=20, zorder=1)

    return ax_3d


def draw_receiver_grid(ax_3d, lx, ly, lz, nx=3, ny=3, center=(0, 0, 0)):
    """Draws a grid of receivers in a 3D plot."""

    x0, y0, z0 = get_center(lx, ly, lz, center)
    # Spacing between receivers
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    # Create a grid of receivers
    for i in range(nx):
        for j in range(ny):
            rx = i * dx + x0
            ry = j * dy + y0
            ax_3d = draw_receiver(ax_3d, rx, ry, lz, color="b")
    return ax_3d


def draw_rectangle(
    ax_3d,
    lx,
    ly,
    center=(0, 0, 0),
    draw_height_lines=False,
    color="r",
    alpha=0.2,
    label="",
    linestyle="-",
    zorder=0,
):
    """Draws a 2D rectangle in a matplotlib axis.

    Args:
        ax: Matplotlib axis.
        lx: Length of the rectangle in meters.
        ly: Width of the rectangle in meters.
        lz: Height of the rectangle in meters.

    Returns:
        The modified matplotlib axis.
    """
    x0, y0, z0 = get_center(lx, ly, 0, center)

    # Coordinates for the vertices of the rectangle
    vertices = np.array(
        [
            [0, 0, z0],
            [0, ly, z0],
            [lx, ly, z0],
            [lx, 0, z0],
            [0, 0, z0],
        ]
    )
    # Plot the rectangle
    x = [vertex[0] + x0 for vertex in vertices]
    y = [vertex[1] + y0 for vertex in vertices]
    z = [vertex[2] for vertex in vertices]

    if draw_height_lines:
        # draw line from each corner to floor
        for i in range(4):
            ax_3d.plot([x[i], x[i]], [y[i], y[i]], [0, z0], color=color, linestyle="--")
            # draw 'x' marker at floor
            ax_3d.scatter(x[i], y[i], 0, color=color, marker="x", s=20)

    # fill the rectangle
    ver_zip = [list(zip(x, y, z))]
    ax_3d.add_collection3d(
        Poly3DCollection(
            ver_zip,
            edgecolors=color,
            linestyle=linestyle,
            facecolors=color,
            linewidths=1,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
    )

    return ax_3d


# # Example usage:
# room_dimensions = {
#     "lx": 3.14,  # width
#     "ly": 4.38,  # depth
#     "lz": 3.27,  # height
# }

# sample_dimensions = {
#     "lx": 1.0,
#     "ly": 1.0,
#     "lz": 0.1,
# }
# num_modes = 5  # Number of modes to calculate

# room_modes_df = create_room_modes_dataframe(**room_dimensions, num_modes=num_modes)
# print(room_modes_df)


# # sort the DataFrame by frequency
# room_modes_df = room_modes_df.sort_values(by="Frequency (Hz)")
# print(room_modes_df)


# # export to a CSV file
# room_modes_df.to_csv("room_modes.csv", index=True)
