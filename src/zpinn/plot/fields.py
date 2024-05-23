import numpy as np
import matplotlib.pyplot as plt


def scalar_field(
    P,
    X,
    Y,
    cmap="seismic",
    balanced_cmap=True,
    xlabel="x (m)",
    ylabel="y (m)",
    cbar_label="Pressure (Pa)",
    ax=None,
):
    """Plots a scalar field.

    Args:
    -----
        - P (np.ndarray): scalar field.
        - X (np.ndarray): x-coordinates of the grid.
        - Y (np.ndarray): y-coordinates of the grid.
        - cmap (str): colormap.
        - balanced_cmap (bool): if True, the colormap will be balanced around 0.
        - xlabel (str): x-axis label.
        - ylabel (str): y-axis label.
        - cbar_label (str): colorbar label.
    """

    if balanced_cmap:
        vmax = np.max(np.abs(P))
        vmin = -vmax
    else:
        vmax = np.max(P)
        vmin = np.min(P)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(
        P,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)

    return ax


def vector_field(
    Ux,
    Uy,
    X,
    Y,
    xlabel="x (m)",
    ylabel="y (m)",
    cbar_label="Velocity (m/s)",
    downsample=3,
    ax=None,
):
    """Plots a vector field.

    Args:
    -----
        - Ux (np.ndarray): x-component of the velocity field.
        - Uy (np.ndarray): y-component of the velocity field.
        - X (np.ndarray): x-coordinates of the grid.
        - Y (np.ndarray): y-coordinates of the grid.
        - xlabel (str): x-axis label.
        - ylabel (str): y-axis label.
        - cbar_label (str): colorbar label.
        - downsample (int): downsample the number of arrows in the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    sc = ax.scatter(X, Y, c=np.sqrt(Ux**2 + Uy**2), cmap="RdBu_r", vmin=0, vmax=1.5)
    qu = ax.quiver(
        X[::downsample, ::downsample],
        Y[::downsample, ::downsample],
        Ux[::downsample, ::downsample],
        Uy[::downsample, ::downsample],
        color="k",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)

    return ax


def vector_field_3d(
    Ux,
    Uy,
    Uz,
    X,
    Y,
    Z,
    xlabel="x (m)",
    ylabel="y (m)",
    zlabel="z (m)",
    cbar_label="Velocity (m/s)",
    ax=None,
    cmap=plt.cm.viridis,
):
    """Plots a 3D vector field.

    Args:
    -----
        - Ux (np.ndarray): x-component of the velocity field.
        - Uy (np.ndarray): y-component of the velocity field.
        - Uz (np.ndarray): z-component of the velocity field.
        - X (np.ndarray): x-coordinates of the grid.
        - Y (np.ndarray): y-coordinates of the grid.
        - Z (np.ndarray): z-coordinates of the grid.
        - xlabel (str): x-axis label.
        - ylabel (str): y-axis label.
        - zlabel (str): z-axis label.
        - cbar_label (str): colorbar label.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Calculate the magnitude of the velocity vectors
    magnitude = np.sqrt(Ux**2 + Uy**2 + Uz**2)

    # Plot particle velocity vector field in 3D
    ax = plt.axes(projection="3d")

    # Normalize the magnitude to map it to colors properly
    norm = plt.Normalize(magnitude.min(), magnitude.max())
    cmap = plt.cm.viridis

    # dynamic range
    Z_range = np.max(Z)

    # Plot the quiver
    q = ax.quiver(
        X,
        Y,
        Z,
        Ux,
        Uy,
        Uz,
        colors="k",
        pivot="tail",
        units = 'inches',
        scale=1.0 / Z_range,
        width=0.01,
    )

    # get arrow head coordinates

    sc = ax.scatter(X, Y, Z, c=magnitude, cmap=cmap, norm=norm)

    # colorbar to the scatter plot
    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim([X.min(), X.max()])
    ax.set_ylim([Y.min(), Y.max()])

    return ax


def plot_scalar_field_grid(
    plot_fn, fields, nrows=1, ncols=1, figsize=(10, 6), **kwargs
):
    """Plots a grid of scalar fields using a provided plotting function.

    Args:
        scalar_field_plotter (function): Function for plotting a single scalar field.
        fields (list): List of tuples containing field data and optional labels:
            `[(P1, X1, Y1, label1), (P2, X2, Y2, label2), ...]`
        nrows (int, optional): Number of rows in the grid. Defaults to 1.
        ncols (int, optional): Number of columns in the grid. Defaults to 1.
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
        **kwargs: Additional keyword arguments passed to `plot_fn`.
    """

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # handle case where only 1 image
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])

    for ax, (P, X, Y, label) in zip(axes.flat, fields):
        if label is not None:
            ax.set_title(label)
        plot_fn(P, X, Y, ax=ax, **kwargs)

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    P = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    ax = scalar_field(P, x, y)

    # Vector field
    Ux = np.sin(2 * np.pi * X)
    Uy = np.ones_like(Y) * 0.4

    ax = vector_field(Ux, Uy, X, Y)

    # make test for 3d vector field
    x, y, z = np.linspace(-1, 1, 6), np.linspace(-1, 1, 6), np.linspace(-1, 1, 6)
    X, Y, Z = np.meshgrid(x, y, z)
    Ux, Uy, Uz = (X, Y, -Z)
    ax = vector_field_3d(Ux, Uy, Uz, X, Y, Z)
    plt.show()
