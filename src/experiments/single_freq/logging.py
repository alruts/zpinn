

writer = 0
# add custom scalars to writer
layout = {
    "Figrures": {
        "Epsilon": [
            "Multiline",
            ["Epsilon/Data", "Epsilon/PDE", "Epsilon/Boundary"],
        ],
        "Loss/PDE": [
            "Multiline",
            ["Loss/PDE/Total", "Loss/PDE/Real", "Loss/PDE/Imag"],
        ],
        "Loss/Boundary": [
            "Multiline",
            ["Loss/Boundary/Total", "Loss/Boundary/Real", "Loss/Boundary/Imag"],
        ],
        "Loss/Data": [
            "Multiline",
            ["Loss/Data/Total", "Loss/Data/Real", "Loss/Data/Imag"],
        ],
        "Zn": [
            "Multiline",
            [
                "Zn/Real",
                "Zn/Imag",
            ],
        ],
        "Zn/estimate1": [
            "Multiline",
            [
                "Zn/estimate1/Real",
                "Zn/estimate1/Imag",
            ],
        ],
        "Zn/estimate2": [
            "Multiline",
            [
                "Zn/estimate2/Real",
                "Zn/estimate2/Imag",
            ],
        ],
    },
}
writer.add_custom_scalars(layout)


    # log to console
    logging.info(f"Step: {step}")
    logging.info(f"Loss: {aux['loss'].item()}")
    logging.info(f"Data Loss: {aux['data_loss'].item()}")
    logging.info(f"PDE Loss: {aux['pde_loss'].item()}")

    if bnd_gen is not None:
        logging.info(f"Boundary Loss: {aux['boundary_loss'].item()}")
        logging.info(f"Zn: {coeffs}")
        writer.add_scalar(
            "Loss/Boundary/Total", aux["boundary_loss"].item(), step
        )
        writer.add_scalar(
            "Loss/Boundary/Real", aux["boundary_loss_re"].item(), step
        )
        writer.add_scalar(
            "Loss/Boundary/Imag", aux["boundary_loss_im"].item(), step
        )
        writer.add_scalar("Zn/Real", coeffs[0].item(), step)
        writer.add_scalar("Zn/Imag", coeffs[1].item(), step)

    # log to tensorboard
    writer.add_scalar("Loss/PDE/Total", aux["pde_loss"].item(), step)
    writer.add_scalar("Loss/PDE/Real", aux["pde_loss_re"].item(), step)
    writer.add_scalar("Loss/PDE/Imag", aux["pde_loss_im"].item(), step)

    writer.add_scalar("Loss/Data/Total", aux["data_loss"].item(), step)
    writer.add_scalar("Loss/Data/Real", aux["data_loss_re"].item(), step)
    writer.add_scalar("Loss/Data/Imag", aux["data_loss_im"].item(), step)

    writer.add_scalar("Epsilon/Data", lambdas["data"].item(), step)
    writer.add_scalar("Epsilon/PDE", lambdas["pde"].item(), step)
    writer.add_scalar("Epsilon/Boundary", lambdas["bc"].item(), step)

    # add pressure distribution to tensorboard
    if step % evaluate_every == 0:
        coords = eval_gen.gen_data(jrandom.PRNGKey(0))
        inferencer = InferenceModel(model, transforms, coords)
        p_ = inferencer.pressure
        uz_ = inferencer.uz
        Zn_ = inferencer.impedance

        estimate1 = p_.mean() / -uz_.mean()
        estimate2 = (p_ / -uz_).mean()

        logging.info(f"p_.mean() / -uz_.mean(): {estimate1}")
        logging.info(f"(p_ / -uz_).mean(): {estimate2}")

        # reshape p
        N = jnp.sqrt(eval_gen.N).astype(int)
        p_ = p_.reshape((N, N))
        uz_ = uz_.reshape((N, N))
        Zn_ = Zn_.reshape((N, N))

        X = coords["chi"].reshape((100, 100))
        Y = coords["yps"].reshape((100, 100))

        fig, ax = plt.subplots(1, 3, figsize=(15, 3))
        ax.flatten()
        ax[0] = scalar_field(
            p_.real,
            X,
            Y,
            ax=ax[0],
            cmap="seismic",
            xlabel="x ()",
            ylabel="y ()",
            cbar_label="Pressure (Pa)",
        )
        ax[0].set_title("Pressure")
        ax[1] = scalar_field(
            uz_.real,
            X,
            Y,
            ax=ax[1],
            cmap="seismic",
            xlabel="x ()",
            ylabel="y ()",
            cbar_label="Particle velocity (m/s)",
        )
        ax[1].set_title("Particle velocity")
        ax[2] = scalar_field(
            Zn_.real,
            X,
            Y,
            ax=ax[2],
            cmap="seismic",
            xlabel="x ()",
            ylabel="y ()",
            cbar_label="Impedance (Pa.s/m3)",
        )
        ax[2].set_title("Impedance")
        writer.add_figure("Fields/Real", plt.gcf(), step)
        plt.close()

        fig, ax = plt.subplots(1, 3, figsize=(15, 3))
        ax.flatten()
        ax[0] = scalar_field(
            p_.imag,
            X,
            Y,
            ax=ax[0],
            cmap="seismic",
            xlabel="x ()",
            ylabel="y ()",
            cbar_label="Pressure (Pa)",
        )
        ax[0].set_title("Pressure")
        ax[1] = scalar_field(
            uz_.imag,
            X,
            Y,
            ax=ax[1],
            cmap="seismic",
            xlabel="x ()",
            ylabel="y ()",
            cbar_label="Particle velocity (m/s)",
        )
        ax[1].set_title("Particle velocity")
        ax[2] = scalar_field(
            Zn_.imag,
            X,
            Y,
            ax=ax[2],
            cmap="seismic",
            xlabel="x ()",
            ylabel="y ()",
            cbar_label="Impedance (Pa.s/m3)",
        )
        ax[2].set_title("Impedance")
        writer.add_figure("Fields/Imag", plt.gcf(), step)
        plt.close()

        #! continue with this tommorw
        #! write simple deterministic fn(x,y,z,f) , do the tranform and see if it works
        writer.add_scalar("Zn/estimate1/Real", estimate1.real.item(), step)
        writer.add_scalar("Zn/estimate1/Imag", estimate1.imag.item(), step)
        writer.add_scalar("Zn/estimate2/Real", estimate2.real.item(), step)
        writer.add_scalar("Zn/estimate2/Imag", estimate2.imag.item(), step)

