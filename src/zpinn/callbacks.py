import sys
import matplotlib.pyplot as plt
import numpy as np
from constants import _rho0, pi
from jax import vmap

sys.path.append("src")
from zpinn.dataio import DomainSampler
from zpinn.diff_ops import dz_fn


class ModelEvaluator:
    """ """
    def __init__(self, model, transforms, writer):
        self.model = model
        self.transforms = transforms
        self.limits = dict(
            x=(-0.5, 0.5),
            y=(-0.5, 0.5),
            z=(0, 0),
            f=(0, 0),
        )
        self.distributions = dict(
            x="grid",
            y="grid",
            z="uniform",
            f="uniform",
        )
        self.sampler = DomainSampler(
            batch_size=int(100**2),
            transforms=transforms,
            limits=self.limits,
            distributions=self.distributions,
        )

        self.writer = writer

        self.re_fn = lambda x, y, z, f: self.model(x, y, z, f)[0]
        self.im_fn = lambda x, y, z, f: self.model(x, y, z, f)[1]

        coords = self.unpack_batch(self.sampler.gen_data())
        chi, ypsilon, zeta, fhi = coords

        _, _, (_, zc), (f0, fc) = self.unpack_transforms()

        a0, ac = self.transforms["real_pressure"]
        b0, bc = self.transforms["imag_pressure"]

        # compute the pressure field and its derivative from model
        phi_re, phi_im = self.compute_pressure_field(chi, ypsilon, zeta, fhi)
        dphi_dzeta_re, dphi_dzeta_im = self.compute_derivative_pressure_field(
            chi, ypsilon, zeta, fhi
        )

        # compute the pressure field
        self.pressure = self.compute_pressure(phi_re, phi_im, a0, ac, b0, bc)
        self.uz = self.compute_particle_velocity(
            dphi_dzeta_re, dphi_dzeta_im, ac, bc, zc, fhi, fc, f0
        )
        self.impedance = self.compute_impedance(self.pressure, self.uz)

    def unpack_batch(self, batch):
        coords = batch
        return (coords["chi"], coords["yps"], coords["zet"], coords["fhi"])

    def unpack_transforms(self):
        return (
            self.transforms["x"],
            self.transforms["y"],
            self.transforms["z"],
            self.transforms["f"],
        )

    def compute_pressure_field(self, chi, ypsilon, zeta, fhi):
        phi_re = vmap(self.re_fn)(chi, ypsilon, zeta, fhi)
        phi_im = vmap(self.im_fn)(chi, ypsilon, zeta, fhi)
        return phi_re, phi_im

    def compute_derivative_pressure_field(self, chi, ypsilon, zeta, fhi):
        dphi_dzeta_re = vmap(dz_fn(self.re_fn))(chi, ypsilon, zeta, fhi)
        dphi_dzeta_im = vmap(dz_fn(self.im_fn))(chi, ypsilon, zeta, fhi)
        return dphi_dzeta_re, dphi_dzeta_im

    def compute_pressure(self, phi_re, phi_im, a0, ac, b0, bc):
        return (phi_re * ac + a0) + 1j * (phi_im * bc + b0)

    def compute_particle_velocity(
        self, dphi_dzeta_re, dphi_dzeta_im, ac, bc, zc, fhi, fc, f0
    ):
        const_ = 1 / (1j * (2 * pi * (fhi * fc + f0)) * _rho0)
        const_ /= zc
        uz = (ac * dphi_dzeta_re) + 1j * (bc * dphi_dzeta_im)
        return uz * const_

    def compute_impedance(self, pressure, uz):
        return pressure / uz


def log_weight_histograms_to_tensorboard(model, writer, step):
    """Logs histograms of weights for all layers in the model to TensorBoard.

    Args:
        model: The Equinox model for which to log weight histograms.
        writer (SummaryWriter): TensorBoard SummaryWriter object for logging.
        global_step (int): Global step value for the current iteration.
    """
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "weight"):
            # Check if layer has a weight attribute (assumes weight for linear layers)
            weight = layer.weight.flatten()
            weight = np.array(weight)
            writer.add_histogram(f"layer_{i}/weights", weight, global_step=step)
