from zpinn.dataio import DomainSampler, PressureDataset
from omegaconf import OmegaConf
import optax

config = OmegaConf.load("conf/experiments/single_freq/config.yaml")


def data_handlers(config, restrict_to=None):

    # extract domain and boundary configurations
    domain = config.collocation.domain
    boundary = config.collocation.boundary

    # create dataset
    dataset = PressureDataset(
        path=config.DATASET.path,
    )

    if restrict_to is not None:
        dataset.restrict_to(restrict_to)

    # domain points
    n_dom = config.DOMAIN.num_points
    dom_limits = {
        "x": (domain.x.min, domain.x.max),
        "y": (domain.y.min, domain.y.max),
        "z": (domain.z.min, domain.z.max),
        "f": (domain.f.min, domain.f.max),
    }
    dom_distr = {
        "x": domain.x.distribution,
        "y": domain.y.distribution,
        "z": domain.z.distribution,
        "f": domain.f.distribution,
    }
    # boundary points
    n_bnd = config.BOUNDARY.num_points
    bnd_limits = {
        "x": (boundary.x.min, boundary.x.max),
        "y": (boundary.y.min, boundary.y.max),
        "z": (boundary.z.min, boundary.z.max),
        "f": (boundary.f.min, boundary.f.max),
    }
    bnd_distr = {
        "x": boundary.x.distribution,
        "y": boundary.y.distribution,
        "z": boundary.z.distribution,
        "f": boundary.f.distribution,
    }
    # create domain generator
    dom_gen = DomainSampler(
        n_points=n_dom,
        limits=dom_limits,
        transforms=dataset.transforms,
        distributions=dom_distr,
    )
    # create boundary generator
    bnd_gen = DomainSampler(
        n_points=n_bnd,
        limits=bnd_limits,
        transforms=dataset.transforms,
        distributions=bnd_distr,
    )

    # create eval generator
    eval_gen = bnd_gen
    eval_gen.n_points = int(100**2)

    return dataset, dom_gen, bnd_gen, eval_gen


def optimizers(config):
    # Define optimizerss
    optimizer_lamdas = optax.adam(learning_rate=config.train.learning_rate.weights)
    optimizer_coeffs = optax.sgd(learning_rate=config.train.learning_rate.coeffs)

    scheduler = optax.join_schedules(
        schedules=[
            optax.constant_schedule(config.train.learning_rate.model),
            optax.cosine_decay_schedule(
                config.train.learning_rate.model, config.train.steps // 2
            ),
        ],
        boundaries=[config.train.steps // 2],
    )

    optimizer_model = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
    )
    return scheduler, optimizer_model, optimizer_lamdas, optimizer_coeffs
