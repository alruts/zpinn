import sys

import jax.random as jrandom

sys.path.append("src")
from zpinn.dataio import BoundarySampler, DomainSampler, PressureDataset


def get_loaders(config, custom_transforms=None, restrict_to=None, snr=None):
    dataset = PressureDataset(config.paths.dataset)

    if custom_transforms is not None:
        dataset.transforms = custom_transforms

    if restrict_to is not None:
        print(f"Restricting dataset to {restrict_to}")
        dataset.restrict_to(**restrict_to)

    dataloader = dataset.get_dataloader(
        batch_size=config.batch.data.batch_size, shuffle=True, snr=snr, rng_key=config.random.seed
    )

    transforms = dataset.transforms
    dom_key, bnd_key = jrandom.split(config.random.seed)

    dom_sampler = DomainSampler(
        batch_size=config.batch.domain.batch_size,
        limits=config.batch.domain.limits,
        distributions=config.batch.domain.distributions,
        transforms=transforms,
        rng_key=dom_key,
    )

    bnd_sampler = BoundarySampler(
        batch_size=config.batch.boundary.batch_size,
        limits=config.batch.boundary.limits,
        distributions=config.batch.boundary.distributions,
        transforms=transforms,
        rng_key=bnd_key,
    )

    try:
        ref_coords, ref_gt = dataset.get_reference(
            f=config.batch.data.restrict_to["f"][0]
        )
    except:
        ref_coords, ref_gt = dataset.get_reference(1000)

    return dataloader, dom_sampler, bnd_sampler, ref_coords, ref_gt, transforms
