import sys

sys.path.append("src")
from zpinn.dataio import BoundarySampler, DomainSampler, PressureDataset


def get_loaders(config, custom_transforms=None):
    dataset = PressureDataset(config.paths.dataset)

    if custom_transforms is not None:
        dataset.transforms = custom_transforms

    dataloader = dataset.get_dataloader(
        batch_size=config.batch.data.batch_size, shuffle=True
    )

    transforms = dataset.transforms

    dom_sampler = DomainSampler(
        batch_size=config.batch.domain.batch_size,
        limits=config.batch.domain.limits,
        distributions=config.batch.domain.distributions,
        transforms=transforms,
    )

    bnd_sampler = BoundarySampler(
        batch_size=config.batch.boundary.batch_size,
        limits=config.batch.boundary.limits,
        distributions=config.batch.boundary.distributions,
        transforms=transforms,
    )

    return dataloader, dom_sampler, bnd_sampler, transforms
