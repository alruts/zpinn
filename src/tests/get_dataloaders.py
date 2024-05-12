import sys

sys.path.append("src")
from zpinn.dataio import PressureDataset, DomainSampler, BoundarySampler


data_path = r"C:\Users\STNj\dtu\thesis\code\data\processed\inf_baffle.pkl"


def get_dataloaders(data_path=data_path):

    dataset = PressureDataset(data_path)
    dataloader = dataset.get_dataloader(batch_size=16, shuffle=True)
    data_iterator = iter(dataloader)
    
    dom_sampler = DomainSampler(
        batch_size=16,
        limits=dict(x=(0, 1), y=(0, 1), z=(0, 1), f=(0, 1)),
        distributions=dict(x="uniform", y="uniform", z="uniform", f="uniform"),
        transforms=dataset.transforms,
    )
    dom_iterator = iter(dom_sampler)

    bnd_sampler = BoundarySampler(
        batch_size=16,
        limits=dict(x=(0, 1), y=(0, 1), z=(0, 0), f=(0, 1)),
        distributions=dict(x="grid", y="grid", z="uniform", f="uniform"),
        transforms=dataset.transforms,
    )
    bnd_iterator = iter(bnd_sampler)

    transforms = dict(
        x0=dataset.transforms["x"][0],
        xc=dataset.transforms["x"][1],
        y0=dataset.transforms["y"][0],
        yc=dataset.transforms["y"][1],
        z0=dataset.transforms["z"][0],
        zc=dataset.transforms["z"][1],
        f0=dataset.transforms["f"][0],
        fc=dataset.transforms["f"][1],
        a0=dataset.transforms["real_pressure"][0],
        ac=dataset.transforms["real_pressure"][1],
        b0=dataset.transforms["imag_pressure"][0],
        bc=dataset.transforms["imag_pressure"][1],
    )
    return data_iterator, dom_iterator, bnd_iterator, transforms
