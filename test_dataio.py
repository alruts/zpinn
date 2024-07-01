# %%
from src.zpinn.dataio import PressureDataset

dataset = PressureDataset('data\processed\inf_baffle.pkl', snr=None, use_non_dim=0)
# restrict dataset
dataset.restrict_to(
    x = [-0.5, 0.5],
    y = [-0.5, 0.5],
    f = [2000,]
)


n_full_batch = dataset.n_x * dataset.n_y * dataset.n_z
dataloader = dataset.get_dataloader(batch_size=n_full_batch, shuffle=False, )
coords, gt = next(iter(dataloader))

# %% 
import matplotlib.pyplot as plt
import numpy as np
coords, gt = next(iter(dataloader))


x = coords['x'].reshape(dataset.n_x, dataset.n_y, dataset.n_z)
y = coords['y'].reshape(dataset.n_x, dataset.n_y, dataset.n_z)
z = coords['z'].reshape(dataset.n_x, dataset.n_y, dataset.n_z)
f = coords["f"].reshape(dataset.n_x, dataset.n_y, dataset.n_z)
p_im = gt["imag_pressure"].reshape(dataset.n_x, dataset.n_y, dataset.n_z)
p_re = gt["real_pressure"].reshape(dataset.n_x, dataset.n_y, dataset.n_z)

plt.subplot(121)
plt.scatter(x[:,:,0], y[:,:,0], c=p_re[:, :, 0], cmap='jet')
plt.colorbar()
plt.axis('equal')
plt.subplot(122)
plt.scatter(x[:,:,1], y[:,:,1], c=p_re[:, :, 1], cmap='jet')
plt.axis('equal')
plt.colorbar()
plt.tight_layout()
plt.show()


# %% 3d scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=p_im, cmap='jet')
plt.show()