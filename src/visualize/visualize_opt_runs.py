# %%
import os
import sys

import pandas
import pandas as pd
import seaborn as sns
import tensorboard
import tensorboard as tb
from matplotlib import pyplot as plt
from packaging import version
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator
from omegaconf import OmegaConf
import seaborn as sns
import numpy as np

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert (
    major_ver >= 2 and minor_ver >= 3
), "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)


# dfs = {
#     "mlp": {
#         "depth":
#     },
#     "mmlp": {
#         "depth": None,
#         "width": None,
#     },
# }


frequencies = [100, 250, 500, 1000, 2000]
path = r"C:\Users\STNj\dtu\thesis\zpinn\multirun\2024-07-04\15-39-13"
cmap = sns.color_palette("husl", 5)
result_table = pd.DataFrame(
    columns=["model", "frequency", "depth", "width", "alpha", "beta"]
)

compute_error = lambda x, y: (x - y) ** 2 / y**2
score_fn = lambda x, y: np.sqrt(x * y)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# loop through all folders in path
for folder in os.listdir(path):
    # get the path to the folder
    folder_path = os.path.join(path, folder)

    # get hydra subfolder
    hydra_path = os.path.join(folder_path, ".hydra")
    # load config
    try:
        cfg = OmegaConf.load(os.path.join(hydra_path, "config.yaml"))
    except FileNotFoundError:
        continue

    # get frequency, depth and width
    frequency = frequencies[cfg._idx]
    hidden_layers = cfg.architecture.hidden_layers
    hidden_features = cfg.architecture.hidden_features
    name = cfg.architecture.name

    # print(f"frequency: {frequency}, depth: {hidden_layers}, width: {hidden_features}")

    # find file that starts with events
    for file in os.listdir(folder_path):
        if file.startswith("events"):
            event_file = os.path.join(folder_path, file)

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    scalar_data = []
    tags = ea.Tags()["scalars"]
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            scalar_data.append({"tag": tag, "step": event.step, "value": event.value})

    df = pd.DataFrame(scalar_data)

    name_str = f"{name}/{hidden_layers}/{hidden_features}"
    new_row = {
        "model": name_str,
        "frequency": frequency,
        "depth": hidden_layers,
        "width": hidden_features,
        "alpha": compute_error(
            df[df["tag"] == "Coeffs/alpha"]["value"].values[-1],
            df[df["tag"] == "Impedance/RealStar"]["value"].values[-1],
        ),
        "beta": compute_error(
            df[df["tag"] == "Coeffs/beta"]["value"].values[-1],
            df[df["tag"] == "Impedance/ImagStar"]["value"].values[-1],
        ),
    }
    result_table.loc[len(result_table)] = new_row

    if name == "modified_siren" and hidden_layers == 3 and hidden_features == 8:
        df_alpha = df[df["tag"] == "Coeffs/alpha"]
        df_beta = df[df["tag"] == "Coeffs/beta"]
        df_gt_alpha = df[df["tag"] == "Impedance/RealStar"]
        df_gt_beta = df[df["tag"] == "Impedance/ImagStar"]

        ax.plot(df_alpha["step"], df_alpha["value"], color=cmap[cfg._idx], label=frequency)
        ax.plot(df_gt_alpha["step"], df_gt_alpha["value"], '--', color=cmap[cfg._idx], label=frequency)
        ax.plot(df_beta["step"], df_beta["value"], color=cmap[cfg._idx], label=frequency)
        ax.plot(df_gt_beta["step"], df_gt_beta["value"], '--', color=cmap[cfg._idx], label=frequency)

    # break
plt.legend()

# make total score for each model


total_score_table = pd.DataFrame(columns=["model", "total"])

# find unique models
unique_models = result_table["model"].unique()
# loop through unique models
for model in unique_models:
    # get the rows for the model
    model_rows = result_table[result_table["model"] == model]
    # get the total score
    total_score = score_fn(model_rows["alpha"].sum() ,model_rows["beta"].sum())
    # add the total score to the result table
    total_score_table.loc[len(total_score_table)] = {
        "model": model,
        "total": total_score,
    }

# sort the total score table
total_score_table = total_score_table.sort_values("total", ascending=True)
total_score_table
