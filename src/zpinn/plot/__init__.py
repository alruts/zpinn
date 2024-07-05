import matplotlib.pyplot as plt

# global params
FONT_SIZE = 10
FIG_SIZE = (4.25, 3.0)  # (width, height) in inches

# set all font sizes to FONT_SIZE
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE
plt.rcParams["figure.titlesize"] = FONT_SIZE

# set the figure size
plt.rcParams["figure.figsize"] = FIG_SIZE

# Enable Latex
try:
    plt.rc("text", usetex=True)
except:
    pass
