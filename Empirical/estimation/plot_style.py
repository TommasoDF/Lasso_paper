# plot_style.py
import matplotlib.pyplot as plt

def apply():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "axes.labelsize": 12,
        "axes.labelcolor": "black",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler("color", ["black"]),
        "grid.color": "0.7",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.5,
        "lines.solid_capstyle": "round",
        "text.usetex": True,
        "font.family": "serif",
        "legend.frameon": False,
        "legend.fontsize": 16,
    })
