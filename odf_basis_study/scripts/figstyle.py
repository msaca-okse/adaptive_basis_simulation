"""Shared figure style: light and dark variants of every figure, the reference categorical palette."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

MODES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"], "neutral": "#8a8984",
              "seq": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#33332f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"], "neutral": "#8f8e88",
             "seq": ["#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]},
}
# fixed entity -> slot (colour follows the entity)
SLOT = {"indexed": 0, "uniform": 1, "odf": 2}
# datasets get slots of their own (yellow, magenta, violet), so they never read as a basis
DATASET_SLOTS = [3, 4, 6]


def setup(mode):
    m = MODES[mode]
    plt.rcParams.update({
        "figure.facecolor": m["surface"], "axes.facecolor": m["surface"], "savefig.facecolor": m["surface"],
        "text.color": m["text"], "axes.labelcolor": m["text2"], "axes.edgecolor": m["grid"],
        "xtick.color": m["text2"], "ytick.color": m["text2"], "axes.grid": True, "grid.color": m["grid"],
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2,
        "axes.prop_cycle": matplotlib.cycler(color=m["series"]),
    })
    return m


def seq_cmap(mode):
    return LinearSegmentedColormap.from_list(f"seq_{mode}", MODES[mode]["seq"])


def color(mode, entity):
    return MODES[mode]["series"][SLOT[entity]]


def save(fig, name, mode):
    path = os.path.join(OUT, f"{name}_{mode}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
