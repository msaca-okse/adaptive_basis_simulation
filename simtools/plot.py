import matplotlib.pyplot as plt


def dark(fontsize=16):
    plt.style.use("dark_background")
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize


def despine(ax):
    for a in ax.flatten() if hasattr(ax, "flatten") else [ax]:
        for spine in a.spines.values():
            spine.set_visible(False)
