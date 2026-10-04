"""Scaling: orientations in a uniform grid of the fundamental zone versus spacing, per crystal symmetry,
against the orientations the coarse-to-fine ODF actually evaluates per level and the final TT bases."""
import os

import numpy as np

import figstyle as fs
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# cubochoric grid of the cubic zone at orix resolution 2 deg: 100 347 orientations (measured); N ~ h^-3 / |G|
N_CUBIC_2DEG = 100347
GROUPS = [("cubic (432)", 24), ("hexagonal (622)", 12), ("orthorhombic (222)", 4), ("triclinic (1)", 1)]
DATASETS = [("levels_domains_mosaicity_1p0deg.npz", "simulated, 1° mosaicity", 2320),
            ("levels_domains_mosaicity_10p0deg.npz", "simulated, 10° mosaicity", 14719),
            ("levels_al.npz", "Al1050, 15 % deformed", None)]


def uniform_count(h, order):
    return N_CUBIC_2DEG * (2.0 / h) ** 3 * 24 / order


def fig_scaling(mode, al_basis_K=None):
    m = fs.setup(mode)
    fig, ax = plt.subplots(figsize=(10, 6))
    h = np.geomspace(0.1, 5, 100)
    ax.set_xlim(5.5, 0.035)
    for name, order in GROUPS:
        ax.plot(h, uniform_count(h, order), color=m["neutral"], lw=1.5, ls="--")
        ax.text(0.095, uniform_count(0.1, order), f"uniform, {name}", color=m["text2"], fontsize=8.5,
                va="center", ha="left")
    for i, (f, label, K) in enumerate(DATASETS):
        path = os.path.join(HERE, f)
        if not os.path.exists(path):
            continue
        z = np.load(path)
        n = len([k for k in z if k.startswith("q")])
        hs = [float(z[f"h{j}"]) for j in range(n)]
        ks = [len(z[f"q{j}"]) for j in range(n)]
        col = m["series"][fs.DATASET_SLOTS[i]]
        ax.plot(hs, ks, "o-", color=col, label=f"{label}: orientations per ODF level")
        K = K if K is not None else al_basis_K
        if K:
            ax.plot([0.2], [K], marker="*", ms=16, color=col, markeredgecolor=m["surface"], ls="none")
    ax.plot([], [], marker="*", ms=14, ls="none", color=m["text2"], label="TT basis taken from the ODF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([4, 2, 1, 0.5, 0.25, 0.1], ["4", "2", "1", "0.5", "0.25", "0.1"])
    ax.minorticks_off()
    ax.set_xlabel("grid spacing (deg)")
    ax.set_ylabel("orientations")
    ax.set_title("Orientations needed: uniform grid versus coarse-to-fine ODF")
    ax.set_ylim(1e3, 1e11)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fs.save(fig, "scaling", mode)


if __name__ == "__main__":
    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for mode in ("light", "dark"):
        print(fig_scaling(mode, K))
