"""Figures of the simulated samples: ODF levels, pole figures per level, ground-truth coverage of the
bases, orientation and misorientation maps, accuracy versus basis size."""
import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/dtu-compute/msaca/adaptive_basis_simulations")
from simtools import figures, io, paths, stability  # noqa: E402
from diffractom.odf_basis import nearest_misorientation_deg, symmetry_rotations  # noqa: E402

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TT = "/dtu-compute/msaca/claude_scratch/odf/tt"
PROC = "/dtu-compute/msaca/adaptive_basis_simulations/processed"
SAMPLES = {"domains_mosaicity_1p0deg": "1° mosaicity", "domains_mosaicity_10p0deg": "10° mosaicity"}
SYM = symmetry_rotations("cubic")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def levels(sample):
    z = np.load(os.path.join(HERE, f"levels_{sample}.npz"))
    return _levels(z)


def _levels(z):
    n = len([k for k in z if k.startswith("q")])
    return [{k: z[f"{k}{i}"] for k in ("q", "w", "h", "s", "res", "t")} for i in range(n)]


def grid_mats(h5):
    with h5py.File(h5, "r") as f:
        return f["grid_mats"][...]


# ------------------------------------------------------------------ pole figures

H111 = np.array([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], float) / np.sqrt(3)


def pole_figure(q, w, n=240, smooth=1.0):
    """Weighted <111> pole figure (upper hemisphere about the rotation axis z, stereographic)."""
    U = Rotation.from_quat(q).as_matrix()
    p = np.einsum("kij,hj->khi", U, H111).reshape(-1, 3)
    ww = np.repeat(w, len(H111))
    p[p[:, 2] < 0] *= -1
    x, y = p[:, 0] / (1 + p[:, 2]), p[:, 1] / (1 + p[:, 2])
    img, _, _ = np.histogram2d(y, x, bins=n, range=[[-1, 1], [-1, 1]], weights=ww)
    img = gaussian_filter(img, smooth)
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    img[xx ** 2 + yy ** 2 > 1] = np.nan
    return img / np.nanmax(img)


def fig_pole_figures(mode):
    m = fs.setup(mode)
    cmap = fs.seq_cmap(mode)
    cmap.set_bad(m["surface"])
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for r, (sample, label) in enumerate(SAMPLES.items()):
        for c, L in enumerate(levels(sample)):
            ax = axes[r, c]
            ax.imshow(pole_figure(L["q"], L["w"]), origin="lower", cmap=cmap, norm=PowerNorm(0.5, 0, 1),
                      extent=[-1, 1, -1, 1])
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=m["text2"], lw=0.8))
            ax.set_axis_off()
            ax.set_title(f"{float(L['h']):g}° grid · {len(L['q']):,} orientations", fontsize=10, color=m["text2"])
        axes[r, 0].text(-1.25, 0, label, rotation=90, va="center", ha="center", fontsize=12, color=m["text"])
    fig.suptitle("{111} pole figure of the bulk ODF at each level (rotation axis in the centre, √ intensity)",
                 fontsize=13, color=m["text"])
    return fs.save(fig, "pole_figures_sim", mode)


# ------------------------------------------------------------------ levels

def fig_levels(mode, extra=None):
    """Candidates and support per level, relative residual per level. extra: {label: levels list}."""
    m = fs.setup(mode)
    sets = {label: levels(s) for s, label in SAMPLES.items()}
    if extra:
        sets.update(extra)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (label, L) in enumerate(sets.items()):
        col = m["series"][fs.DATASET_SLOTS[i]]
        h = [float(l["h"]) for l in L]
        ax[0].plot(h, [len(l["q"]) for l in L], "o-", color=col, label=f"{label}: candidates")
        sup = [int((l["w"] > 1e-2 * l["w"].max()).sum()) for l in L]
        ax[0].plot(h, sup, "s--", color=col, ms=8, lw=1.5, label=f"{label}: support (> 1 % of max)")
        ax[1].plot(h, [float(l["res"]) for l in L], "o-", color=col, label=label)
    for a in ax:
        a.set_xscale("log")
        a.invert_xaxis()
        a.set_xlabel("grid spacing of the level (deg)")
        a.set_xticks([4, 2, 1, 0.5, 0.25], ["4", "2", "1", "0.5", "0.25"])
        a.minorticks_off()
    ax[0].set_yscale("log")
    ax[0].set_ylabel("orientations")
    ax[0].set_title("Orientations per level")
    ax[1].set_ylabel("‖w (A x − b)‖ / ‖w b‖")
    ax[1].set_title("Relative residual of the bulk fit")
    ax[0].legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fs.save(fig, "levels", mode)


# ------------------------------------------------------------------ coverage

def coverage(sample):
    path = os.path.join(CACHE, f"coverage_{sample}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    gt = io.read_ground_truth(sample)
    idx = np.random.default_rng(0).choice(len(gt["orientation"]), 20000, replace=False)
    q_gt = Rotation.from_matrix(gt["orientation"][idx]).as_quat()
    bases = {"indexed": np.load(paths.basis_path(sample)),
             "uniform": grid_mats(os.path.join(PROC, sample, "reconstruction_uniform.h5")),
             "odf": grid_mats(os.path.join(PROC, sample, "reconstruction_odf.h5"))}
    out = {}
    for k, U in bases.items():
        out[k] = nearest_misorientation_deg(q_gt, Rotation.from_matrix(U).as_quat(), SYM)
        out[f"K_{k}"] = np.array(len(U))
    np.savez(path, **out)
    return out


NAMES = {"indexed": "indexed basis", "uniform": "uniform grid", "odf": "basis from the ODF"}


def fig_coverage(mode):
    m = fs.setup(mode)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for a, (sample, label) in zip(ax, SAMPLES.items()):
        cov = coverage(sample)
        for k in ("indexed", "uniform", "odf"):
            v = np.sort(cov[k])
            a.plot(v, np.arange(1, len(v) + 1) / len(v), color=fs.color(mode, k),
                   label=f"{NAMES[k]} (K = {int(cov['K_' + k]):,}): median {np.median(v):.2f}°")
        a.set_xlim(0, 2.0)
        a.set_ylim(0, 1.01)
        a.set_xlabel("ground truth → nearest basis orientation (deg)")
        a.set_title(label)
        a.legend(fontsize=9, loc="lower right")
    ax[0].set_ylabel("fraction of the ground truth")
    fig.tight_layout()
    return fs.save(fig, "coverage", mode)


# ------------------------------------------------------------------ maps

def medoid(sample, name):
    return np.load(os.path.join(TT, f"medoid_{sample}_{name}.npz"))["medoid"]


def fig_maps(mode, sample, vmax=20.0):
    m = fs.setup(mode)
    grid = stability.display_grid(sample, 4)
    names = [("indexed", "published_adaptive"), ("uniform", "published_uniform"), ("odf", "notebook_odf")]
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.2))
    ax[0, 0].imshow(stability.ipf_image(grid["gt"], grid["gt_valid"]), interpolation="nearest")
    ax[0, 0].set_title("ground truth (IPF-Z)")
    ax[1, 0].set_axis_off()
    for c, (k, name) in enumerate(names, start=1):
        med = medoid(sample, name)
        ori, valid = figures.upsample(med, np.isfinite(med[..., 0, 0]), 4)
        valid &= grid["gt_valid"]
        mis = figures.misorientation_map(grid["gt"], ori, valid)
        ax[0, c].imshow(stability.ipf_image(ori, valid), interpolation="nearest")
        ax[0, c].set_title(f"{NAMES[k]} (IPF-Z)")
        ax[1, c].imshow(stability.misorientation_image(mis, valid, vmax), interpolation="nearest")
        ax[1, c].set_title(f"misorientation to GT: median {np.nanmedian(mis):.3f}°", fontsize=11)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    stability.misorientation_colorbar(fig, list(ax[1, 1:]), vmax)
    fig.suptitle(f"{SAMPLES[sample]}: weighted-medoid orientation per voxel", fontsize=13)
    return fs.save(fig, f"maps_{sample}", mode)


# ------------------------------------------------------------------ accuracy versus K

def results(sample):
    out = []
    for f in glob.glob(os.path.join(TT, f"{sample}_*.json")):
        d = json.load(open(f))
        name = d["name"]
        if name == "published_adaptive":
            d["K"], kind = len(np.load(paths.basis_path(sample))), "indexed"
        elif name == "published_uniform":
            d["K"], kind = len(grid_mats(os.path.join(PROC, sample, "reconstruction_uniform.h5"))), "uniform"
        else:
            kind = "odf"
            if "K" not in d:
                d["K"] = len(grid_mats(os.path.join(PROC, sample, "reconstruction_odf.h5")))
        d["kind"] = kind
        out.append(d)
    return out


def label_of(d):
    n = d["name"]
    if n == "notebook_odf":
        return "notebook (4° start, 0.25°)"
    if n.startswith("odf_a"):
        return "0.5° level"
    lev = {"L2": "0.5°", "L3": "0.25°", "L4": "0.125°"}[n[:2]]
    if "sample" in n:
        return f"{lev} level, sampled"
    return f"{lev} level, > {n.split('_')[1][1:]} max, thin {n.split('_d')[-1]}°"


def fig_accuracy(mode):
    m = fs.setup(mode)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, (sample, label) in zip(ax, SAMPLES.items()):
        rs = results(sample)
        for k in ("indexed", "uniform", "odf"):
            pts = [d for d in rs if d["kind"] == k]
            a.scatter([d["K"] for d in pts], [d["mis_median"] for d in pts], s=70 if k != "odf" else 50,
                      color=fs.color(mode, k), edgecolor=m["surface"], linewidth=2, zorder=3, label=NAMES[k])
            for d in pts:
                if d["name"] == "notebook_odf":
                    a.scatter([d["K"]], [d["mis_median"]], s=260, facecolor="none", edgecolor=fs.color(mode, "odf"),
                              linewidth=2, zorder=4)
                    a.annotate("textomo_odf.ipynb", (d["K"], d["mis_median"]), textcoords="offset points",
                               xytext=(0, -22), ha="center", fontsize=9, color=m["text2"])
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("orientations in the basis, K")
        a.set_title(label)
        a.set_yticks([0.2, 0.3, 0.5, 1.0, 1.5], ["0.2", "0.3", "0.5", "1.0", "1.5"])
        a.minorticks_off()
    ax[0].set_ylabel("median misorientation to GT (deg)")
    ax[0].legend(fontsize=9, loc="upper left", markerscale=1.0)
    fig.tight_layout()
    return fs.save(fig, "accuracy_vs_K", mode)


if __name__ == "__main__":
    for mode in ("light", "dark"):
        print(fig_levels(mode, extra={"Al1050": _levels(np.load(os.path.join(HERE, "levels_al.npz")))}))
        print(fig_pole_figures(mode))
        print(fig_coverage(mode))
        print(fig_accuracy(mode))
        for s in SAMPLES:
            print(fig_maps(mode, s))
