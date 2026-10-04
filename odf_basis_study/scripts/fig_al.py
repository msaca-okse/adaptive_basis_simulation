"""Figures of the Al1050 data: IPF maps (indexed, uniform, ODF basis), misorientation between the ODF and the
indexed maps, {111} pole figures of the ODF levels, coverage of the indexed basis by the ODF basis.
Maps as visualization/paper_figures.ipynb: density mask (0.3 x 99th percentile), 5-pixel edge crop,
weighted medoid of the top 24 coefficients.
usage: python fig_al.py RECONSTRUCTION_ODF_H5"""
import json
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/dtu-compute/msaca/adaptive_basis_simulations")
from simtools import figures  # noqa: E402
from diffractom.odf_basis import nearest_misorientation_deg, symmetry_rotations  # noqa: E402

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from fig_sim import pole_figure, NAMES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AL = "/dtu-compute/msaca/claude_scratch/odf_al"
PROCESS = "/dtu/3d-imaging-center/projects/2025_QIM_BlackBeauty/raw_data_extern/2025_Danmax_Al1050/process"
CACHE = os.path.join(HERE, "cache")
SYM = symmetry_rotations("cubic")
MASK_FRACTION, EDGE_CROP, TOP_K = 0.3, 5, 24


def load_map(name, h5):
    """Medoid orientation per pixel (map orientation of the paper figures), the mask, the density."""
    cache = os.path.join(CACHE, f"al_medoid_{name}.npz")
    stamp = f"{h5}:{os.path.getmtime(h5)}"
    if os.path.exists(cache):
        c = np.load(cache)
        if str(c["stamp"]) == stamp:
            return c["medoid"], c["keep"], c["density"]
    with h5py.File(h5, "r") as f:
        coeffs = f["coeffs"][...].transpose(2, 1, 0)[::-1].astype(np.float64)
        mats = f["grid_mats"][...]
    density = coeffs.sum(axis=-1)
    keep = density > MASK_FRACTION * np.percentile(density, 99)
    keep[:EDGE_CROP] = keep[-EDGE_CROP:] = False
    keep[:, :EDGE_CROP] = keep[:, -EDGE_CROP:] = False
    medoid = figures.weighted_medoid(coeffs, mats, keep, top_k=TOP_K)
    np.savez(cache, medoid=medoid, keep=keep, density=density, stamp=stamp)
    return medoid, keep, density


def ipf(medoid, keep):
    rgba = figures.rgba(figures.ipf_rgb(medoid), keep)
    return rgba


def fig_maps(mode, odf_h5, vmax=20.0):
    m = fs.setup(mode)
    maps = {"indexed": load_map("indexed", os.path.join(PROCESS, "reconstruction_adaptive.h5")),
            "uniform": load_map("uniform", os.path.join(PROCESS, "reconstruction_uniform.h5")),
            "odf": load_map("odf", odf_h5)}
    fig, ax = plt.subplots(2, 3, figsize=(15, 10.4))
    for c, (k, (med, keep, dens)) in enumerate(maps.items()):
        ax[0, c].imshow(ipf(med, keep), interpolation="nearest")
        ax[0, c].set_title(f"{NAMES[k]} (IPF-Z)")
    med_i, keep_i, _ = maps["indexed"]
    stats = {}
    for c, k in enumerate(("uniform", "odf"), start=1):
        med, keep, _ = maps[k]
        both = keep & keep_i
        mis = figures.misorientation_map(med_i, med, both)
        stats[k] = {"median": float(np.nanmedian(mis)), "mean": float(np.nanmean(mis)),
                    "frac_lt1": float(np.nanmean(mis[both] < 1.0)), "frac_gt5": float(np.nanmean(mis[both] > 5.0))}
        ax[1, c].imshow(stability_mis_image(mis, both, vmax), interpolation="nearest")
        short = {"uniform": "uniform", "odf": "ODF basis"}[k]
        ax[1, c].set_title(f"{short} vs indexed map: median {stats[k]['median']:.2f}°", fontsize=11)
    dens = maps["indexed"][2]
    im = ax[1, 0].imshow(np.where(keep_i, dens, np.nan), cmap=fs.seq_cmap(mode), interpolation="nearest")
    ax[1, 0].set_title("density (indexed basis)")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    from simtools import stability
    stability.misorientation_colorbar(fig, list(ax[1, 1:]), vmax)
    fig.suptitle("Al1050 (15 % deformed): weighted-medoid orientation per pixel", fontsize=13)
    json.dump(stats, open(os.path.join(CACHE, "al_map_stats.json"), "w"), indent=1)
    return fs.save(fig, "maps_al", mode), stats


def stability_mis_image(mis, valid, vmax):
    cmap, norm = figures.misorientation_colormap(vmax)
    image = cmap(norm(np.log10(0.1 + np.where(valid, mis, np.nan))))
    image[~valid, 3] = 0
    return image


def fig_pole_figures(mode, npz):
    m = fs.setup(mode)
    cmap = fs.seq_cmap(mode)
    cmap.set_bad(m["surface"])
    z = np.load(npz)
    n = len([k for k in z if k.startswith("q")])
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.9))
    for c in range(n):
        ax = axes[c]
        ax.imshow(pole_figure(z[f"q{c}"], z[f"w{c}"]), origin="lower", cmap=cmap, norm=PowerNorm(0.5, 0, 1),
                  extent=[-1, 1, -1, 1])
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=m["text2"], lw=0.8))
        ax.set_axis_off()
        ax.set_title(f"{float(z[f'h{c}']):g}° grid · {len(z[f'q{c}']):,}", fontsize=10, color=m["text2"])
    fig.suptitle("Al1050: {111} pole figure of the bulk ODF at each level (rotation axis in the centre)", fontsize=12)
    return fs.save(fig, "pole_figures_al", mode)


def fig_coverage(mode, odf_h5, indexed_basis):
    m = fs.setup(mode)
    with h5py.File(odf_h5, "r") as f:
        q_odf = Rotation.from_matrix(f["grid_mats"][...]).as_quat()
    q_ix = Rotation.from_matrix(indexed_basis).as_quat()
    rng = np.random.default_rng(0)
    a = nearest_misorientation_deg(q_ix[rng.choice(len(q_ix), min(20000, len(q_ix)), replace=False)], q_odf, SYM)
    b = nearest_misorientation_deg(q_odf[rng.choice(len(q_odf), min(20000, len(q_odf)), replace=False)], q_ix, SYM)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for v, k, lab in [(a, "indexed", f"indexed orientation → nearest ODF-basis orientation (median {np.median(a):.2f}°)"),
                      (b, "odf", f"ODF-basis orientation → nearest indexed orientation (median {np.median(b):.2f}°)")]:
        v = np.sort(v)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=fs.color(mode, k), label=lab)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("misorientation (deg)")
    ax.set_ylabel("fraction")
    ax.set_title(f"Al1050: indexed basis (K = {len(q_ix):,}) and ODF basis (K = {len(q_odf):,})")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fs.save(fig, "coverage_al", mode), {"ix_to_odf_median": float(np.median(a)),
                                              "odf_to_ix_median": float(np.median(b))}


if __name__ == "__main__":
    odf_h5 = sys.argv[1]
    b0 = np.load("/zhome/71/c/146676/texture_tomography/adaptive_basis/basis.npy")
    basis = Rotation.from_euler("z", 50.0, degrees=True).as_matrix() @ b0  # into the operator frame
    out = {}
    for mode in ("light", "dark"):
        print(fig_pole_figures(mode, os.path.join(AL, "odf_al.npz")))
        path, out["coverage"] = fig_coverage(mode, odf_h5, basis)
        print(path)
        path, out["maps"] = fig_maps(mode, odf_h5)
        print(path)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(CACHE, "al_stats.json"), "w"), indent=1)
