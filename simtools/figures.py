"""
Orientation maps and figure panels for the simulated samples: ground truth, point-by-point
(pbp) and texture tomography (TT) maps on a common upsampled grid, IPF colours,
misorientation to the ground truth, grain and subgrain boundaries, zoom windows.

Conventions (as in the article's figures):
  * all maps live on a grid of UPSAMPLE x 99 pixels per side, centred on the rotation axis;
    reconstructions are upsampled by nearest neighbour, the ground truth is interpolated
    from the simulation mesh at that resolution;
  * maps are flipped [::-1, ::-1] when drawn.
"""
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from orix import plot as orix_plot
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt
from scipy.ndimage import label as ndimage_label
from scipy.spatial.transform import Rotation as R

from . import io
from .orientations import cubic_symmetry_operators

DPI = 300
SCALEBAR_X = 0.18  # left end of the scale bar, as a fraction of the map width (clear of the zoom lines)

# boundary colours
C_REC_MAIN = np.array([0.72, 0.11, 0.11])   # dark crimson
C_REC_SUB = np.array([0.95, 0.55, 0.50])    # light salmon-red
C_GT_MAIN = np.array([0.08, 0.25, 0.62])    # dark navy-blue
C_GT_SUB = np.array([0.50, 0.72, 0.94])     # light cornflower-blue


# --------------------------------------------------------------------------- orientation fields

def ipf_rgb(ori, axis=(0, 0, 1)):
    """IPF colours (TSL key, cubic) of orientation matrices (..., 3, 3) that map crystal to
    sample coordinates; NaN orientations give NaN colours."""
    shape = ori.shape[:-2]
    u = np.nan_to_num(ori.reshape(-1, 3, 3), nan=0.0).transpose(0, 2, 1)
    bad = ~np.isfinite(ori.reshape(-1, 9)).all(axis=1)
    u[bad] = np.eye(3)
    key = orix_plot.IPFColorKeyTSL(symmetry.Oh, direction=Vector3d(np.asarray(axis, dtype=float)))
    rgb = key.orientation2color(Orientation.from_matrix(u, symmetry.Oh)).reshape(*shape, 3)
    rgb[bad.reshape(shape)] = np.nan
    return rgb


def rgba(rgb, mask):
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    out[..., :3] = np.nan_to_num(rgb, nan=0.0)
    out[..., 3] = mask.astype(np.float32)
    return out


def ground_truth_map(sample, n_pixels, step):
    """Ground-truth orientations interpolated from the simulation mesh onto an
    (n_pixels x n_pixels) grid with spacing `step`, centred at the origin.
    Nearest-neighbour values; pixels outside the mesh's convex hull are NaN."""
    gt = io.read_ground_truth(sample)
    points = gt["centroids"][:, :2]
    half = (n_pixels - 1) / 2.0 * step
    xg = np.linspace(-half, half, n_pixels)
    xi, yi = np.meshgrid(xg, xg, indexing="ij")
    outside = np.isnan(griddata(points, gt["orientation"][:, 0, 0], (xi, yi), method="linear", fill_value=np.nan))
    ori = griddata(points, gt["orientation"], (xi, yi), method="nearest")
    ori[outside] = np.nan
    return ori, np.isfinite(ori[..., 0, 0])


def upsample(ori, valid, factor):
    """Nearest-neighbour upsampling of an orientation field and its mask."""
    up = np.repeat(np.repeat(ori, factor, axis=0), factor, axis=1)
    up_valid = np.repeat(np.repeat(valid, factor, axis=0), factor, axis=1)
    return up, up_valid


def weighted_medoid(coeffs, grid_mats, valid, top_k=24):
    """
    Per pixel, the orientation among the top_k largest coefficients that minimises the
    coefficient-weighted sum of misorientations to the others (cubic symmetry).
    coeffs: (Ny, Nx, K); returns (Ny, Nx, 3, 3), NaN where not valid or empty.
    """
    Ny, Nx, K = coeffs.shape
    sym = cubic_symmetry_operators()
    out = np.full((Ny, Nx, 3, 3), np.nan)
    t0 = time.time()
    for iy in range(Ny):
        for ix in range(Nx):
            if not valid[iy, ix]:
                continue
            w = coeffs[iy, ix]
            if w.max() <= 0:
                continue
            top = np.argpartition(w, -top_k)[-top_k:] if K > top_k else np.arange(K)
            top = top[w[top] > 0]
            if len(top) <= 1:
                out[iy, ix] = grid_mats[np.argmax(w)]
                continue
            mats = grid_mats[top]
            equiv = np.einsum("kij,sjl->ksil", mats, sym)             # (n, 24, 3, 3)
            traces = np.einsum("iab,jsab->ijs", mats, equiv)
            angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)).min(axis=-1)
            out[iy, ix] = mats[np.argmin(angles @ w[top])]
    print(f"  weighted medoid of {int(valid.sum())} pixels in {time.time() - t0:.0f} s")
    return out


def load_tt_medoid(reconstruction_h5, valid, cache, top_k=24):
    """Weighted medoid of a TT reconstruction, cached next to the figures (the cache is
    reused only if it was made from the same file, same modification time)."""
    import os
    stamp = f"{os.path.abspath(reconstruction_h5)}:{os.path.getmtime(reconstruction_h5)}:{top_k}"
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=False)
        if str(c["stamp"]) == stamp:
            print(f"  medoid from cache {os.path.basename(cache)}")
            return c["medoid"]
    with h5py.File(reconstruction_h5, "r") as f:
        grid_mats = f["grid_mats"][...]
        coeffs = f["coeffs"][...]
    medoid = weighted_medoid(coeffs, grid_mats, valid, top_k=top_k)
    np.savez(cache, medoid=medoid, stamp=stamp)
    return medoid


def misorientation_map(gt, rec, valid):
    """Misorientation angle (deg) between two orientation fields, cubic symmetry."""
    out = np.full(valid.shape, np.nan)
    if valid.any():
        a = Orientation.from_matrix(gt[valid].transpose(0, 2, 1), symmetry.Oh)
        b = Orientation.from_matrix(rec[valid].transpose(0, 2, 1), symmetry.Oh)
        out[valid] = np.degrees(np.asarray(a.angle_with(b)).ravel())
    return out


def kam(ori, valid):
    """Kernel average misorientation (deg): mean misorientation to the 4-connected neighbours."""
    Ny, Nx = valid.shape
    acc = np.zeros((Ny, Nx))
    cnt = np.zeros((Ny, Nx), dtype=int)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        yc = slice(max(dy, 0), Ny + min(dy, 0)); yn = slice(max(-dy, 0), Ny + min(-dy, 0))
        xc = slice(max(dx, 0), Nx + min(dx, 0)); xn = slice(max(-dx, 0), Nx + min(-dx, 0))
        both = valid[yc, xc] & valid[yn, xn]
        if not both.any():
            continue
        a = Orientation.from_matrix(ori[yc, xc][both].transpose(0, 2, 1), symmetry.Oh)
        b = Orientation.from_matrix(ori[yn, xn][both].transpose(0, 2, 1), symmetry.Oh)
        region = np.zeros(both.shape)
        region[both] = np.degrees(np.asarray(a.angle_with(b)).ravel())
        acc[yc, xc] += region
        cnt[yc, xc] += both
    out = np.full((Ny, Nx), np.nan)
    ok = (cnt > 0) & valid
    out[ok] = acc[ok] / cnt[ok]
    return out


def median_filter_orientations(ori, valid, kernel_size=5):
    """Quaternion median filter (component-wise median after aligning signs to the centre
    pixel); removes the ground truth's mesh noise while keeping boundaries sharp."""
    Ny, Nx = valid.shape
    pad = kernel_size // 2
    q = np.zeros((Ny, Nx, 4))
    q[valid] = R.from_matrix(ori[valid]).as_quat()
    qp = np.pad(q, ((pad, pad), (pad, pad), (0, 0)))
    vp = np.pad(valid, ((pad, pad), (pad, pad)))
    W = kernel_size * kernel_size
    wq = np.empty((Ny, Nx, W, 4))
    wv = np.empty((Ny, Nx, W), dtype=bool)
    i = 0
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            wq[:, :, i] = qp[dy:dy + Ny, dx:dx + Nx]
            wv[:, :, i] = vp[dy:dy + Ny, dx:dx + Nx]
            i += 1
    centre = wq[:, :, W // 2:W // 2 + 1]
    wq[np.sum(wq * centre, axis=-1) < 0] *= -1
    wq[~wv] = np.nan
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)  # pixels with no valid neighbours
        med = np.nanmedian(wq, axis=2)
    med /= np.where(np.linalg.norm(med, axis=-1, keepdims=True) < 1e-10, 1.0, np.linalg.norm(med, axis=-1, keepdims=True))
    out = np.full_like(ori, np.nan)
    ok = valid & np.all(np.isfinite(med), axis=-1)
    out[ok] = R.from_quat(med[ok]).as_matrix()
    return out, ok


def boundaries(ori, valid, region, threshold_deg):
    """Boundary pixels: KAM >= threshold, inside `region`."""
    k = kam(ori, valid)
    return np.isfinite(k) & valid & (k >= threshold_deg) & region


def eroded(mask, n_pixels):
    return binary_erosion(mask, structure=np.ones((2 * n_pixels + 1, 2 * n_pixels + 1), dtype=bool))


def orange_pixels(ori, hue=(0.01, 0.22), min_saturation=0.3, min_value=0.25):
    """Pixels whose IPF-Z colour is orange (hue, saturation and value within the bounds)."""
    valid = np.isfinite(ori[..., 0, 0])
    hsv = rgb_to_hsv(np.nan_to_num(ipf_rgb(ori, (0, 0, 1)), nan=0.0))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return valid & (h > hue[0]) & (h < hue[1]) & (s > min_saturation) & (v > min_value)


def orange_grain(gt_ori, hue=(0.01, 0.22), min_saturation=0.3, min_value=0.25):
    """Mask of the largest connected 'orange' region of the ground truth's IPF-Z map, holes
    filled (the grain shown enlarged in the article's figures). Use it on the voxel grid: at
    finer resolution, neighbouring orange grains can connect."""
    labels, n = ndimage_label(orange_pixels(gt_ori, hue, min_saturation, min_value))
    if n == 0:
        raise ValueError("no orange region found")
    largest = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    return binary_fill_holes(labels == largest)


def tight_bbox(mask):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return slice(int(r0), int(r1) + 1), slice(int(c0), int(c1) + 1)


def zoom_window(mask, margin=0.1):
    """Rectangle around a mask (its bounding box grown by `margin` of its size on every
    side, clipped to the map), as (rows, cols) slices on the unflipped map."""
    rows, cols = tight_bbox(mask)
    out = []
    for sl, n in [(rows, mask.shape[0]), (cols, mask.shape[1])]:
        pad = int(round(margin * (sl.stop - sl.start)))
        out.append(slice(max(sl.start - pad, 0), min(sl.stop + pad, n)))
    return tuple(out)


def _flip_window(window, shape):
    """The window's position after flipping the map [::-1, ::-1] (as it is drawn)."""
    return tuple(slice(n - sl.stop, n - sl.start) for sl, n in zip(window, shape))


# --------------------------------------------------------------------------- saving panels

def _save_image(image, path, pixel_um, scalebar_um, figsize=4, box=None):
    """Save an RGBA image (already flipped for display) without axes on a transparent
    background; optionally with a scale bar and a black rectangle `box` = (rows, cols)."""
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(image, interpolation="nearest")
    ax.axis("off")
    if box is not None:
        rows, cols = box
        ax.add_patch(plt.Rectangle((cols.start - 0.5, rows.start - 0.5), cols.stop - cols.start,
                                   rows.stop - rows.start, fill=False, edgecolor="black", linewidth=1.5))
    if scalebar_um:
        ax.add_artist(ScaleBar(pixel_um, "um", fixed_value=scalebar_um, location="lower left",
                               bbox_to_anchor=(SCALEBAR_X, 0), bbox_transform=ax.transAxes,
                               frameon=False, color="0.35", font_properties={"size": 12}))
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def save_ipf(path, ori, valid, pixel_um, scalebar_um=None, axis=(0, 0, 1), crop=None, box=None):
    """IPF map (flipped [::-1, ::-1] as in the article). `crop` = (rows, cols) slices of the
    unflipped map: save only that window; `box`: draw that window as a rectangle."""
    image = rgba(ipf_rgb(ori, axis), valid)
    if box is not None:
        box = _flip_window(box, image.shape[:2])
    if crop is not None:
        image = image[crop]
    _save_image(image[::-1, ::-1], path, pixel_um, scalebar_um, box=box)


def misorientation_colormap(vmax_deg):
    """Colormap and normalisation of the misorientation maps: jet on log10(0.1 + angle)."""
    norm = plt.Normalize(vmin=np.log10(0.1), vmax=np.log10(0.1 + vmax_deg))
    cmap = plt.cm.jet.copy()
    cmap.set_bad(alpha=0)
    return cmap, norm


def save_misorientation(path, misori, valid, pixel_um, vmax_deg, scalebar_um=None, crop=None):
    cmap, norm = misorientation_colormap(vmax_deg)
    data = np.log10(0.1 + np.where(valid, misori, np.nan))
    image = cmap(norm(data))
    image[~valid, 3] = 0
    if crop is not None:
        image = image[crop]
    _save_image(image[::-1, ::-1], path, pixel_um, scalebar_um)


def _tick_label(v):
    return "0°" if v < 0.05 else (f"{v:.1f}°" if v < 10 else f"{v:.0f}°")


def save_misorientation_colorbars(directory, vmax_deg):
    """Horizontal and vertical colorbars (with and without label) for the misorientation maps."""
    cmap, norm = misorientation_colormap(vmax_deg)
    ticks = np.linspace(norm.vmin, norm.vmax, 6)
    labels = [_tick_label(max(0.0, 10**t - 0.1)) for t in ticks]
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    for name, size, orientation, label in [("horizontal", (4, 0.5), "horizontal", True),
                                           ("vertical", (0.5, 4), "vertical", True),
                                           ("vertical_nolabel", (0.5, 4), "vertical", False)]:
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_alpha(0)
        cb = fig.colorbar(mappable, cax=ax, orientation=orientation)
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
        cb.ax.tick_params(labelsize=12)
        if label:
            cb.set_label("Misorientation (°)", fontsize=14)
        fig.savefig(f"{directory}/misorientation_colorbar_{name}.png", dpi=DPI, bbox_inches="tight",
                    pad_inches=0.05, transparent=True)
        plt.close(fig)


def grain_boundary_image(gt_main, rec_main=None):
    """RGBA image of grain boundaries only: reconstruction in red, ground truth in blue on top
    (the style of the article's 1 degree figure)."""
    img = np.zeros((*gt_main.shape, 4), dtype=np.float32)
    if rec_main is not None:
        img[rec_main] = [1, 0, 0, 1]
    img[gt_main] = [0, 0, 1, 1]
    return img


def boundary_image(gt_main, gt_sub, rec_main=None, rec_sub=None):
    """RGBA image of grain (main) and subgrain boundaries: subgrain layer first, main layer
    on top; within a layer, pixels on both the ground-truth and the reconstruction boundary
    get the mean of the two colours."""
    shape = gt_main.shape

    def layer(a, ca, b, cb):
        img = np.zeros((*shape, 4), dtype=np.float32)
        if b is None:
            img[a, :3], img[a, 3] = ca, 1
            return img
        for m, c in [(a & ~b, ca), (b & ~a, cb), (a & b, 0.5 * (ca + cb))]:
            img[m, :3], img[m, 3] = c, 1
        return img

    sub = layer(gt_sub, C_GT_SUB, rec_sub, C_REC_SUB)
    main = layer(gt_main, C_GT_MAIN, rec_main, C_REC_MAIN)
    out = sub.copy()
    top = main[..., 3] > 0
    out[top] = main[top]
    return out


def save_boundaries(path, image, pixel_um=None, scalebar_um=None):
    _save_image(image[::-1, ::-1], path, pixel_um, scalebar_um, figsize=6)


def save_boundary_legends(directory):
    """Legends of the two boundary styles: grain boundaries only (red / blue) and grain +
    subgrain boundaries (four colours); vertical and horizontal versions of each."""
    styles = {
        "main": [(np.array([0.0, 0.0, 1.0]), "Ground truth grain boundary"),
                 (np.array([1.0, 0.0, 0.0]), "Reconstruction grain boundary")],
        "main_sub": [(C_GT_MAIN, "Ground truth grain boundary"), (C_GT_SUB, "Ground truth subgrain boundary"),
                     (C_REC_MAIN, "Reconstruction grain boundary"), (C_REC_SUB, "Reconstruction subgrain boundary")],
    }
    for style, entries in styles.items():
        handles = [Line2D([0], [0], color=c, linewidth=3.5, solid_capstyle="butt") for c, _ in entries]
        kw = dict(handles=handles, labels=[l for _, l in entries], frameon=False, fontsize=12,
                  handlelength=2.4, handleheight=1.1, labelspacing=0.55, handletextpad=0.8)
        for name, size, ncol in [("vertical", (6.0, 0.7 * len(entries)), 1), ("horizontal", (3.5 * len(entries), 1.0), len(entries))]:
            fig, ax = plt.subplots(figsize=size)
            fig.patch.set_alpha(0)
            ax.set_visible(False)
            fig.legend(loc="center", ncol=ncol, **kw)
            fig.savefig(f"{directory}/legend_boundaries_{style}_{name}.png", dpi=DPI, bbox_inches="tight",
                        pad_inches=0.12, transparent=True)
            plt.close(fig)


def save_ipf_colorkey(path):
    fig = orix_plot.IPFColorKeyTSL(symmetry.Oh).plot(return_figure=True)
    fig.savefig(path, dpi=400, transparent=True)
    plt.close(fig)
