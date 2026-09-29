"""
Stability analysis of adaptive-basis texture tomography: reconstructions for a sweep of one
parameter (the number of FISTA iterations, or the kernel width sigma), and their comparison
with the ground truth and with each other.

The reconstruction is the one of texture_tomography/<sample>/textomo_adaptive.ipynb (same data
normalisation, weights, geometry, solver); only the swept parameter changes. The evaluation
uses the maps of the figure notebooks (simtools.figures): per voxel the weighted medoid
orientation of the TT coefficients, upsampled onto the display grid of the ground truth.
"""
import os
import time

import h5py
import numpy as np
import pandas as pd
import pyopencl.array as clarray
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

from diffractom import FISTAHuber, Grid, Material, SinglePhaseForwardOperator
from diffractom.operators.single_phase_forward_operator import estimate_L_power, group_reflections_into_rings
from integration.frame_loader import output_path
from integration.integrated_data import IntegratedData

from . import figures, io, paths


# --------------------------------------------------------------------------- reconstruction

def load_problem(sample):
    """Data, weights, geometry, material and adaptive basis of a sample, exactly as in
    texture_tomography/<sample>/textomo_adaptive.ipynb."""
    data = IntegratedData(output_path(sample))
    assert data.polarization_corrected is True, f"{data.json_path}: the data must be polarization corrected"
    arr = np.ascontiguousarray(np.asarray(data.I, dtype=np.float64).transpose(1, 0, 2, 3))  # (N_Omega, My, N_eta, N_theta)
    N_Omega, My, N_eta, N_theta = arr.shape
    arr = arr / arr.sum(axis=(0, 1, 2), keepdims=True) * arr.sum()  # equal total intensity per ring

    weights = np.ones_like(arr, dtype=np.float32)
    weights[:, :, 40:50, :] = 0    # eta bins along the rotation axis
    weights[:, :, 130:140, :] = 0

    cfg = {
        "energy": 12.398 / data.wavelength_A, "Nx": 99, "Ny": 99, "My": My, "N_Omega": N_Omega, "N_eta": N_eta,
        "angle_range": [0, 180], "cor_offset": 0, "eta_angle_range": [0, 360],
        "j_direction_0": [0, -1, 0], "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
        "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1],
    }
    lattice = data.material["lattice_params"]
    material = Material.from_lattice_parameters(
        a=lattice["a"], b=lattice["b"], c=lattice["c"],
        alpha=lattice["alpha"], beta=lattice["beta"], gamma=lattice["gamma"],
        symmetry_group="cubic", wavelength_A=data.wavelength_A, min_two_theta=0.0, max_two_theta=0.0,
        hkl_list=np.array([hkl for ring in data.rings for hkl in ring["hkl"]]),
    )
    rings = group_reflections_into_rings(material)
    ring_two_theta = np.array([material.reflections["two_theta"][r[0]] for r in rings])
    assert len(rings) == N_theta and np.allclose(ring_two_theta, data.two_theta_rad, rtol=1e-4), \
        "material rings do not match the integrated rings"
    return {
        "sample": sample,
        "data": arr.reshape(N_Omega, My, N_eta * N_theta).astype(np.float32),
        "weights": weights.reshape(N_Omega, My, N_eta * N_theta),
        "cfg": cfg,
        "material": material,
        "basis": np.load(paths.basis_path(sample)),
    }


class Reconstructor:
    """Forward operator for one kernel width sigma, with the data and weights on the GPU;
    `run(niter)` gives an unregularised nonnegative FISTA-Huber reconstruction from zero."""

    def __init__(self, problem, sigma_deg, max_gb=1.0):
        self.sigma_deg = float(sigma_deg)
        grid = Grid.from_rotation_matrices(problem["basis"], np.deg2rad(sigma_deg))
        self.grid_mats = R.concatenate(grid.rotations_at_level(0)).as_matrix()
        self.K = len(self.grid_mats)
        self.op = SinglePhaseForwardOperator(cfg=problem["cfg"], material=problem["material"], grid=grid,
                                             max_gb=max_gb, normalized=True)
        q = self.op.queue
        self.y = clarray.to_device(q, problem["data"])
        self.w = clarray.to_device(q, problem["weights"])
        wy = self.y * self.w
        self.data_norm = float(np.sqrt(clarray.vdot(wy, wy).get().real))
        self.L = 1.1 * estimate_L_power(self.op, niter=6, seed=0, eps=1e-30, verbose=0)

    def run(self, niter, huber_delta=100.0):
        """Returns coeffs (Ny, Nx, K) flipped to the ground-truth convention (as the TT
        notebooks save them), the objective per iteration, the relative weighted residual
        ||w (A x - y)|| / ||w y|| of the result, and the run time."""
        op = self.op
        x = clarray.zeros(op.queue, (op.Nx, op.Ny, self.K), np.float32, order="F")
        solver = FISTAHuber(op, prox_kind="nonneg", L=self.L, huber_delta=huber_delta)
        t0 = time.perf_counter()
        solver.run(x, self.y, niter=int(niter), weights=self.w, verbose=0)
        seconds = time.perf_counter() - t0
        r = (op.direct(x) - self.y) * self.w
        residual = float(np.sqrt(clarray.vdot(r, r).get().real)) / self.data_norm
        coeffs = x.get()[::-1].copy()
        objective = np.array([s["f"] for s in solver.iter_stats])
        del x, r
        return coeffs, objective, residual, seconds

    def free(self):
        self.op.free_memory()


def save_reconstruction(path, coeffs, grid_mats, objective, **attrs):
    """Same layout as the TT notebooks' reconstruction files (read by figures.load_tt_medoid)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("grid_mats", data=grid_mats, compression="gzip")
        ds = f.create_dataset("coeffs", data=coeffs, compression="gzip")
        ds.attrs["axes"] = "(Ny, Nx, K)"
        f.create_dataset("objective", data=objective)
        f.attrs["grid"] = "adaptive"
        f.attrs["K"] = len(grid_mats)
        for key, value in attrs.items():
            f.attrs[key] = value


def read_attrs(path):
    with h5py.File(path, "r") as f:
        return dict(f.attrs), f["objective"][...]


def read_coeffs(path):
    with h5py.File(path, "r") as f:
        return f["coeffs"][...]


# --------------------------------------------------------------------------- evaluation

def display_grid(sample, upsample):
    """Voxel size, display pixel size and the ground truth on the display grid (as in the
    figure notebooks)."""
    with h5py.File(io.scan_files(sample)[0]) as f0, h5py.File(io.scan_files(sample)[1]) as f1:
        step_um = float(f1[io.scan_key(f1)]["eiger/frames/y"][0] - f0[io.scan_key(f0)]["eiger/frames/y"][0])
    pixel_um = step_um / upsample
    gt, gt_valid = figures.ground_truth_map(sample, 99 * upsample, pixel_um)
    return {"step_um": step_um, "pixel_um": pixel_um, "upsample": upsample,
            "gt": gt, "gt_valid": gt_valid, "gt_valid_voxels": gt_valid[::upsample, ::upsample]}


def medoid_map(h5, grid, cache, top_k=24):
    """Weighted medoid orientation map of a reconstruction on the display grid (cached)."""
    medoid = figures.load_tt_medoid(h5, grid["gt_valid_voxels"], cache, top_k=top_k)
    ori, valid = figures.upsample(medoid, np.isfinite(medoid[..., 0, 0]), grid["upsample"])
    return ori, valid & grid["gt_valid"]


def misorientation_to_ground_truth(grid, maps):
    """Misorientation maps to the ground truth and their statistics (deg)."""
    misori, rows = {}, []
    for name, (ori, valid) in maps.items():
        m = figures.misorientation_map(grid["gt"], ori, valid)
        misori[name] = m
        rows.append({"map": name, "mean (deg)": np.nanmean(m), "median (deg)": np.nanmedian(m),
                     "90th percentile (deg)": np.nanpercentile(m, 90),
                     "fraction > 5 deg": np.nanmean(m[valid] > 5.0)})
    return misori, pd.DataFrame(rows).set_index("map")


def boundary_analysis(grid, maps, gb_threshold, subgrain_threshold, gt_median_kernel=5):
    """
    Grain (KAM >= gb_threshold) and subgrain (KAM >= subgrain_threshold) boundaries of the
    ground truth (median filtered) and of every map, inside the sample eroded by one voxel.
    Statistics per map and kind, distances in micrometres:
      * deviation: distance from each reconstructed boundary pixel to the nearest ground-truth
        boundary of the same kind (as in the figure notebooks; low = no spurious boundaries);
      * gt -> rec: distance from each ground-truth boundary pixel to the nearest reconstructed
        one (low = no missing boundaries).
    """
    pixel_um = grid["pixel_um"]
    region = figures.eroded(grid["gt_valid"], grid["upsample"])
    gt_f, gt_f_valid = figures.median_filter_orientations(grid["gt"], grid["gt_valid"], gt_median_kernel)
    gt_b = {"grain": figures.boundaries(gt_f, gt_f_valid, region, gb_threshold),
            "subgrain": figures.boundaries(gt_f, gt_f_valid, region, subgrain_threshold)}
    gt_dist = {kind: distance_transform_edt(~b) for kind, b in gt_b.items()}
    rec_b, rows = {}, []
    for name, (ori, valid) in maps.items():
        rec_b[name] = {"grain": figures.boundaries(ori, valid, region, gb_threshold),
                       "subgrain": figures.boundaries(ori, valid, region, subgrain_threshold)}
        for kind in ("grain", "subgrain"):
            rec = rec_b[name][kind]
            d = gt_dist[kind][rec] * pixel_um
            back = distance_transform_edt(~rec)[gt_b[kind]] * pixel_um if rec.any() else np.array([np.nan])
            rows.append({"map": name, "boundary": kind, "pixels": int(rec.sum()),
                         "pixels (gt)": int(gt_b[kind].sum()),
                         "mean deviation (um)": d.mean() if len(d) else np.nan,
                         "median deviation (um)": np.median(d) if len(d) else np.nan,
                         "mean gt -> rec (um)": back.mean(),
                         "median gt -> rec (um)": np.median(back)})
    return gt_b, rec_b, pd.DataFrame(rows)


def pairwise_misorientation(maps):
    """Mean and median misorientation (deg) between the medoid maps of every pair."""
    names = list(maps)
    mean = pd.DataFrame(0.0, index=names, columns=names)
    median = mean.copy()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = maps[a][1] & maps[b][1]
            m = figures.misorientation_map(maps[a][0], maps[b][0], both)
            mean.loc[a, b] = mean.loc[b, a] = np.nanmean(m)
            median.loc[a, b] = median.loc[b, a] = np.nanmedian(m)
    return mean, median


def pairwise_coefficient_difference(h5_paths):
    """Relative coefficient difference ||x_i - x_j|| / ||x_j|| (row i, column j); needs the
    same basis orientations in every reconstruction."""
    names = list(h5_paths)
    coeffs = {n: read_coeffs(p) for n, p in h5_paths.items()}
    shapes = {c.shape for c in coeffs.values()}
    assert len(shapes) == 1, f"the reconstructions have different shapes: {shapes}"
    norms = {n: np.linalg.norm(c) for n, c in coeffs.items()}
    out = pd.DataFrame(0.0, index=names, columns=names)
    for a in names:
        for b in names:
            if a != b:
                out.loc[a, b] = np.linalg.norm(coeffs[a] - coeffs[b]) / norms[b]
    return out


# --------------------------------------------------------------------------- drawing (flipped [::-1, ::-1], as in the article)

def ipf_image(ori, valid):
    return figures.rgba(figures.ipf_rgb(ori), valid)[::-1, ::-1]


def misorientation_image(misori, valid, vmax_deg):
    cmap, norm = figures.misorientation_colormap(vmax_deg)
    image = cmap(norm(np.log10(0.1 + np.where(valid, misori, np.nan))))
    image[~valid, 3] = 0
    return image[::-1, ::-1]


def misorientation_colorbar(fig, ax, vmax_deg):
    import matplotlib.pyplot as plt
    cmap, norm = figures.misorientation_colormap(vmax_deg)
    ticks = np.linspace(norm.vmin, norm.vmax, 6)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.02, pad=0.01)
    cb.set_ticks(ticks)
    cb.set_ticklabels([figures._tick_label(max(0.0, 10**t - 0.1)) for t in ticks])
    cb.set_label("misorientation")
    return cb


def boundary_overlay(gt_b, rec_b):
    """Grain + subgrain boundaries of a reconstruction over the ground truth's (four colours)."""
    return figures.boundary_image(gt_b["grain"], gt_b["subgrain"], rec_b["grain"], rec_b["subgrain"])[::-1, ::-1]
