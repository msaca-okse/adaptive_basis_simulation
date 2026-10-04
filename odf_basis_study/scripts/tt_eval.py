"""TT reconstruction with a given basis (as textomo_adaptive.ipynb: sigma 0.4 deg, FISTA-Huber nonneg,
delta 100, 200 iterations, streamed) and its accuracy as in the accuracy table of the timing notes:
weighted medoid of the top 24 coefficients per voxel, upsampled 4x, misorientation to the ground truth.

usage: python tt_eval.py SAMPLE NAME BASIS_SPEC [sigma_deg]
  BASIS_SPEC: "adaptive" (basis.npy), "h5:<reconstruction file>" (evaluate only), or
              "odf:<npz>:<level>:<rel>:<dmin>" (support of the ODF level above rel * max, thinned at dmin
              deg in order of weight), "sample:<npz>:<level>:<M>:<dmin>[:alpha]" (M draws from w^alpha,
              jittered by the level's sigma, thinned at dmin)
"""
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/dtu-compute/msaca/adaptive_basis_simulations")
from simtools import figures, io, paths, stability  # noqa: E402
from diffractom import FISTAHuber, Grid, SinglePhaseForwardOperator, estimate_L_power_streamed  # noqa: E402
from odf import nearest_angle, prune, to_fz  # noqa: E402

OUT = "/dtu-compute/msaca/claude_scratch/odf/tt"
sample, name, spec = sys.argv[1], sys.argv[2], sys.argv[3]
sigma_deg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
os.makedirs(OUT, exist_ok=True)
h5 = os.path.join(OUT, f"{sample}_{name}.h5")
summary = {"sample": sample, "name": name, "spec": spec, "sigma_deg": sigma_deg}


def basis_from_spec(spec):
    kind, *a = spec.split(":")
    if kind == "adaptive":
        return np.load(paths.basis_path(sample))
    z = np.load(a[0])
    q, w = z[f"q{a[1]}"], z[f"w{a[1]}"]
    if kind == "odf":
        rel, dmin = float(a[2]), float(a[3])
        s = np.flatnonzero(w > rel * w.max())
        s = s[np.argsort(-w[s])]
        keep = s[prune(q[s], dmin)]
    elif kind == "sample":
        M, dmin = int(a[2]), float(a[3])
        alpha = float(a[4]) if len(a) > 4 else 1.0
        rng = np.random.default_rng(0)
        p = np.maximum(w, 0) ** alpha
        idx = rng.choice(len(q), M, p=p / p.sum())
        sig = np.deg2rad(0.6 * float(z[f"h{a[1]}"]))  # the level's kernel width
        jit = Rotation.from_rotvec(rng.normal(scale=sig, size=(M, 3)))
        qs = to_fz((Rotation.from_quat(q[idx]) * jit).as_quat())
        keep = prune(qs, dmin)
        return Rotation.from_quat(qs[keep]).as_matrix()
    return Rotation.from_quat(q[keep]).as_matrix()


if not spec.startswith("h5:"):
    basis = basis_from_spec(spec)
    p = stability.load_problem(sample)
    grid = Grid.from_rotation_matrices(basis, np.deg2rad(sigma_deg))
    t0 = time.perf_counter()
    op = SinglePhaseForwardOperator(cfg=p["cfg"], material=p["material"], grid=grid, max_gb=1.0, normalized=True,
                                    reserve_coefficient_arrays=0)
    L = 1.1 * estimate_L_power_streamed(op, niter=6, seed=0, verbose=0)
    t1 = time.perf_counter()
    x = np.zeros(op.coeff_shape, np.float32)
    x = FISTAHuber(op, prox_kind="nonneg", L=L, huber_delta=100.0).run(x, p["data"], niter=200, weights=p["weights"])
    t2 = time.perf_counter()
    r = (op.direct(x) - p["data"]) * p["weights"].ravel()
    w_data = p["data"] * p["weights"].ravel()
    summary.update(K=len(basis), pf_mode=op.pf_mode, build_s=t1 - t0, fista_s=t2 - t1,
                   residual=float(np.linalg.norm(r) / np.linalg.norm(w_data)))
    del r, w_data
    grid_mats = Rotation.concatenate(grid.rotations_at_level(0)).as_matrix()
    stability.save_reconstruction(h5, x, grid_mats, np.zeros(1), sample=sample, sigma_deg=sigma_deg)
    op.free_memory()
    print(f"K {len(basis)} pf {summary['pf_mode']} build {t1 - t0:.0f} s FISTA {t2 - t1:.0f} s "
          f"residual {summary['residual']:.4f}", flush=True)
    gt = io.read_ground_truth(sample)
    idx = np.random.default_rng(0).choice(len(gt["orientation"]), 20000, replace=False)
    a = nearest_angle(Rotation.from_matrix(gt["orientation"][idx]).as_quat(), Rotation.from_matrix(basis).as_quat())
    summary.update(gt_to_basis_median=float(np.median(a)), gt_to_basis_p99=float(np.percentile(a, 99)))
else:
    h5 = spec[3:]

grid_d = stability.display_grid(sample, 4)
ori, valid = stability.medoid_map(h5, grid_d, os.path.join(OUT, f"medoid_{sample}_{name}.npz"))
m = figures.misorientation_map(grid_d["gt"], ori, valid)
summary.update(mis_median=float(np.nanmedian(m)), mis_mean=float(np.nanmean(m)),
               mis_p90=float(np.nanpercentile(m, 90)), frac_gt5=float(np.nanmean(m[np.isfinite(m)] > 5)))
print(json.dumps(summary), flush=True)
with open(os.path.join(OUT, f"{sample}_{name}.json"), "w") as f:
    json.dump(summary, f)
