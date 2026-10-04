"""Al1050: TT with the ODF basis (as textomo_adaptive.ipynb otherwise), and the weighted data residual of
the ODF, indexed (adaptive) and uniform reconstructions.
usage: python al_tt.py ODF_NPZ [rel] [min_distance_deg] [sigma_deg] [level]"""
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from al_problem import load
from diffractom import FISTAHuber, Grid, MatrixGrid, SinglePhaseForwardOperator, estimate_L_power_streamed
from diffractom.odf_basis import ODFLevel, basis_from_odf
from integration.frame_loader import PROCESS

npz = sys.argv[1]
rel = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
dmin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
sigma_deg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
lev_arg = int(sys.argv[5]) if len(sys.argv) > 5 else None
tag = f"L{lev_arg}_r{rel:g}_d{dmin:g}_s{sigma_deg:g}"
HERE = os.path.dirname(os.path.abspath(__file__))
out_h5 = os.path.join(HERE, f"reconstruction_odf_{tag}.h5")

p = load()
b, weights, cfg, mat = p["data"], p["weights"], p["cfg"], p["material"]
wseg = weights.ravel()
wb_norm = float(np.linalg.norm(b * wseg))
z = np.load(npz)
last = max(int(k[1:]) for k in z if k.startswith("q")) if lev_arg is None else lev_arg
level = ODFLevel(q=z[f"q{last}"], w=z[f"w{last}"], spacing_deg=float(z[f"h{last}"]), sigma_deg=float(z[f"s{last}"]),
                 residual=float(z[f"res{last}"]), seconds=float(z[f"t{last}"]))
basis = basis_from_odf(level, mat, rel=rel, min_distance_deg=dmin)
print(f"ODF level {last} ({level.spacing_deg} deg): basis K {len(basis)} ({tag})", flush=True)


def residual(op, x):
    r = (op.direct(x) - b) * wseg
    return float(np.linalg.norm(r)) / wb_norm


summary = {"tag": tag, "K": len(basis)}
grid = MatrixGrid(basis, np.deg2rad(sigma_deg))
t0 = time.perf_counter()
op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=1.0, normalized=True,
                                reserve_coefficient_arrays=0)
L = 1.1 * estimate_L_power_streamed(op, niter=6, seed=0, verbose=0)
t1 = time.perf_counter()
x = FISTAHuber(op, prox_kind="nonneg", L=L, huber_delta=p["huber_delta"]).run(
    np.zeros(op.coeff_shape, np.float32), b, niter=200, weights=weights)
t2 = time.perf_counter()
summary.update(pf_mode=op.pf_mode, build_s=t1 - t0, fista_s=t2 - t1, residual=residual(op, x))
grid_mats = R.concatenate(grid.rotations_at_level(0)).as_matrix()
with h5py.File(out_h5, "w") as f:
    f.create_dataset("grid_mats", data=grid_mats, compression="gzip")
    ds = f.create_dataset("coeffs", data=x, compression="gzip")
    ds.attrs["axes"] = "(K, Ny, Nx)"
    for k, v in summary.items():
        f.attrs[k] = v
    f.attrs["sigma_deg"] = sigma_deg
op.free_memory()
del op, x
print(json.dumps(summary), flush=True)

# residuals of the published reconstructions (same data, weights and geometry)
for name in ("adaptive", "uniform"):
    cache = os.path.join(HERE, f"residual_{name}.json")
    if os.path.exists(cache):
        continue
    with h5py.File(os.path.join(PROCESS, f"reconstruction_{name}.h5"), "r") as f:
        xs, mats, sig = f["coeffs"][...], f["grid_mats"][...], float(f.attrs["sigma_deg"])
    t0 = time.perf_counter()
    op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=MatrixGrid(mats, np.deg2rad(sig)),
                                    max_gb=1.0, normalized=True, reserve_coefficient_arrays=0)
    res = {"name": name, "K": len(mats), "sigma_deg": sig, "residual": residual(op, xs),
           "seconds": time.perf_counter() - t0}
    op.free_memory()
    del op, xs
    json.dump(res, open(cache, "w"))
    print(json.dumps(res), flush=True)
