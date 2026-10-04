"""diffractom.odf_basis (branch odf-basis) vs the prototype: same levels, same basis, same TT accuracy."""
import sys, time
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/dtu-compute/msaca/adaptive_basis_simulations")
from simtools import io, stability
from diffractom.odf_basis import reconstruct_odf, basis_from_odf, symmetry_rotations, nearest_misorientation_deg

sample = sys.argv[1]
p = stability.load_problem(sample)
t0 = time.perf_counter()
H0, NLEV = (float(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2.0, 4)
levels = reconstruct_odf(p["cfg"], p["material"], p["data"], p["weights"], spacing_deg=H0, levels=NLEV)
t1 = time.perf_counter()
basis = basis_from_odf(levels[-1], p["material"], rel=1e-2, min_distance_deg=0.2)
print(f"ODF {t1 - t0:.0f} s, basis K {len(basis)} ({time.perf_counter() - t1:.1f} s)")
proto = np.load(f"odf_{'1p0' if '1p0' in sample else '10p0'}_b.npz")
for lev, L in (enumerate(levels) if H0 == 2.0 else []):
    q0, w0 = proto[f"q{lev}"], proto[f"w{lev}"]
    same_q = q0.shape == L.q.shape and np.abs(np.abs((q0 * L.q).sum(1)) - 1).max() < 1e-6
    print(f"level {lev}: K {len(L.q)} (prototype {len(q0)}), same orientations {same_q}"
          + (f", max rel weight diff {np.abs(L.w - w0).max() / w0.max():.2e}" if same_q else ""))
gt = io.read_ground_truth(sample)
idx = np.random.default_rng(0).choice(len(gt["orientation"]), 20000, replace=False)
a = nearest_misorientation_deg(Rotation.from_matrix(gt["orientation"][idx]).as_quat(),
                               Rotation.from_matrix(basis).as_quat(), symmetry_rotations(p["material"]))
print(f"GT -> basis: median {np.median(a):.3f} p99 {np.percentile(a, 99):.3f} deg")
np.save(f"pkg_basis_{sample}.npy", basis)
