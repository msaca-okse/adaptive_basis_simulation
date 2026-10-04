"""Coarse-to-fine bulk ODF of the Al1050 data, saved after every level; compare its support with the
adaptive (indexing) basis. usage: python al_odf.py LEVELS OUT.npz [keep_mass]"""
import sys, time
import numpy as np
from scipy.spatial.transform import Rotation
from al_problem import load
from diffractom.odf_basis import reconstruct_odf, symmetry_rotations, nearest_misorientation_deg

levels_n, out = int(sys.argv[1]), sys.argv[2]
keep_mass = float(sys.argv[3]) if len(sys.argv) > 3 else None
t0 = time.perf_counter()
p = load()
print(f"data {p['shape4']} loaded in {time.perf_counter() - t0:.0f} s", flush=True)
sym = symmetry_rotations(p["material"])
q_ad = Rotation.from_matrix(p["adaptive_basis"]).as_quat()
q_ad_s = q_ad[np.random.default_rng(0).choice(len(q_ad), min(20000, len(q_ad)), replace=False)]
save = {}


def done(lev, L):
    save.update({f"q{lev}": L.q, f"w{lev}": L.w, f"h{lev}": L.spacing_deg, f"s{lev}": L.sigma_deg,
                 f"res{lev}": L.residual, f"t{lev}": L.seconds})
    np.savez(out, **save)
    for rel in (1e-2, 1e-3):
        s = L.support(rel)
        a = nearest_misorientation_deg(q_ad_s, L.q[s], sym)
        print(f"level {lev} ({L.spacing_deg:.3f} deg): support > {rel:g} max {len(s):8d} "
              f"({L.w[s].sum() / L.w.sum():.4f} of weight); indexed -> support median {np.median(a):.3f} "
              f"p99 {np.percentile(a, 99):.3f}", flush=True)


t0 = time.perf_counter()
reconstruct_odf(p["cfg"], p["material"], p["data"], p["weights"], spacing_deg=4.0, levels=levels_n,
                niter_between=50, max_gb=8.0, keep_mass=keep_mass, callback=done)
print(f"ODF in {time.perf_counter() - t0:.0f} s", flush=True)
