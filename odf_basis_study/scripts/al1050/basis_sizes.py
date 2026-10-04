import numpy as np, time
from scipy.spatial.transform import Rotation
from diffractom.odf_basis import ODFLevel, basis_from_odf, nearest_misorientation_deg, symmetry_rotations
from al_problem import load
z = np.load("odf_al.npz")
L = ODFLevel(q=z["q3"], w=z["w3"], spacing_deg=float(z["h3"]), sigma_deg=float(z["s3"]), residual=0, seconds=0)
sym = symmetry_rotations("cubic")
b0 = np.load("/zhome/71/c/146676/texture_tomography/adaptive_basis/basis.npy")
q_ix = Rotation.from_matrix(Rotation.from_euler("z", 50.0, degrees=True).as_matrix() @ b0).as_quat()
q_ix = q_ix[np.random.default_rng(0).choice(len(q_ix), 20000, replace=False)]
for rel in (1e-2, 3e-2, 1e-1):
    for d in (0.4, 0.5, 0.7):
        t0 = time.perf_counter()
        B = basis_from_odf(L, "cubic", rel=rel, min_distance_deg=d)
        a = nearest_misorientation_deg(q_ix, Rotation.from_matrix(B).as_quat(), sym)
        s = L.support(rel)
        print(f"rel {rel:g} (support {len(s)}, {L.w[s].sum() / L.w.sum():.3f} of weight) thin {d}: K {len(B):7d}; "
              f"indexed -> basis median {np.median(a):.3f} p90 {np.percentile(a, 90):.3f} ({time.perf_counter() - t0:.0f} s)", flush=True)
