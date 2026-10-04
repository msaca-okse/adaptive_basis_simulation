"""ODF levels of a simulated sample with the odf_basis defaults (as textomo_odf.ipynb), saved for the figures."""
import sys, time
import numpy as np
sys.path.insert(0, "/dtu-compute/msaca/adaptive_basis_simulations")
from simtools import stability
from diffractom.odf_basis import reconstruct_odf

sample = sys.argv[1]
p = stability.load_problem(sample)
t0 = time.perf_counter()
levels = reconstruct_odf(p["cfg"], p["material"], p["data"], p["weights"])
print(f"ODF in {time.perf_counter() - t0:.0f} s", flush=True)
save = {}
for i, L in enumerate(levels):
    save.update({f"q{i}": L.q, f"w{i}": L.w, f"h{i}": L.spacing_deg, f"s{i}": L.sigma_deg, f"res{i}": L.residual,
                 f"t{i}": L.seconds})
np.savez(f"levels_{sample}.npz", **save)
