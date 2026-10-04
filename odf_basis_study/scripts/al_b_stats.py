import json, os, sys
import numpy as np
sys.path.insert(0, "/dtu-compute/msaca/adaptive_basis_simulations")
from simtools import figures
from fig_al import load_map, PROCESS
med_i, keep_i, _ = load_map("indexed", os.path.join(PROCESS, "reconstruction_adaptive.h5"))
med, keep, _ = load_map("odfB", "/dtu-compute/msaca/claude_scratch/odf_al/reconstruction_odf_L3_r0.01_d0.5_s0.4.h5")
both = keep & keep_i
mis = figures.misorientation_map(med_i, med, both)
s = {"median": float(np.nanmedian(mis)), "mean": float(np.nanmean(mis)), "frac_lt1": float(np.nanmean(mis[both] < 1)),
     "frac_gt5": float(np.nanmean(mis[both] > 5))}
print(json.dumps(s)); json.dump(s, open("cache/al_b_stats.json", "w"))
