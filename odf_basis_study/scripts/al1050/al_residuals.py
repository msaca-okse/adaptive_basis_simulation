"""Weighted relative data residual ||w (A x - b)|| / ||w b|| of saved Al reconstructions (one OpenCL context)."""
import json, os, sys, time
import h5py
import numpy as np
import pyopencl as cl
from al_problem import load
from diffractom import MatrixGrid, SinglePhaseForwardOperator
from integration.frame_loader import PROCESS

HERE = os.path.dirname(os.path.abspath(__file__))
p = load()
b, wseg = p["data"], p["weights"].ravel()
wb = float(np.linalg.norm(b * wseg))
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
files = {"indexed": os.path.join(PROCESS, "reconstruction_adaptive.h5"),
         "uniform": os.path.join(PROCESS, "reconstruction_uniform.h5")}
files.update({os.path.basename(f)[:-3]: os.path.join(HERE, f) for f in sys.argv[1:]})
out = {}
for name, h5 in files.items():
    with h5py.File(h5, "r") as f:
        x, mats, sig = f["coeffs"][...], f["grid_mats"][...], float(f.attrs["sigma_deg"])
    t0 = time.perf_counter()
    op = SinglePhaseForwardOperator(cfg=p["cfg"], material=p["material"], grid=MatrixGrid(mats, np.deg2rad(sig)),
                                    max_gb=1.0, normalized=True, reserve_coefficient_arrays=0, ctx=ctx, queue=queue)
    r = (op.direct(x) - b) * wseg
    out[name] = {"K": len(mats), "sigma_deg": sig, "residual": float(np.linalg.norm(r)) / wb,
                 "seconds": time.perf_counter() - t0}
    print(name, json.dumps(out[name]), flush=True)
    op.free_memory()
    del op, x, r
json.dump(out, open(os.path.join(HERE, "residuals.json"), "w"), indent=1)
