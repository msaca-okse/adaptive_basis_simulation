"""texture_tomography/texture_tomography/textomo_odf.ipynb (Al1050) from textomo_adaptive.ipynb: the basis comes
from the ODF of the data summed over translations instead of from indexing; everything else is unchanged.
usage: python make_notebook.py SUPPORT_REL MIN_DISTANCE_DEG"""
import copy
import json
import sys
import uuid

REPO = "/zhome/71/c/146676/texture_tomography/texture_tomography"
SUPPORT_REL, MIN_DISTANCE = float(sys.argv[1]), float(sys.argv[2])


def src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [],
            "source": src(text)}


nb = json.load(open(f"{REPO}/textomo_adaptive.ipynb"))
cells = [copy.deepcopy(c) for c in nb["cells"]]
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None


def find(text):
    return next(i for i, c in enumerate(cells) if text in "".join(c["source"]))


cells[0] = md("""
# Texture tomography of Al1050 with a basis from the ODF

The same reconstruction as `textomo_adaptive.ipynb`, but the orientation basis comes from the data themselves
instead of from peak segmentation and point-by-point indexing:

1. the data summed over the translations are a bulk measurement of the sample's orientation distribution
   (ODF): data = PF @ w, with w the volume per orientation (the texture operator on a single pixel);
2. the ODF is reconstructed coarse to fine: a uniform (cubochoric) grid of the fundamental zone at 4 deg,
   then refinements that halve the spacing around the strongest orientations (99 % of the weight), down
   to 0.5 deg. The sample is 15 % deformed, so its ODF is broad: a 0.25 deg level would hold ~10^7
   candidates;
3. the basis is the support of the finest level (weight above `SUPPORT_REL` x the maximum), thinned to a
   minimum distance; the reconstruction then runs as in `textomo_adaptive.ipynb`.

The ODF is reconstructed in the operator's frame, so the basis needs no rotation by `OMEGA_OFFSET_DEG`
(the indexed basis does). `diffractom.odf_basis` (diffractom branch `odf-basis`).
""")
imports = "".join(cells[1]["source"])
imports = imports.replace(
    "from diffractom.operators.single_phase_forward_operator import group_reflections_into_rings\n",
    "from diffractom.operators.single_phase_forward_operator import group_reflections_into_rings\n"
    "from diffractom import MatrixGrid\n"
    "from diffractom.odf_basis import basis_from_odf, nearest_misorientation_deg, reconstruct_odf, symmetry_rotations\n")
imports = imports.replace("import matplotlib.pyplot as plt\n", "import matplotlib.pyplot as plt\nimport pyopencl as cl\n")
cells[1]["source"] = src(imports)

i_grid = find("basis = np.load(os.path.join(REPO, \"adaptive_basis\", \"basis.npy\"))")
odf_cells = [
    md("""
## ODF from the data summed over translations
Each level is a nonnegative least-squares fit of the summed data (same weights as the reconstruction) with
Gaussian kernels of width 0.6 x the spacing; 50 iterations at the first level and between, 200 at the last.
"""),
    code("""
ODF_SPACING_DEG = 4.0   # level 0: uniform grid of the fundamental zone
ODF_LEVELS = 4          # 4, 2, 1, 0.5 deg
KEEP_MASS = 0.99        # refine around the strongest orientations holding this fraction of the weight

ctx = cl.create_some_context(interactive=False)  # one OpenCL context for the ODF and the reconstruction
queue = cl.CommandQueue(ctx)

t0 = time.perf_counter()
levels = reconstruct_odf(cfg, mat, arr.reshape(N_Omega, My, N_eta * N_theta), weights, spacing_deg=ODF_SPACING_DEG,
                         levels=ODF_LEVELS, keep_mass=KEEP_MASS, niter_between=50, max_gb=8.0, ctx=ctx, queue=queue)
odf_time = time.perf_counter() - t0
print(f"ODF in {odf_time:.0f} s")
"""),
    md("""
## Orientation grid: the support of the finest ODF level
Orientations above `SUPPORT_REL` x the maximum weight, thinned to `MIN_DISTANCE_DEG` in order of weight.
"""),
    code(f"""
SUPPORT_REL = {SUPPORT_REL:g}
MIN_DISTANCE_DEG = {MIN_DISTANCE:g}
sigma_deg = 0.4  # width of the Gaussian kernel around every basis orientation (as the adaptive basis)

basis = basis_from_odf(levels[-1], mat, rel=SUPPORT_REL, min_distance_deg=MIN_DISTANCE_DEG)
grid = MatrixGrid(basis, np.deg2rad(sigma_deg))  # arrays, no per-orientation objects
K = len(grid.nodes_at_level(0))
print(f"K = {{K}} orientations, sigma = {{sigma_deg}} deg")
"""),
    md("""
## Comparison with the adaptive (indexing) basis
The misorientation from each indexed orientation (rotated into the operator frame) to the closest
orientation of this basis.
"""),
    code("""
indexed = R.from_euler("z", OMEGA_OFFSET_DEG, degrees=True).as_matrix() @ np.load(os.path.join(REPO, "adaptive_basis", "basis.npy"))
q_ix = R.from_matrix(indexed).as_quat()
q_ix = q_ix[np.random.default_rng(0).choice(len(q_ix), 20000, replace=False)]
mis = nearest_misorientation_deg(q_ix, R.from_matrix(basis).as_quat(), symmetry_rotations(mat))

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.hist(mis, bins=100, range=(0, 3))
ax.axvline(np.median(mis), c="w", label=f"median {np.median(mis):.2f} deg")
ax.set_xlabel("indexed orientation -> closest ODF-basis orientation (deg)")
ax.legend()
plt.show()
"""),
]
cells = cells[:i_grid - 1] + odf_cells + cells[i_grid + 1:]
i_op = find("reserve_coefficient_arrays=0)")
cells[i_op]["source"] = src("".join(cells[i_op]["source"]).replace(
    "reserve_coefficient_arrays=0)", "reserve_coefficient_arrays=0, ctx=ctx, queue=queue)"))
i_save = find("reconstruction_adaptive.h5")
save = "".join(cells[i_save]["source"])
save = save.replace("reconstruction_adaptive.h5", "reconstruction_odf.h5")
save = save.replace('f.attrs["grid"] = "adaptive"', 'f.attrs["grid"] = "odf"')
save = save.replace('    f.attrs["theta_deg"] = theta_deg\n', '')
save = save.replace('    f.attrs["fista_time_s"] = fista_time\n',
                    '    f.attrs["fista_time_s"] = fista_time\n'
                    '    f.attrs["odf_time_s"] = odf_time\n'
                    '    f.attrs["odf_spacing_deg"] = ODF_SPACING_DEG\n'
                    '    f.attrs["odf_levels"] = ODF_LEVELS\n'
                    '    f.attrs["odf_keep_mass"] = KEEP_MASS\n'
                    '    f.attrs["support_rel"] = SUPPORT_REL\n'
                    '    f.attrs["min_distance_deg"] = MIN_DISTANCE_DEG\n')
cells[i_save]["source"] = src(save)
if not "".join(cells[-1]["source"]).strip():
    cells = cells[:-1]
nb["cells"] = cells
json.dump(nb, open(f"{REPO}/textomo_odf.ipynb", "w"), indent=1, ensure_ascii=False)
print("wrote textomo_odf.ipynb with", len(cells), "cells")
