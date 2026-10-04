"""textomo_odf.ipynb for both samples, from their textomo_adaptive.ipynb: the basis comes from the
ODF of the data summed over translations instead of from indexing; everything else is unchanged."""
import copy
import json
import uuid

REPO = "/dtu-compute/msaca/adaptive_basis_simulations/texture_tomography"


def src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [],
            "source": src(text)}


for sample, mos, dmin in [("domains_mosaicity_1p0deg", "1", 0.2), ("domains_mosaicity_10p0deg", "10", 0.2)]:
    nb = json.load(open(f"{REPO}/{sample}/textomo_adaptive.ipynb"))
    cells = [copy.deepcopy(c) for c in nb["cells"]]
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    cells[0] = md(f"""
# Texture tomography with a basis from the ODF ({mos} degree mosaicity)

The same reconstruction as `textomo_adaptive.ipynb`, but the orientation basis comes from the data
themselves instead of from peak segmentation and indexing:

1. the data summed over the translations are a bulk measurement of the sample's orientation
   distribution (ODF): data = PF @ w, with w the volume per orientation (the texture operator on a
   single pixel);
2. the ODF is reconstructed coarse to fine: a uniform (cubochoric) grid of the fundamental zone at
   4 deg, then four refinements that halve the spacing around the orientations with weight
   (0.25 deg at the end);
3. the basis is the support of the finest level (weight above 1 % of the maximum), thinned to a
   minimum distance; the TT reconstruction then runs as in `textomo_adaptive.ipynb`.

`diffractom.odf_basis` (diffractom branch `odf-basis`).
""")
    imports = "".join(cells[1]["source"])
    imports = imports.replace(
        "from diffractom.operators.single_phase_forward_operator import group_reflections_into_rings\n",
        "from diffractom.operators.single_phase_forward_operator import group_reflections_into_rings\n"
        "from diffractom.odf_basis import basis_from_odf, nearest_misorientation_deg, reconstruct_odf, symmetry_rotations\n")
    imports = imports.replace("from simtools import paths, plot\n", "from simtools import io, paths, plot\n")
    imports = imports.replace("import matplotlib.pyplot as plt\n", "import matplotlib.pyplot as plt\nimport pyopencl as cl\n")
    cells[1]["source"] = src(imports)
    odf_cells = [
        md("""
## ODF from the data summed over translations
Each level is a nonnegative least-squares fit of the summed data (same eta weights as the reconstruction
below) with Gaussian kernels of width 0.6 x the spacing. Orientations above `KEEP_REL` x the maximum
weight are refined at half the spacing for the next level.
"""),
        code("""
ODF_SPACING_DEG = 4.0   # level 0: uniform grid of the fundamental zone
ODF_LEVELS = 5          # 4, 2, 1, 0.5, 0.25 deg
KEEP_REL = 1e-3         # refine around orientations above this fraction of the maximum weight

weights = np.ones((N_eta, N_theta), dtype=np.float32)  # as in the reconstruction below
weights[40:50, :] = 0
weights[130:140, :] = 0

ctx = cl.create_some_context(interactive=False)  # one OpenCL context for the ODF and the reconstruction
queue = cl.CommandQueue(ctx)

t0 = time.perf_counter()
levels = reconstruct_odf(cfg, mat, arr.reshape(N_Omega, My, N_eta * N_theta).astype(np.float32), weights,
                         spacing_deg=ODF_SPACING_DEG, levels=ODF_LEVELS, keep_rel=KEEP_REL, ctx=ctx, queue=queue)
odf_time = time.perf_counter() - t0
print(f"ODF in {odf_time:.0f} s")
"""),
        md(f"""
## Orientation grid: the support of the finest ODF level
Orientations above `SUPPORT_REL` x the maximum weight, thinned to `MIN_DISTANCE_DEG` in order of weight.
"""),
        code(f"""
SUPPORT_REL = 1e-2
MIN_DISTANCE_DEG = {dmin}
sigma_deg = 0.4  # width of the Gaussian kernel around every basis orientation (as the adaptive basis)

basis = basis_from_odf(levels[-1], mat, rel=SUPPORT_REL, min_distance_deg=MIN_DISTANCE_DEG)
grid = Grid.from_rotation_matrices(basis, np.deg2rad(sigma_deg))
K = len(grid.nodes_at_level(0))
print(f"K = {{K}} orientations, sigma = {{sigma_deg}} deg")
"""),
        md("""
## Comparison with the ground truth
The misorientation from each ground-truth mesh element to the closest basis orientation, for this
basis and the adaptive (indexing) basis.
"""),
        code("""
gt = io.read_ground_truth(SAMPLE)
idx = np.random.default_rng(0).choice(len(gt["orientation"]), 5000, replace=False)
q_gt = R.from_matrix(gt["orientation"][idx]).as_quat()
sym = symmetry_rotations(mat)
mis = {name: nearest_misorientation_deg(q_gt, R.from_matrix(u).as_quat(), sym)
       for name, u in [("ODF basis", basis), ("adaptive basis", np.load(paths.basis_path(SAMPLE)))]}

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for name, m in mis.items():
    ax.hist(m, bins=100, range=(0, 1.5), histtype="step", lw=2, label=f"{name}: median {np.median(m):.3f} deg")
ax.set_xlabel("misorientation to the closest basis orientation (deg)")
ax.set_ylabel("ground-truth mesh elements")
ax.legend()
plot.despine(ax)
plt.show()
"""),
    ]
    cells = cells[:9] + odf_cells + cells[11:]
    i_op = next(i for i, c in enumerate(cells) if "reserve_coefficient_arrays=0)" in "".join(c["source"]))
    op_src = "".join(cells[i_op]["source"]).replace("reserve_coefficient_arrays=0)",
                                                    "reserve_coefficient_arrays=0, ctx=ctx, queue=queue)")
    cells[i_op]["source"] = src(op_src)
    i_save = next(i for i, c in enumerate(cells) if "reconstruction_adaptive.h5" in "".join(c["source"]))
    save = "".join(cells[i_save]["source"])
    save = save.replace("reconstruction_adaptive.h5", "reconstruction_odf.h5")
    save = save.replace('f.attrs["grid"] = "adaptive"', 'f.attrs["grid"] = "odf"')
    save = save.replace('    f.attrs["fista_time_s"] = fista_time\n',
                        '    f.attrs["fista_time_s"] = fista_time\n'
                        '    f.attrs["odf_time_s"] = odf_time\n'
                        '    f.attrs["odf_spacing_deg"] = ODF_SPACING_DEG\n'
                        '    f.attrs["odf_levels"] = ODF_LEVELS\n'
                        '    f.attrs["odf_keep_rel"] = KEEP_REL\n'
                        '    f.attrs["support_rel"] = SUPPORT_REL\n'
                        '    f.attrs["min_distance_deg"] = MIN_DISTANCE_DEG\n')
    cells[i_save]["source"] = src(save)
    nb["cells"] = cells
    json.dump(nb, open(f"{REPO}/{sample}/textomo_odf.ipynb", "w"), indent=1, ensure_ascii=False)
    print("wrote", sample)
