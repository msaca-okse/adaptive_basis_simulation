# ODF-first orientation basis (study, branch `odf-basis`)

An orientation basis for texture tomography taken from the data themselves instead of from peak indexing:
the data summed over translations are a bulk measurement of the sample's ODF (the texture operator on a
single pixel), which is reconstructed coarse to fine (4, 2, 1, 0.5, 0.25 deg; refined only around the
orientations with weight); the support of the finest level, thinned, is the TT basis. Code:
`diffractom.odf_basis` on the diffractom branch `odf-basis`. Not part of the article.

## Results

Median misorientation to the ground truth (weighted medoid of the top 24 coefficients, upsampled 4x, as the
accuracy table of `timing/pipeline_timings.md`; sigma 0.4 deg, 200 FISTA iterations):

| Basis | K (1 deg) | median (1 deg) | K (10 deg) | median (10 deg) |
|---|---|---|---|---|
| uniform grid | 100 329 | 1.094 | 100 302 | 1.220 |
| indexed (point by point) | 345 | 0.241 | 3 025 | 0.346 |
| ODF, `textomo_odf.ipynb` | 2 320 | 0.237 | 14 719 | 0.325 |

ODF step: 65 s (1 deg) and 112 s (10 deg) on a V100; segmentation + indexing + refinement took 7.1 and
18.1 min on an A40.

Al1050 (texture_tomography repository, branch `odf-basis`): relative weighted data residual 0.356 with the
ODF basis of its `textomo_odf.ipynb` (K 34 845) against 0.394 with the indexed basis (K 48 017) and 0.771
with the uniform grid (K 82 198, sigma 2 deg); the ODF and indexed maps agree to a median of 0.56 deg. The sample is 15 %
deformed, so its ODF is broad: 1.6 million candidate orientations at 0.5 deg.

## Files

- `figures/`: every figure in a light and a dark variant.
- `scripts/`: the evaluation (`tt_eval.py`), the figures (`fig_*.py`, `figstyle.py`), the ODF levels of the
  simulated samples (`levels_sim.py`), the grid comparison (`compare_grids.py`), the notebook generator
  (`make_notebooks.py`) and the package-versus-prototype check (`pkg_test.py`); `scripts/al1050/`: the same
  for the real data (`al_problem.py` rebuilds the problem of `textomo_adaptive.ipynb`). They were run from a
  scratch directory with absolute paths (`/dtu-compute/msaca/claude_scratch/...`), on LSF GPU nodes.
