# Pipeline timings (simulated samples)

Measured 2026-10-03 with `timing/time_pipelines.py` on NVIDIA A40, 16 cores of Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz (host n-62-18-4); diffractom f58a234 2026-10-03 Merge branch 'streaming-fused': next forward pass inside the adjoint pass (GPU and streamed), fused residual/norm/clip, PF forward skips empty rows. Wall time of the notebooks of each stage, with the parameters they contain; the integration is shared by both TT pipelines and counted in both totals. Times marked † were not rerun: measured 2026-10-02 on n-62-18-4.

| Sample | Pipeline | Peak segmentation | PBP indexing | Refinement + basis | Integration | TT reconstruction | Total |
|---|---|---|---|---|---|---|---|
| 1.0 deg | pbp | 3.5 min † | 0.8 min † | 2.8 min † | -- | -- | **7.1 min** |
| 1.0 deg | TT-Adaptive | 3.5 min † | 0.8 min † | 2.8 min † | 5.0 min † | 0.8 min | **12.9 min** |
| 1.0 deg | TT-Uniform | -- | -- | -- | 5.0 min † | 16.2 min | **21.2 min** |
| 10.0 deg | pbp | 4.0 min † | 4.9 min † | 9.2 min † | -- | -- | **18.0 min** |
| 10.0 deg | TT-Adaptive | 4.0 min † | 4.9 min † | 9.2 min † | 5.5 min † | 1.0 min | **24.5 min** |
| 10.0 deg | TT-Uniform | -- | -- | -- | 5.5 min † | 16.1 min | **21.6 min** |

## Accuracy of the maps

From `visualization/<sample>/figure_panels.ipynb` on the maps of this run: the median misorientation to the ground truth over the sample, and the mean distance of the grain boundaries (KAM >= 4 deg) to the nearest ground-truth grain boundary (1 voxel = 4 display pixels).

| Sample | Map | Median misorientation (deg) | Grain-boundary deviation (µm) | (voxels) |
|---|---|---|---|---|
| 1.0 deg | pbp | 0.189 | 0.0272 | 0.27 |
| 1.0 deg | TT-Uniform | 1.094 | 0.0215 | 0.21 |
| 1.0 deg | TT-Adaptive | 0.241 | 0.0182 | 0.18 |
| 10.0 deg | pbp | 0.404 | 0.0502 | 0.49 |
| 10.0 deg | TT-Uniform | 1.220 | 0.0182 | 0.18 |
| 10.0 deg | TT-Adaptive | 0.346 | 0.0165 | 0.16 |

Per notebook:

| Notebook | Wall time | Measured | Host |
|---|---|---|---|
| `adaptive_basis/domains_mosaicity_1p0deg/segment_peaks/02_segment.ipynb` | 3.5 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/02_pbp_index.ipynb` | 0.8 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/03_refine.ipynb` | 2.8 min | 2026-10-02 | n-62-18-4 |
| `integration/domains_mosaicity_1p0deg/01_inspect_integrate.ipynb` | 5.0 min | 2026-10-02 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_adaptive.ipynb` | 0.8 min | 2026-10-03 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_uniform.ipynb` | 16.2 min | 2026-10-03 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_10p0deg/segment_peaks/02_segment.ipynb` | 4.0 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/02_pbp_index.ipynb` | 4.9 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/03_refine.ipynb` | 9.2 min | 2026-10-02 | n-62-18-4 |
| `integration/domains_mosaicity_10p0deg/01_inspect_integrate.ipynb` | 5.5 min | 2026-10-02 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_10p0deg/textomo_adaptive.ipynb` | 1.0 min | 2026-10-03 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_10p0deg/textomo_uniform.ipynb` | 16.1 min | 2026-10-03 | n-62-18-4 |
