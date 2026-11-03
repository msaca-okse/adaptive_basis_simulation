# Pipeline timings (simulated samples)

Measured 2026-11-03 with `timing/time_pipelines.py` on NVIDIA A40, 16 cores of Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz (host n-62-18-4); diffractom v0.3.0 (17a73a9, release-0.3.0). Wall time of the notebooks of each stage, with the parameters they contain; the integration is shared by both TT pipelines and counted in both totals. Times marked † were not rerun: measured 2026-10-02 on n-62-18-4.

| Sample | Pipeline | Peak segmentation | PBP indexing | Refinement + basis | Integration | TT reconstruction | Total |
|---|---|---|---|---|---|---|---|
| 1.0 deg | pbp | 3.5 min † | 0.8 min † | 2.8 min † | -- | -- | **7.1 min** |
| 1.0 deg | TT-Adaptive | 3.5 min † | 0.8 min † | 2.8 min † | 5.0 min † | 1.2 min | **13.4 min** |
| 1.0 deg | TT-Uniform | -- | -- | -- | 5.0 min † | 25.7 min | **30.8 min** |
| 10.0 deg | pbp | 4.0 min † | 4.9 min † | 9.2 min † | -- | -- | **18.0 min** |
| 10.0 deg | TT-Adaptive | 4.0 min † | 4.9 min † | 9.2 min † | 5.5 min † | 1.0 min | **24.5 min** |
| 10.0 deg | TT-Uniform | -- | -- | -- | 5.5 min † | 25.5 min | **31.0 min** |

## Accuracy of the maps

The uniform grid of the notebooks is drawn with a fixed seed (`np.random.seed(4200)`, K = 100 260). The times above were measured before the seed was added, with an unseeded draw of the same size (K = 100 557 for the 1 degree sample); the accuracy below is that of the seeded run, which made the notebooks of this repository.

From `visualization/<sample>/figure_panels.ipynb` on the maps of this run: the median misorientation to the ground truth over the sample, and the mean distance of the grain boundaries (KAM >= 4 deg) to the nearest ground-truth grain boundary (1 voxel = 4 display pixels).

| Sample | Map | Median misorientation (deg) | Grain-boundary deviation (µm) | (voxels) |
|---|---|---|---|---|
| 1.0 deg | pbp | 0.189 | 0.0272 | 0.27 |
| 1.0 deg | TT-Uniform | 1.057 | 0.0201 | 0.20 |
| 1.0 deg | TT-Adaptive | 0.216 | 0.0169 | 0.17 |
| 10.0 deg | pbp | 0.404 | 0.0502 | 0.49 |
| 10.0 deg | TT-Uniform | 1.243 | 0.0177 | 0.17 |
| 10.0 deg | TT-Adaptive | 0.329 | 0.0164 | 0.16 |

Per notebook:

| Notebook | Wall time | Measured | Host |
|---|---|---|---|
| `adaptive_basis/domains_mosaicity_1p0deg/segment_peaks/02_segment.ipynb` | 3.5 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/02_pbp_index.ipynb` | 0.8 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/03_refine.ipynb` | 2.8 min | 2026-10-02 | n-62-18-4 |
| `integration/domains_mosaicity_1p0deg/01_inspect_integrate.ipynb` | 5.0 min | 2026-10-02 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_adaptive.ipynb` | 1.2 min | 2026-11-03 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_uniform.ipynb` | 25.7 min | 2026-11-03 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_10p0deg/segment_peaks/02_segment.ipynb` | 4.0 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/02_pbp_index.ipynb` | 4.9 min | 2026-10-02 | n-62-18-4 |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/03_refine.ipynb` | 9.2 min | 2026-10-02 | n-62-18-4 |
| `integration/domains_mosaicity_10p0deg/01_inspect_integrate.ipynb` | 5.5 min | 2026-10-02 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_10p0deg/textomo_adaptive.ipynb` | 1.0 min | 2026-11-03 | n-62-18-4 |
| `texture_tomography/domains_mosaicity_10p0deg/textomo_uniform.ipynb` | 25.5 min | 2026-11-03 | n-62-18-4 |
