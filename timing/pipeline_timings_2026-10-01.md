# Pipeline timings (simulated samples)

Measured 2026-10-01 with `timing/time_pipelines.py`, all stages in one job on one machine: NVIDIA A40, 16 cores of Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz (host n-62-18-4). Wall time of the notebooks of each stage, with the parameters they contain; the integration is shared by both pipelines and counted in both totals.

| Sample | Pipeline | Peak segmentation | Peak indexing | Integration | TT reconstruction | Total |
|---|---|---|---|---|---|---|
| 1.0 deg | TT-Adaptive | 3.2 min | 3.7 min | 5.8 min | 0.6 min | **13.3 min** |
| 1.0 deg | TT-Uniform | -- | -- | 5.8 min | 1.54 h (93 min) | **1.64 h (98 min)** |
| 10.0 deg | TT-Adaptive | 3.7 min | 15.0 min | 6.5 min | 1.5 min | **26.7 min** |
| 10.0 deg | TT-Uniform | -- | -- | 6.5 min | 1.55 h (93 min) | **1.66 h (99 min)** |

Per notebook:

| Notebook | Wall time |
|---|---|
| `adaptive_basis/domains_mosaicity_1p0deg/segment_peaks/02_segment.ipynb` | 3.2 min |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/02_pbp_index.ipynb` | 0.8 min |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/03_refine.ipynb` | 2.9 min |
| `integration/domains_mosaicity_1p0deg/01_inspect_integrate.ipynb` | 5.8 min |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_adaptive.ipynb` | 0.6 min |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_uniform.ipynb` | 1.54 h (93 min) |
| `adaptive_basis/domains_mosaicity_10p0deg/segment_peaks/02_segment.ipynb` | 3.7 min |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/02_pbp_index.ipynb` | 4.7 min |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/03_refine.ipynb` | 10.3 min |
| `integration/domains_mosaicity_10p0deg/01_inspect_integrate.ipynb` | 6.5 min |
| `texture_tomography/domains_mosaicity_10p0deg/textomo_adaptive.ipynb` | 1.5 min |
| `texture_tomography/domains_mosaicity_10p0deg/textomo_uniform.ipynb` | 1.55 h (93 min) |
