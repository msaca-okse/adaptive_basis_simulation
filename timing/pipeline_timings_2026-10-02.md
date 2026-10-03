# Pipeline timings (simulated samples)

Measured 2026-10-02 with `timing/time_pipelines.py`, all stages in one job on one machine: NVIDIA A40, 16 cores of Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz (host n-62-18-4). Wall time of the notebooks of each stage, with the parameters they contain; the integration is shared by both pipelines and counted in both totals.

| Sample | Pipeline | Peak segmentation | Peak indexing | Integration | TT reconstruction | Total |
|---|---|---|---|---|---|---|
| 1.0 deg | TT-Adaptive | 3.5 min | 3.7 min | 5.0 min | 0.6 min | **12.7 min** |
| 1.0 deg | TT-Uniform | -- | -- | 5.0 min | 1.56 h (94 min) | **1.65 h (99 min)** |
| 10.0 deg | TT-Adaptive | 4.0 min | 14.0 min | 5.5 min | 1.0 min | **24.5 min** |
| 10.0 deg | TT-Uniform | -- | -- | 5.5 min | not rerun (93 min on 2026-10-01) | -- |

Per notebook:

| Notebook | Wall time |
|---|---|
| `adaptive_basis/domains_mosaicity_1p0deg/segment_peaks/02_segment.ipynb` | 3.5 min |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/02_pbp_index.ipynb` | 0.8 min |
| `adaptive_basis/domains_mosaicity_1p0deg/indexing/03_refine.ipynb` | 2.8 min |
| `integration/domains_mosaicity_1p0deg/01_inspect_integrate.ipynb` | 5.0 min |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_adaptive.ipynb` | 0.6 min |
| `texture_tomography/domains_mosaicity_1p0deg/textomo_uniform.ipynb` | 1.56 h (94 min) |
| `adaptive_basis/domains_mosaicity_10p0deg/segment_peaks/02_segment.ipynb` | 4.0 min |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/02_pbp_index.ipynb` | 4.9 min |
| `adaptive_basis/domains_mosaicity_10p0deg/indexing/03_refine.ipynb` | 9.2 min |
| `integration/domains_mosaicity_10p0deg/01_inspect_integrate.ipynb` | 5.5 min |
| `texture_tomography/domains_mosaicity_10p0deg/textomo_adaptive.ipynb` | 1.0 min |

**Note:** the 10.0 deg TT-Uniform reconstruction was not rerun on 2026-10-02 (the job was stopped once the other stages had matched the 2026-10-01 times); its total above therefore leaves the TT reconstruction out. On 2026-10-01 it took 1.55 h (93 min).

## Compared with 2026-10-01 (same node, diffractom before the memory changes)

| Sample | Pipeline | Stage | 2026-10-01 | 2026-10-02 |
|---|---|---|---|---|
| 1p0deg | adaptive | Peak segmentation | 3.2 min | 3.5 min |
| 1p0deg | adaptive | Peak indexing | 3.7 min | 3.7 min |
| 1p0deg | adaptive | Integration | 5.8 min | 5.0 min |
| 1p0deg | uniform | Integration | 5.8 min | 5.0 min |
| 1p0deg | adaptive | TT reconstruction | 0.6 min | 0.6 min |
| 1p0deg | uniform | TT reconstruction | 1.54 h (93 min) | 1.56 h (94 min) |
| 10p0deg | adaptive | Peak segmentation | 3.7 min | 4.0 min |
| 10p0deg | adaptive | Peak indexing | 15.0 min | 14.0 min |
| 10p0deg | adaptive | Integration | 6.5 min | 5.5 min |
| 10p0deg | uniform | Integration | 6.5 min | 5.5 min |
| 10p0deg | adaptive | TT reconstruction | 1.5 min | 1.0 min |
| 10p0deg | uniform | TT reconstruction | 1.55 h (93 min) | not rerun |
