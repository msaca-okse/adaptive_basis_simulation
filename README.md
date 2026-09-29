# Adaptive basis texture tomography of simulated data

This repository accompanies the manuscript:

"Bridging powder and multi-crystal diffraction with basis-adaptive texture tomography"
Martin Sæbye Carøe, Mads Allerup Carlsen, Felix Tristan Frankus, Adam André William Cretton,
Michela La Bella, Innokentiy Kantor, Mads Ry Vogel Jørgensen, Henning Friis Poulsen,
Jakob Sauer Jørgensen, Nils Axel Henningsson

## Overview

This code does texture tomography reconstructions with an adaptive orientation basis on two
simulated scanning 3DXRD datasets of an aluminium sample, using the library *diffractom*
https://doi.org/10.5281/zenodo.20431767. The two datasets differ in the intra-granular
mosaicity of the simulated sample: 1 degree (`domains_mosaicity_1p0deg`) and 10 degrees
(`domains_mosaicity_10p0deg`). The real (DanMAX) dataset of the article is analysed in a
companion repository with the same layout.

The pipeline is a series of notebooks, one folder per dataset. The parameters are set and
checked in the notebooks themselves.

1. `adaptive_basis/<dataset>/segment_peaks`: segment diffraction peaks on every detector frame.
2. `adaptive_basis/<dataset>/indexing`: point-by-point indexing and refinement with ImageD11. The last notebook builds
   the adaptive basis, `adaptive_basis/<dataset>/basis.npy`.
3. `integration/<dataset>`: azimuthal integration of every frame over the selected powder rings.
4. `texture_tomography/<dataset>`: the texture tomography reconstruction with the adaptive basis.
5. `visualization/<dataset>`: the figure panels of the article.
6. `stability/domains_mosaicity_10p0deg`: stability of the adaptive-basis reconstruction with respect to the
   number of FISTA iterations (`01_iterations.ipynb`) and the kernel width sigma (`02_sigma.ipynb`).

## Installation

Install the "diffractom" library by following the instructions on "https://github.com/msaca-okse/diffractom".
In the process, you will create a conda environment. Now clone this repository and install the required
dependencies inside the conda environment:

```bash
git clone <this repository>
cd adaptive_basis_simulations
python -m pip install -r requirements.txt
```

The integration and the texture tomography run on the GPU through OpenCL.

## Data

The simulated datasets will be made available on Zenodo; a link will be added here.
Place them in `data/` so that the layout is

```
data/
  Eiger_4M.poni                       pyFAI geometry of the simulated detector
  samples/                            ground-truth samples (mesh, orientations) and AL.cif
  scans/domains_mosaicity_1p0deg/     scan_<i>.1.h5, one file per translation
  scans/domains_mosaicity_10p0deg/
```

Each scan file holds the 360 detector frames of one translation (omega = 0 to 179.5 degrees in
steps of 0.5 degrees) in a sparse (COO) format, together with the ground truth of the simulated
sample in `metadata/`. Intermediate outputs of the notebooks are written to `processed/<dataset>/`.

## Running the code

Run the notebooks of a dataset in order:

```
adaptive_basis/<dataset>/segment_peaks/01_find_parameters.ipynb
adaptive_basis/<dataset>/segment_peaks/02_segment.ipynb
adaptive_basis/<dataset>/segment_peaks/03_inspect.ipynb
adaptive_basis/<dataset>/indexing/01_inspect_peaks.ipynb
adaptive_basis/<dataset>/indexing/02_pbp_index.ipynb
adaptive_basis/<dataset>/indexing/03_refine.ipynb
integration/<dataset>/01_inspect_integrate.ipynb
texture_tomography/<dataset>/textomo_adaptive.ipynb
```

The adaptive bases used in the article are included (`adaptive_basis/<dataset>/basis.npy`), so the
first six notebooks can be skipped. Segmentation, point-by-point indexing and integration use many CPU
cores; they were run on a cluster node with 32 cores.

`simtools/` holds the code shared by the notebooks (reading the sparse scans, segmentation,
point-by-point helpers, figure maps, the stability sweeps), and `integration/frame_loader.py` holds the reader used by the integration.

## Citation

If you use this repository, please also cite the underlying software it builds on:

Carøe, Martin Sæbye (2026). *diffractom*. Zenodo. https://doi.org/10.5281/zenodo.20431767
