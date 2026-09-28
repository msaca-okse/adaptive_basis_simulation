import os
import sys

import h5py
import numpy as np
from scipy.sparse import coo_matrix

# the notebooks import this module as integration.frame_loader, with the repository on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simtools import io, paths  # noqa: E402

PONI_PATH = paths.PONI_PATH
PARAMETERS_AL = paths.PARFILE
MASKFILE = None  # simulated detector: no bad pixels, no gaps

# Threshold (in dty units, um) used to check that a translation has a single dty value.
DTY_STEP_TOL = 1e-6

# The simulation (xrd_simulator, polarization=True) used a linearly polarized beam with
# polarization vector = lab y = the frames' column axis = pyFAI's chi = 0 direction, i.e.
# horizontal. pyFAI integrate2d convention: +1 horizontal, -1 vertical, 0 circular, None = off.
POLARIZATION_FACTOR = 1.0

# Output of the polarization-corrected integration, kept separate from
# paths.integrated_path(sample) (the uncorrected run) so the two can be compared.
OUTPUT_SUFFIX = "_polcorr"


def output_path(sample):
    """Polarization-corrected integrated data for this sample, without suffix."""
    return paths.integrated_path(sample) + OUTPUT_SUFFIX


class SimulatedScanFile:
    """
    One scan file = one translation. The frames are stored sparse (COO):
    concatenated intensity/row/col arrays plus the number of non-zeros per
    frame. The sparse arrays are small and read into memory on open; dense
    frames are built on request.
    """

    def __init__(self, path):
        self.path = path
        self._scan = None

    def open(self):
        self._scan = io.SparseScan(self.path).__enter__()
        if len(self._scan.omega) != len(self._scan.nnz):
            raise ValueError(f"{self.path}: frame/motor length mismatch")
        return self

    def close(self):
        self._scan = None

    @property
    def n_omega(self):
        return self._scan.n_omega

    @property
    def omega(self):
        return self._scan.omega

    @property
    def dty(self):
        return self._scan.dty

    @property
    def frame_shape(self):
        return self._scan.shape

    def frame(self, i_omega):
        return self._scan.frame(i_omega)

    def frames(self, omega_slice=slice(None)):
        idx = range(*omega_slice.indices(self.n_omega))
        return np.array([self._scan.frame(i) for i in idx])


def read_frame_batch(path, start, stop):
    """
    Standalone module-level function so it can be safely pickled/imported by
    worker processes. Opens its own file handle and returns dense frames
    start..stop of the translation stored in path.
    """
    with io.SparseScan(path) as scan:
        return np.array([scan.frame(i) for i in range(start, stop)], dtype=np.float32)


def read_batch_into_shm(path, start, stop, shm_name, batch_capacity, frame_shape, dtype_str):
    """
    Like read_frame_batch, but densifies directly into an existing shared
    memory buffer instead of returning a pickled array. Only the frame count
    crosses back over the process boundary. The caller creates the shared
    memory block (size = batch_capacity * prod(frame_shape) * itemsize) and
    reads buf[:n] afterwards, since the last batch is usually shorter.
    """
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        n = stop - start
        buf = np.ndarray((batch_capacity,) + tuple(frame_shape), dtype=np.dtype(dtype_str), buffer=shm.buf)
        with h5py.File(path, "r") as f:
            g = f[io.scan_key(f)][io.FRAMES]
            bounds = np.concatenate(([0], np.cumsum(g["nnz"][:])))
            a, b = bounds[start], bounds[stop]
            intensity = g["intensity"][a:b]
            row = g["row"][a:b].astype(np.int64)
            col = g["col"][a:b].astype(np.int64)
        bounds = bounds[start : stop + 1] - a
        for i in range(n):
            s = slice(bounds[i], bounds[i + 1])
            buf[i] = coo_matrix((intensity[s], (row[s], col[s])), shape=tuple(frame_shape)).toarray()
        return n
    finally:
        shm.close()


class SimulatedDataset:
    """
    s3dxrd loader for the simulated data. Every scan file holds one
    translation; translations are sorted by their actual dty value. The
    rotation-step count is checked to be constant across translations.

        with SimulatedDataset("domains_mosaicity_1p0deg") as ds:
            ds.n_translations
            ds.translations()            # (N_y,) dty values, sorted
            ds.omega(i_trans)            # omega values for that translation
            ds.frames(i_trans)           # (N_omega, My, Mx) dense frames
    """

    def __init__(self, sample):
        self.sample = sample
        self._files = [SimulatedScanFile(p) for p in io.scan_files(sample)]

    def __enter__(self):
        for f in self._files:
            f.open()
        self._files.sort(key=lambda f: f.dty)

        n_omega_values = {f.n_omega for f in self._files}
        if len(n_omega_values) > 1:
            raise ValueError(
                "Expected the same number of rotation steps in every translation; "
                f"found varying counts: {sorted(n_omega_values)}."
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        for f in self._files:
            f.close()

    # ---- generic loader interface expected by the notebooks ----

    @property
    def n_translations(self):
        return len(self._files)

    def n_omega(self, i_trans):
        return self._files[i_trans].n_omega

    def omega(self, i_trans):
        return self._files[i_trans].omega

    def translations(self):
        return np.array([f.dty for f in self._files])

    def frame(self, i_trans, i_omega):
        return self._files[i_trans].frame(i_omega)

    def frames(self, i_trans, omega_slice=slice(None)):
        return self._files[i_trans].frames(omega_slice)

    def raw_location(self, i_trans):
        """(file_path, frame_start, frame_stop) for this translation, so a
        separate process can read the frames itself (see read_batch_into_shm)."""
        f = self._files[i_trans]
        return f.path, 0, f.n_omega

    @property
    def frame_shape(self):
        return tuple(self._files[0].frame_shape)

    frame_dtype = "float32"

    def mask(self):
        """pyFAI convention: True = masked. Nothing is masked in the simulation."""
        return np.zeros(self.frame_shape, dtype=bool)

    def poni_path(self):
        return PONI_PATH


import matplotlib.pyplot as plt
def dark(fontsize=16):
    plt.style.use("dark_background")
    ticksize = fontsize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
