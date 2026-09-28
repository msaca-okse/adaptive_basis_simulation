import glob
import os
import re

import h5py
import ImageD11.columnfile
import ImageD11.parameters
import numpy as np
from scipy.sparse import coo_matrix

from . import paths

# Layout of one scan_<i>.1.h5 file (one translation, i.e. one dty position):
#   <i>.1/eiger/frames/{intensity,row,col}  concatenated sparse (COO) frames
#   <i>.1/eiger/frames/nnz                  (n_omega,) non-zeros per frame
#   <i>.1/eiger/frames/{omega,y}            (n_omega,) motor positions
#   metadata/...                            detector, wavelength and ground truth
FRAMES = "eiger/frames"


def scan_files(sample):
    """All scan files of a sample, ordered by scan index (= translation order)."""
    files = glob.glob(os.path.join(paths.scan_dir(sample), "scan_*.1.h5"))
    files.sort(key=lambda p: int(re.search(r"scan_(\d+)\.1\.h5$", p).group(1)))
    if not files:
        raise FileNotFoundError(f"No scan files in {paths.scan_dir(sample)}")
    return files


def scan_key(h5file):
    """The single data group of a scan file, e.g. '52.1'."""
    keys = [k for k in h5file.keys() if k != "metadata"]
    assert len(keys) == 1, f"Expected one scan group, found {keys}"
    return keys[0]


class SparseScan:
    """
    One translation: all sparse frames are read into memory (tens of MB),
    and frames are densified on request.

        with SparseScan(path) as scan:
            scan.omega, scan.dty       # (n_omega,) arrays, dty is constant
            scan.frame(i)              # dense (nrows, ncols) float64 array
    """

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        with h5py.File(self.path, "r") as f:
            g = f[scan_key(f)][FRAMES]
            self.intensity = g["intensity"][:]
            self.row = g["row"][:].astype(np.int64)
            self.col = g["col"][:].astype(np.int64)
            self.nnz = g["nnz"][:]
            self.omega = g["omega"][:]
            y = g["y"][:]
            self.shape = (int(f["metadata/eiger/nrows"][()]), int(f["metadata/eiger/ncols"][()]))
        if not np.allclose(y, y[0]):
            raise ValueError(f"{self.path}: y varies within the scan")
        self.dty = float(y[0])
        self.bounds = np.concatenate(([0], np.cumsum(self.nnz)))
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    @property
    def n_omega(self):
        return len(self.omega)

    def frame(self, i):
        a, b = self.bounds[i], self.bounds[i + 1]
        # coo_matrix sums duplicate entries, same as textom.sparse.densify
        return coo_matrix(
            (self.intensity[a:b], (self.row[a:b], self.col[a:b])), shape=self.shape
        ).toarray()


def read_metadata(sample):
    """Detector, beam and sample metadata (identical in every scan file)."""
    with h5py.File(scan_files(sample)[0], "r") as f:
        m = f["metadata"]
        return {
            "wavelength_A": float(m["wavelength"][()]),
            "distance_um": float(m["eiger/distance"][()]),
            "pixelsize_um": float(m["eiger/pixelsize_y"][()]),
            "detector_shape": (int(m["eiger/nrows"][()]), int(m["eiger/ncols"][()])),
            "omega_step_deg": float(m["motors/omegastepsize"][()]),
            "reference_cell": m["sample/lattice/reference_cell"][()],
            "space_group": m["sample/lattice/sgname"][()].decode(),
        }


def read_ground_truth(sample):
    """Simulated sample: tetrahedral mesh element centroids, orientations and strains."""
    with h5py.File(scan_files(sample)[0], "r") as f:
        m = f["metadata/sample"]
        vertices = m["mesh/vertices"][:]
        cells = m["mesh/cells"][:]
        return {
            "centroids": vertices[cells].mean(axis=1),
            "orientation": m["lattice/orientation"][:],
            "strain": m["lattice/strain"][:],
        }


def write_peaks(pks, h5path, parfile=paths.PARFILE):
    """Write a dict of peak columns (fc, sc, omega, dty, sum_intensity,
    Number_of_pixels) to an ImageD11 columnfile in hdf5 format."""
    if os.path.exists(h5path):
        raise ValueError(f"File already exists : {h5path}")
    colf = ImageD11.columnfile.colfile_from_dict(pks)
    colf.setparameters(ImageD11.parameters.read_par_file(parfile))
    colf.updateGeometry()
    ImageD11.columnfile.colfileobj_to_hdf(colf, h5path)
    return colf


def read_peaks(h5path, parfile=paths.PARFILE):
    """Read a columnfile and compute the diffraction geometry with parfile."""
    colf = ImageD11.columnfile.columnfile(h5path)
    colf.setparameters(ImageD11.parameters.read_par_file(parfile))
    colf.updateGeometry()
    return colf
