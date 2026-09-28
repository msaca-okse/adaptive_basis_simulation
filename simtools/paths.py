import os

# Everything is resolved relative to the repository, so the notebooks run
# from any working directory inside it.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.join(REPO, "data")  # simulated data (from Zenodo), git-ignored
PROCESS = os.path.join(REPO, "processed")  # intermediate outputs, git-ignored

SAMPLES = ("domains_mosaicity_1p0deg", "domains_mosaicity_10p0deg")

PARFILE = os.path.join(REPO, "adaptive_basis", "Al.par")  # ImageD11 geometry + unit cell
PONI_PATH = os.path.join(DATA, "Eiger_4M.poni")  # pyFAI geometry
CIF_PATH = os.path.join(DATA, "samples", "AL.cif")


def scan_dir(sample):
    """Folder holding the scan_<i>.1.h5 files (one per translation) of a sample."""
    _check_sample(sample)
    return os.path.join(DATA, "scans", sample)


def process_dir(sample):
    """Folder for intermediate outputs of a sample (created on first use)."""
    _check_sample(sample)
    path = os.path.join(PROCESS, sample)
    os.makedirs(path, exist_ok=True)
    return path


def peaks_path(sample):
    """Segmented 2D peaks for all translations, as an ImageD11 columnfile."""
    return os.path.join(process_dir(sample), "peaks_2d.h5")


def segmenter_options_path(sample):
    """Segmentation parameters chosen in 01_find_parameters. Kept in git."""
    _check_sample(sample)
    return os.path.join(REPO, "adaptive_basis", sample, "segment_peaks", "segmenter_options.json")


def basis_path(sample):
    """The adaptive orientation basis, (K, 3, 3) rotation matrices. Kept in git."""
    _check_sample(sample)
    return os.path.join(REPO, "adaptive_basis", sample, "basis.npy")


def integrated_path(sample):
    """Integrated data, without suffix (see integration/integrated_data.py)."""
    return os.path.join(process_dir(sample), sample + "_integrated")


def _check_sample(sample):
    if sample not in SAMPLES:
        raise ValueError(f"Unknown sample {sample!r}, expected one of {SAMPLES}")
