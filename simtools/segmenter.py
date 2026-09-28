import numpy as np
from ImageD11 import cImageD11

from .io import SparseScan


def to_uint16(frame, norm_factor):
    """Scale so that norm_factor maps to the top of the uint16 range, then clip.
    With norm_factor=None the frame is only converted to float32."""
    frame = frame.astype(np.float32)
    frame[np.isinf(frame)] = 0
    if norm_factor is None:
        return frame
    frame *= 65535 / norm_factor
    return frame.clip(0, 65535).astype(np.uint16)


def local_max(frame, threshold):
    """
    Segment a frame into local maxima; every pixel above threshold is assigned
    to the maximum it climbs to.

    Returns:
        fc, sc, sum_intensity, Number_of_pixels (numpy arrays, one entry per peak)
    """
    float_im = frame.astype(np.float32)
    labels = np.zeros(float_im.shape, np.int32)
    work = np.zeros(float_im.shape, np.int8)
    thresholded = np.where(float_im > threshold, float_im, 0)
    npks = cImageD11.localmaxlabel(thresholded, labels, work)
    blobs = cImageD11.blobproperties(thresholded, labels, npks, 0)
    cImageD11.blob_moments(blobs)
    return (
        blobs[:, cImageD11.f_raw],
        blobs[:, cImageD11.s_raw],
        blobs[:, cImageD11.s_I],
        blobs[:, cImageD11.s_1],
    )


def segment_frame(frame, options):
    return local_max(to_uint16(frame, options["norm_factor"]), options["threshold"])


def segment_scan(path, options):
    """
    Segment all frames of one translation. Peaks get omega at the centre of
    their rotation step (omega + ostep / 2), as the frames integrate over it.
    """
    cImageD11.cimaged11_omp_set_num_threads(1)
    cols = {k: [] for k in ("fc", "sc", "sum_intensity", "Number_of_pixels", "omega", "dty")}
    with SparseScan(path) as scan:
        ostep = scan.omega[1] - scan.omega[0]
        for i in range(scan.n_omega):
            fc, sc, si, npx = segment_frame(scan.frame(i), options)
            cols["fc"].append(fc)
            cols["sc"].append(sc)
            cols["sum_intensity"].append(si)
            cols["Number_of_pixels"].append(npx)
            cols["omega"].append(np.full(len(fc), scan.omega[i] + ostep / 2.0))
            cols["dty"].append(np.full(len(fc), scan.dty))
    return {k: np.concatenate(v) for k, v in cols.items()}
