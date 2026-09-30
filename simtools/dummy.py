import pickle

import numpy as np
import ImageD11.sinograms.geometry as geometry


def load_dset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def add_pbp_columns(colf, dty_precision=7):
    """
    Add the columns point-by-point indexing needs (dtyi, cosomega, sinomega)
    and return the translation grid. The rotation axis is at the central
    translation (dty = 0 in the simulation).

    Returns:
        ypositions, ystep, ymin, y0
    """
    dtyr = np.round(colf.dty, dty_precision)
    ypositions = np.unique(dtyr)
    ystep = ypositions[1] - ypositions[0]
    ymin = np.min(dtyr)
    colf.addcolumn(geometry.dty_to_dtyi(colf.dty, ystep, ymin), "dtyi")
    colf.dty[:] = dtyr[:]
    colf.addcolumn(np.cos(np.radians(colf.omega)), "cosomega")
    colf.addcolumn(np.sin(np.radians(colf.omega)), "sinomega")
    y0 = ypositions[len(ypositions) // 2]
    return ypositions, ystep, ymin, y0


class dset(object):
    """Minimal stand-in for ImageD11.sinograms.dataset.DataSet, providing what the
    point-by-point indexing and refinement (ImageD11.sinograms.point_by_point) read from a
    dataset, for data that exist only as one merged peak file.

    Bins are guessed from the peaks, so ny counts the translations that have
    peaks (the two outermost translations on each side of the simulated scans are empty).
    """

    def __init__(self, colf, parfile, savefile, ny, nomega):
        self.parfile = parfile
        self.omega = colf.omega.copy()
        self.dty = colf.dty.copy()
        self.icolfile = savefile + "_pbp.h5"
        self.path = savefile + "_dummy_dset.pkl"
        self.shape = (ny, nomega)
        self.guessbins()

        _step_grid = geometry.step_grid_from_ybincens(self.ybincens, self.ystep, 1, y0=0)
        si, sj = np.array(_step_grid).T
        sx, sy = geometry.step_to_sample(si, sj, self.ystep)
        self.sx_grid = sx.reshape(ny, ny)
        self.sy_grid = sy.reshape(ny, ny)

    def update_colfile_pars(self, cf, phase_name=None):  # as in ImageD11.sinograms.dataset
        """Load parameters and update geometry for colfile"""
        cf.parameters.loadparameters(self.parfile, phase_name=phase_name)
        cf.updateGeometry()

    def guessbins(self):  # as in ImageD11.sinograms.dataset
        ny, nomega = self.shape
        self.omin = self.omega.min()
        self.omax = self.omega.max()
        if (self.omax - self.omin) > 360:
            # multi-turn scan...
            self.omin = 0.0
            self.omax = 360.0
            self.omega_for_bins = self.omega % 360
        else:
            self.omega_for_bins = self.omega
        self.ostep = (self.omax - self.omin) / (nomega - 1)
        self.ymin = self.dty.min()
        self.ymax = self.dty.max()
        if ny > 1:
            self.ystep = (self.ymax - self.ymin) / (ny - 1)
        else:
            self.ystep = 1
        self.obincens = np.linspace(self.omin, self.omax, nomega)
        self.ybincens = np.linspace(self.ymin, self.ymax, ny)
        self.obinedges = np.linspace(self.omin - self.ostep / 2, self.omax + self.ostep / 2, nomega + 1)
        self.ybinedges = np.linspace(self.ymin - self.ystep / 2, self.ymax + self.ystep / 2, ny + 1)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
