import ImageD11.grain
import matplotlib.pyplot as plt
import numpy as np
from ImageD11.nbGui import nb_utils as utils
from scipy.ndimage import binary_fill_holes
from xfab import symmetry


class MultiChannelMap(object):
    """
    A point-by-point map with several candidate grains ("channels") per voxel,
    unpacked from the flat ImageD11 PBPMap into (NY, NY, nchannels, ...) arrays
    and sorted so that channel 0 is the best candidate.
    """

    def __init__(self, pbpmap, ucell):
        self.completeness = None
        self.pbpmap = pbpmap
        self.ucell = ucell
        NY = pbpmap.NY
        # find the max number of channels
        counter = np.zeros(shape=(NY, NY), dtype=int)
        for i, j in zip(pbpmap.i_shift, pbpmap.j_shift):
            counter[i, j] += 1
        self.nchannels = np.max(counter)

        self.ubi = np.zeros(shape=(NY, NY, self.nchannels, 3, 3))
        self.u = np.zeros(shape=(NY, NY, self.nchannels, 3, 3))
        self.ntotal = np.zeros(shape=(NY, NY, self.nchannels))
        self.nuniq = np.zeros(shape=(NY, NY, self.nchannels))
        self.rgb = np.zeros(shape=(NY, NY, self.nchannels, 3, 3))
        self.grain = np.empty(shape=(NY, NY, self.nchannels), dtype=object)
        self.has_strain = "eps00" in pbpmap.titles
        if self.has_strain:
            self.refined_strain = np.zeros((NY, NY, self.nchannels, 3, 3))

        all_grains = []
        counter = np.zeros(shape=(NY, NY), dtype=int)
        for k, (i, j) in enumerate(zip(pbpmap.i_shift, pbpmap.j_shift)):
            c = counter[i, j]
            self.ubi[i, j, c] = pbpmap.ubi[..., k]
            self.ntotal[i, j, c] = pbpmap.ntotal[..., k]
            self.nuniq[i, j, c] = pbpmap.nuniq[..., k]
            g = ImageD11.grain.grain(pbpmap.ubi[..., k])
            g.ref_unitcell = ucell
            self.grain[i, j, c] = g
            all_grains.append(g)
            if self.has_strain:
                self.refined_strain[i, j, c] = pbpmap.eps[..., k]
            self.u[i, j, c] = g.u
            counter[i, j] += 1

        utils.get_rgbs_for_grains(all_grains)
        counter = np.zeros(shape=(NY, NY), dtype=int)
        for k, (i, j) in enumerate(zip(pbpmap.i_shift, pbpmap.j_shift)):
            self.rgb[i, j, counter[i, j], 0] = all_grains[k].rgb_x
            self.rgb[i, j, counter[i, j], 1] = all_grains[k].rgb_y
            self.rgb[i, j, counter[i, j], 2] = all_grains[k].rgb_z
            counter[i, j] += 1

        self.sort()

    def normalize_per_grain(self, segmap):
        """Completeness = unique peaks relative to the best voxel of the same grain."""
        self.completeness = np.zeros_like(self.nuniq)
        for label in np.unique(segmap):
            if ~np.isnan(label):
                m = segmap == label
                norm = np.max(self.nuniq[m, 0])
                self.completeness[m] = self.nuniq[m] / norm

    def sort(self, primary=None, secondary=None):
        """Sort channels per voxel, descending. Default: by completeness if it
        has been computed, otherwise by unique peaks; ties broken by total peaks."""
        if primary is None:
            primary = -self.nuniq if self.completeness is None else -self.completeness
        if secondary is None:
            secondary = -self.ntotal

        NY = self.pbpmap.NY
        for i in range(NY):
            for j in range(NY):
                s = np.array(
                    list(zip(primary[i, j, :], secondary[i, j, :])),
                    dtype=[("nuniq", "i4"), ("ntotal", "i4")],
                )
                index = np.argsort(s, order=["nuniq", "ntotal"])
                self.ubi[i, j] = self.ubi[i, j, index]
                self.ntotal[i, j] = self.ntotal[i, j, index]
                self.nuniq[i, j] = self.nuniq[i, j, index]
                self.grain[i, j] = self.grain[i, j, index]
                self.rgb[i, j] = self.rgb[i, j, index]
                self.u[i, j] = self.u[i, j, index]
                if self.completeness is not None:
                    self.completeness[i, j] = self.completeness[i, j, index]
                if self.has_strain:
                    self.refined_strain[i, j] = self.refined_strain[i, j, index]

    def ipf_voxel(self, i, j):
        plt.style.use("default")
        grains = [g for g in self.grain[i, j, :] if g is not None]
        if len(grains) == 0:
            print("No grains")
            return
        utils.plot_all_ipfs(grains)
        print("nuniq: ", self.nuniq[i, j, 0 : len(grains)])
        plt.show()


def cubic_misorientation_deg(U, u):
    """Smallest misorientation angle (degrees) between two orientation matrices, cubic symmetry."""
    return np.min(symmetry.Umis(U, u, crystal_system=7)[:, 1])


def flood_fill(
    property_map,
    distance_metric,
    footprint,
    local_disorientation_tolerance,
    mask,
    background_value=np.nan,
    fill_holes=False,
    max_grains=99,
    min_grain_size=0,
    seed=0,
    verbose=False,
):
    """
    Segment a map into grains by flood filling from random seed points: a
    neighbour joins the grain if distance_metric(current, neighbour) is below
    local_disorientation_tolerance. The random seed points are drawn from a
    generator seeded with `seed`, so the segmentation is reproducible.
    """
    assert footprint.shape[0] % 2 != 0
    assert footprint.shape[1] % 2 != 0
    rng = np.random.default_rng(seed)

    M, N = property_map.shape[0], property_map.shape[1]
    segmentation = np.zeros((M, N))
    skips = np.ones((M, N), dtype=bool)
    label = 1
    done = False
    iteration = 0

    while not done and iteration < max_grains:
        rows, cols = np.where(mask * (segmentation == 0) * skips)
        if len(rows) > 0:
            n = rng.integers(0, len(rows))
            seed_point = (rows[n], cols[n])
            flood_mask = np.zeros((M, N), dtype=bool)
            grain_mask = _flood(
                property_map, distance_metric, seed_point, footprint,
                local_disorientation_tolerance, mask, flood_mask,
            )
            if fill_holes:
                grain_mask = binary_fill_holes(grain_mask)
            if np.sum(grain_mask) > min_grain_size:
                segmentation[grain_mask] = label
                label += 1
            else:
                skips[grain_mask] = False
            iteration += 1
            if verbose:
                print(f"Iteration {iteration}, found grain with {np.sum(grain_mask)} voxels")
        else:
            done = True

    segmentation[segmentation == 0] = background_value
    return segmentation


def _flood(property_map, distance_metric, seed_point, footprint, tolerance, mask, flood_mask):
    m = footprint.shape[0] // 2
    n = footprint.shape[1] // 2
    i, j = seed_point
    flood_mask[i, j] = True
    if mask[i, j] == 0:
        raise ValueError("Seed point not in mask")

    unchartered_indices = [seed_point]
    while len(unchartered_indices) > 0:
        i, j = unchartered_indices.pop()
        U = property_map[i, j]
        for k in range(footprint.shape[0]):
            for l in range(footprint.shape[1]):
                if footprint[k, l] > 0:
                    row = i - m + k
                    col = j - n + l
                    if not flood_mask[row, col] and mask[row, col]:
                        if distance_metric(U, property_map[row, col]) < tolerance:
                            flood_mask[row, col] = True
                            unchartered_indices.insert(0, (row, col))
    return flood_mask


def segment_grains(multi_channel, mask, local_disorientation_tolerance, max_grains, min_grain_size, seed=0):
    """Flood-fill segmentation of the channel-0 orientation map (padded by one voxel)."""
    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    property_map = np.pad(multi_channel.u[:, :, 0], ((1, 1), (1, 1), (0, 0), (0, 0)))
    padded_mask = np.pad(mask, ((1, 1), (1, 1)))
    segmap = flood_fill(
        property_map=property_map,
        distance_metric=cubic_misorientation_deg,
        footprint=footprint,
        local_disorientation_tolerance=local_disorientation_tolerance,
        mask=padded_mask,
        max_grains=max_grains,
        min_grain_size=min_grain_size,
        seed=seed,
    )
    return segmap[1:-1, 1:-1]
