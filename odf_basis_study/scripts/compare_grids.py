"""Orientation grids in the cubic fundamental zone: the current one (random rotations mapped into the
FZ, greedy pruning at theta) vs cubochoric (orix.sampling.get_sample_fundamental). Metrics:
N, nearest-neighbour distance (packing), and the covering distance: misorientation from uniformly
random test orientations to the nearest grid orientation (percentiles; max = covering radius)."""
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from orix.quaternion.symmetry import Oh
from orix.sampling import get_sample_fundamental
from diffractom import Grid
from diffractom.crystallography import point_groups

SYM = Rotation.concatenate(point_groups.cubic)  # 24 proper rotations


def equivalents(q):
    """All 24 x 2 quaternions (x, y, z, w) equivalent to the rotations q (N, 4)."""
    R = Rotation.from_quat(q)
    out = np.concatenate([(R * s).as_quat() for s in SYM])
    return np.concatenate([out, -out])


def angle_from_chord(d):
    return np.degrees(4 * np.arcsin(np.clip(d / 2, 0, 1)))


def metrics(q, n_test=200_000, seed=1):
    tree = cKDTree(equivalents(q))
    test = Rotation.random(n_test, random_state=seed).as_quat()
    d, _ = tree.query(test)
    cov = angle_from_chord(d)
    d2, _ = tree.query(q, k=2)          # nearest other point (k=1 is itself)
    nn = angle_from_chord(d2[:, 1])
    return {"N": len(q), "nn_min": nn.min(), "nn_median": np.median(nn),
            "cov_median": np.median(cov), "cov_p99": np.percentile(cov, 99), "cov_max": cov.max()}


def current(n_random, theta):
    g = Grid.from_random_fundamental_zone(n_random, "cubic", sigma=0.01)
    g.prune_close_orientations(theta_deg=theta)
    return Rotation.concatenate(g.rotations_at_level(0)).as_quat()


def cubochoric(res):
    r = get_sample_fundamental(resolution=res, point_group=Oh, method="cubochoric")
    w = r.data  # (N, 4) w, x, y, z
    return np.ascontiguousarray(w[:, [1, 2, 3, 0]])


rows = []
for name, make in [
    ("current 150k/1.0", lambda: current(150_000, 1.0)),          # the uniform notebook
    ("current 600k/1.0", lambda: current(600_000, 1.0)),
    ("cubochoric 2.0", lambda: cubochoric(2.0)),
    ("cubochoric 1.5", lambda: cubochoric(1.5)),
    ("cubochoric 1.25", lambda: cubochoric(1.25)),
    ("cubochoric 1.0", lambda: cubochoric(1.0)),
]:
    t0 = time.perf_counter()
    q = make()
    t1 = time.perf_counter()
    m = metrics(q)
    m.update(name=name, t_make=t1 - t0)
    rows.append(m)
    print(f"{name:20s} N {m['N']:8d}  nn min {m['nn_min']:.2f} med {m['nn_median']:.2f}  "
          f"cover med {m['cov_median']:.2f} p99 {m['cov_p99']:.2f} max {m['cov_max']:.2f} deg  "
          f"(made in {m['t_make']:.0f} s)", flush=True)
