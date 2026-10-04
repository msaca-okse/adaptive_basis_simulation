"""The real Al1050 problem exactly as texture_tomography/texture_tomography/textomo_adaptive.ipynb builds it
(data reduction, ring scaling, weights, geometry, material, the rotated adaptive basis). The reduced data
are cached here (arr.npy, 1.5 GB)."""
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = "/zhome/71/c/146676/texture_tomography"
sys.path.insert(0, REPO)
from integration.frame_loader import OUTPUT_PATH, PROCESS  # noqa: E402
from integration.integrated_data import IntegratedData  # noqa: E402
from diffractom import Material  # noqa: E402
from diffractom.operators.single_phase_forward_operator import group_reflections_into_rings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_FRAMES, OMEGA_BIN, OMEGA_OFFSET_DEG = 3000, 6, 50.0


def load():
    data = IntegratedData(OUTPUT_PATH)
    assert data.polarization_corrected is True
    positions = np.round(data.translations, 4)
    unique_positions = np.unique(positions)
    cache = os.path.join(HERE, "arr.npy")
    if os.path.exists(cache):
        arr = np.load(cache, mmap_mode="r")
        arr = np.ascontiguousarray(arr)
    else:
        t0 = time.perf_counter()
        n_omega = N_FRAMES // OMEGA_BIN
        reduced = np.zeros((len(unique_positions), n_omega, data.n_eta, data.n_rings), dtype=np.float32)
        count = np.zeros(len(unique_positions))
        for t in range(data.n_translations):
            u = np.searchsorted(unique_positions, positions[t])
            reduced[u] += data.I[t, :N_FRAMES].reshape(n_omega, OMEGA_BIN, data.n_eta, data.n_rings).sum(axis=1)
            count[u] += 1
        reduced /= count[:, None, None, None].astype(np.float32)
        arr = np.ascontiguousarray(reduced.transpose(1, 0, 2, 3))
        del reduced
        ring_sums = arr.sum(axis=(0, 1, 2), dtype=np.float64)
        arr *= (ring_sums.sum() / ring_sums).astype(np.float32)
        np.save(cache, arr)
        print(f"reduced in {time.perf_counter() - t0:.0f} s", flush=True)
    N_Omega, My, N_eta, N_theta = arr.shape
    eta = data.eta_deg
    weights = np.ones((N_eta, N_theta), dtype=np.float32)
    weights[np.abs(np.abs(eta) - 90) < 10, :] = 0
    profile = arr.mean(axis=(0, 1))
    weights[profile < 0.05 * np.median(profile, axis=0, keepdims=True)] = 0
    rotation_axis_mm = float(np.load(os.path.join(PROCESS, "al1050_15pct_center_slice", "rotation_axis_com.npy")))
    translation_step_mm = float(np.median(np.diff(unique_positions)))
    omega_step = float(np.median(np.diff(data.omega_deg)))
    cfg = {
        "energy": 12.398 / data.wavelength_A, "Nx": 115, "Ny": 115, "My": My, "N_Omega": N_Omega, "N_eta": N_eta,
        "angle_range": [-OMEGA_OFFSET_DEG, N_FRAMES * omega_step - OMEGA_OFFSET_DEG],
        "cor_offset": -rotation_axis_mm / translation_step_mm, "eta_angle_range": [0, 360],
        "j_direction_0": [0, -1, 0], "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
        "detector_direction_origin": [0, 1, 0], "detector_direction_positive_90": [0, 0, 1],
    }
    lattice = data.material["lattice_params"]
    mat = Material.from_lattice_parameters(
        a=lattice["a"], b=lattice["b"], c=lattice["c"], alpha=lattice["alpha"], beta=lattice["beta"],
        gamma=lattice["gamma"], symmetry_group="cubic", wavelength_A=data.wavelength_A, min_two_theta=0.0,
        max_two_theta=0.0, hkl_list=np.array([hkl for ring in data.rings for hkl in ring["hkl"]]))
    rings = group_reflections_into_rings(mat)
    assert len(rings) == N_theta
    basis = np.load(os.path.join(REPO, "adaptive_basis", "basis.npy"))
    basis = R.from_euler("z", OMEGA_OFFSET_DEG, degrees=True).as_matrix() @ basis
    return {"data": arr.reshape(N_Omega, My, N_eta * N_theta), "weights": weights, "cfg": cfg, "material": mat,
            "adaptive_basis": basis, "huber_delta": 5.3e4, "eta_deg": eta, "shape4": arr.shape}
