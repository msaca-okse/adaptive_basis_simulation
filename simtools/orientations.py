import itertools

import numpy as np


def cubic_symmetry_operators():
    """The 24 proper rotations of the cube: signed permutation matrices with det +1."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            S = np.zeros((3, 3))
            S[range(3), perm] = signs
            if np.isclose(np.linalg.det(S), 1):
                ops.append(S)
    return np.array(ops)


def min_misorientation_deg(U_query, U_basis, chunk=256):
    """
    For each orientation in U_query (N, 3, 3), the misorientation angle in degrees
    to the closest orientation in U_basis (K, 3, 3), under cubic symmetry.
    Orientations map crystal to lab coordinates (v_lab = U @ v_crystal), so
    symmetry acts from the right: U and U @ S are the same orientation.
    """
    S = cubic_symmetry_operators()
    B = np.einsum("kij,sjl->ksil", U_basis, S).reshape(-1, 3, 3)  # all equivalents
    out = np.empty(len(U_query))
    for a in range(0, len(U_query), chunk):
        Q = U_query[a : a + chunk]
        # trace(Q^T B) for every pair
        tr = np.einsum("nji,mji->nm", Q, B)
        cos = np.clip((tr.max(axis=1) - 1) / 2, -1, 1)
        out[a : a + chunk] = np.degrees(np.arccos(cos))
    return out

