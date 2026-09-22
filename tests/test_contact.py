"""Checks on the vendored contact-field reconstruction.

These pin the two things the gate depends on: that a patch of known pressure and area integrates
back to the force that produced it, and that the stiffness is taken from shape B of each pair.
Getting that pairing wrong under-reported force by ~170x upstream, behind a patch that still
looked right, so it is worth a test rather than a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from grind_sim.pressure import (
    effective_stiffness,
    face_geometry_from_vertices,
    face_pressure,
    reconstruct_faces,
)


def square_patch(side: float, n: int) -> np.ndarray:
    """Return ``[N, 3, 3]`` triangle vertices tiling a flat square of the given side at z=0."""
    xs = np.linspace(-side / 2, side / 2, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a = (xs[i], xs[j], 0.0)
            b = (xs[i + 1], xs[j], 0.0)
            c = (xs[i + 1], xs[j + 1], 0.0)
            d = (xs[i], xs[j + 1], 0.0)
            tris.append([a, b, c])
            tris.append([a, c, d])
    return np.array(tris, dtype=np.float64)


def test_face_geometry_recovers_area_and_normal():
    verts = square_patch(0.1, 4)
    geom = face_geometry_from_vertices(verts)
    assert geom.area.sum() == pytest.approx(0.01)
    assert np.allclose(np.abs(geom.normal[:, 2]), 1.0)


def test_degenerate_triangles_get_zero_area_and_a_finite_normal():
    verts = np.zeros((1, 3, 3))
    geom = face_geometry_from_vertices(verts)
    assert geom.area[0] == 0.0
    assert np.all(np.isfinite(geom.normal))


def test_uniform_patch_integrates_to_its_force():
    """A 10 cm square at 1 mm depth and kh=1e6 carries kh*depth*area = 10 N."""
    verts = square_patch(0.1, 8)
    n_faces = len(verts)
    points = verts.reshape(-1, 3)
    depth = np.full(n_faces, -1.0e-3)  # Newton reports depth negative inside the overlap
    shape_pair = np.tile(np.array([0, 1]), (n_faces, 1))
    shape_kh = np.array([1.0e12, 1.0e6])  # rigid shape a, compliant shape b

    pressure, area, _, _, _ = reconstruct_faces(points, depth, shape_pair, shape_kh, n_faces)
    assert float(np.sum(pressure * area)) == pytest.approx(1.0e6 * 1.0e-3 * 0.01)
    assert np.allclose(pressure, 1.0e3)


def test_stiffness_comes_from_shape_b_not_shape_a():
    verts = square_patch(0.1, 2)
    n_faces = len(verts)
    points = verts.reshape(-1, 3)
    depth = np.full(n_faces, -1.0e-3)
    shape_kh = np.array([1.0e12, 1.0e6])

    b_second, area, *_ = reconstruct_faces(points, depth, np.tile([0, 1], (n_faces, 1)),
                                           shape_kh, n_faces)
    b_first, _, *_ = reconstruct_faces(points, depth, np.tile([1, 0], (n_faces, 1)),
                                       shape_kh, n_faces)
    # Swapping the pair swaps which stiffness is used: a million-fold difference, not a rounding.
    assert b_second[0] == pytest.approx(1.0e3)
    assert b_first[0] == pytest.approx(1.0e9)


def test_series_stiffness_matches_newtons_own_definition():
    assert effective_stiffness(np.array(2.0), np.array(3.0)) == pytest.approx(1.2)
    assert effective_stiffness(np.array(0.0), np.array(0.0)) == 0.0


def test_pressure_uses_the_depth_magnitude():
    assert face_pressure(np.array(-2.0e-3), np.array(1.0e6)) == pytest.approx(2.0e3)
    assert face_pressure(np.array(2.0e-3), np.array(1.0e6)) == pytest.approx(2.0e3)


def test_empty_patch_returns_empty_arrays():
    pressure, area, centroid, normal, pairs = reconstruct_faces(
        np.zeros((0, 3)), np.zeros(0), np.zeros((0, 2)), np.array([1.0]), 0
    )
    assert len(pressure) == len(area) == len(centroid) == len(normal) == len(pairs) == 0
