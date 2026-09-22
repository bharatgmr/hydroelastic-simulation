"""Load reduction: the force device pushes along the tool axis, the patch pushes back somewhere
else, and the offset between them is a moment the machine has to carry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grind_sim.loads import reduce_loads

UP = np.array([0.0, 0.0, 1.0])


def patch(centre_x: float, n: int = 400, pressure_pa: float = 1.0e4, side: float = 0.02):
    """A square patch of uniform pressure centred at ``centre_x`` on the z=0 surface."""
    step = side / n
    xs = centre_x + (np.arange(n) - (n - 1) / 2) * step  # symmetric about centre_x
    centroid = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
    normal = np.tile(UP, (n, 1))
    pressure = np.full(n, pressure_pa)
    area = np.full(n, side * step)
    return centroid, normal, pressure, area


def test_centred_patch_has_no_moment_arm():
    loads = reduce_loads(*patch(0.0), tool_origin=np.zeros(3), tool_axis=UP,
                         tcp_origin=np.zeros(3))
    assert loads.moment_arm == pytest.approx(0.0, abs=1e-9)
    assert loads.bending_moment_tool == pytest.approx(0.0, abs=1e-9)
    assert loads.force_axial == pytest.approx(loads.force_normal_surface)


def test_offset_patch_produces_moment_equal_to_force_times_arm():
    """The whole point of the correction: F applied on axis, reacted 60 mm away."""
    offset = 0.06
    loads = reduce_loads(*patch(offset), tool_origin=np.zeros(3), tool_axis=UP,
                         tcp_origin=np.zeros(3))
    assert loads.moment_arm == pytest.approx(offset, abs=1e-6)
    assert loads.bending_moment_tool == pytest.approx(loads.force_axial * offset, rel=1e-6)


def test_moment_about_the_tcp_differs_from_the_moment_about_the_tool():
    """Same patch, two reference points, two different moments — they are not interchangeable."""
    loads = reduce_loads(*patch(0.06), tool_origin=np.zeros(3), tool_axis=UP,
                         tcp_origin=np.array([0.06, 0.0, 0.0]))
    assert loads.bending_moment_tcp == pytest.approx(0.0, abs=1e-6)  # TCP sits at the patch
    assert loads.bending_moment_tool > 0.1


def test_tilted_tool_axis_splits_the_reaction():
    """With the tool tilted, what the device holds and what the part feels are different loads."""
    tilt = math.radians(20.0)
    axis = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    loads = reduce_loads(*patch(0.0), tool_origin=np.zeros(3), tool_axis=axis,
                         tcp_origin=np.zeros(3))
    assert loads.force_axial == pytest.approx(loads.force_normal_surface * math.cos(tilt),
                                              rel=1e-9)
    assert loads.force_axial < loads.force_normal_surface


def test_axial_offset_along_the_axis_is_not_a_moment_arm():
    """Sliding the reference point along the axis changes nothing about the arm."""
    low = reduce_loads(*patch(0.05), tool_origin=np.zeros(3), tool_axis=UP,
                       tcp_origin=np.zeros(3))
    high = reduce_loads(*patch(0.05), tool_origin=np.array([0.0, 0.0, 0.3]), tool_axis=UP,
                        tcp_origin=np.zeros(3))
    assert low.moment_arm == pytest.approx(high.moment_arm, abs=1e-9)
    assert low.bending_moment_tool == pytest.approx(high.bending_moment_tool, rel=1e-9)


def test_face_normals_are_oriented_before_summing():
    """Newton's normals are unsigned; a flipped patch must not report a negative load."""
    centroid, normal, pressure, area = patch(0.0)
    loads = reduce_loads(centroid, -normal, pressure, area, tool_origin=np.zeros(3),
                         tool_axis=UP, tcp_origin=np.zeros(3))
    assert loads.force_normal_surface > 0.0


def test_uniform_patch_integrates_to_pressure_times_area():
    centroid, normal, pressure, area = patch(0.0, pressure_pa=5.0e3)
    loads = reduce_loads(centroid, normal, pressure, area, tool_origin=np.zeros(3),
                         tool_axis=UP, tcp_origin=np.zeros(3))
    assert loads.force_normal_surface == pytest.approx(5.0e3 * area.sum())
    assert loads.pressure_mean == pytest.approx(5.0e3)
    assert loads.patch_area == pytest.approx(area.sum())


def test_empty_patch_reduces_to_zero():
    loads = reduce_loads(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                         tool_origin=np.zeros(3), tool_axis=UP, tcp_origin=np.zeros(3))
    assert loads.n_faces == 0
    assert loads.force_axial == 0.0
    assert loads.moment_arm == 0.0
