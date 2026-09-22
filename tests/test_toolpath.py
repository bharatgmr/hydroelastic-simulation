"""The flat-part path must match the planner's spacing arithmetic, not merely look like a raster."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grind_sim.toolpath import PathParams, PlateSpec, hatch_path


def test_pitch_is_the_planners_formula():
    """requested_displacement = tool_contact_width * (1 - overlap/100)."""
    assert PathParams(tool_contact_width=0.1778, overlap_percent=0.0).pitch == pytest.approx(0.1778)
    assert PathParams(tool_contact_width=0.1778, overlap_percent=50.0).pitch == pytest.approx(0.0889)
    assert PathParams(tool_contact_width=0.1778, overlap_percent=75.0).pitch == pytest.approx(0.04445)


def test_pitch_rejects_nonsense():
    with pytest.raises(ValueError):
        PathParams(tool_contact_width=0.0).pitch
    with pytest.raises(ValueError):
        PathParams(overlap_percent=100.0).pitch


def test_line_spacing_equals_the_pitch():
    path = hatch_path(PlateSpec(0.3, 0.3), PathParams(overlap_percent=50.0))
    rows = {w.line_index: w.position[1] for w in path.waypoints}
    ys = np.array([rows[i] for i in sorted(rows)])
    assert np.allclose(np.diff(ys), path.params.pitch)


def test_point_spacing_along_a_line_is_the_pathgap():
    params = PathParams(pathgap_sanding_line=0.0075)
    path = hatch_path(PlateSpec(0.3, 0.3), params)
    line0 = [w.position for w in path.waypoints if w.line_index == 0]
    steps = np.linalg.norm(np.diff(np.array(line0), axis=0), axis=1)
    assert np.allclose(steps, 0.0075)


def test_every_waypoint_carries_the_planner_frame():
    path = hatch_path(PlateSpec(0.3, 0.3), PathParams())
    for w in path.waypoints:
        assert w.normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
        assert w.frame.rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)  # vz = -normal
        assert np.cross(w.frame.rotation[:, 0], w.frame.rotation[:, 1]) == pytest.approx(
            w.frame.rotation[:, 2], abs=1e-12)
        assert float(np.dot(w.travel, w.frame.rotation[:, 0])) == pytest.approx(0.0, abs=1e-12)


def test_serpentine_reverses_travel_on_alternate_lines():
    path = hatch_path(PlateSpec(0.3, 0.3), PathParams(alternate=True))
    first = next(w for w in path.waypoints if w.line_index == 0)
    second = next(w for w in path.waypoints if w.line_index == 1)
    assert float(np.dot(first.travel, second.travel)) == pytest.approx(-1.0, abs=1e-12)


def test_unidirectional_keeps_one_travel_direction():
    path = hatch_path(PlateSpec(0.3, 0.3), PathParams(alternate=False))
    travels = {tuple(np.round(w.travel, 9)) for w in path.waypoints}
    assert len(travels) == 1
    assert path.conventions["row_alternation"] == "unidirectional"


def test_waypoints_stay_on_the_plate():
    plate = PlateSpec(0.3, 0.25)
    path = hatch_path(plate, PathParams())
    points = path.positions
    assert np.all(np.abs(points[:, 0]) <= plate.size_x / 2 + 1e-9)
    assert np.all(np.abs(points[:, 1]) <= plate.size_y / 2 + 1e-9)
    assert np.allclose(points[:, 2], 0.0)


def test_hatch_angle_rotates_the_whole_pattern():
    straight = hatch_path(PlateSpec(0.3, 0.3), PathParams(hatch_angle_deg=0.0))
    turned = hatch_path(PlateSpec(0.3, 0.3), PathParams(hatch_angle_deg=90.0))
    assert straight.waypoints[0].travel == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert turned.waypoints[0].travel == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_tighter_overlap_puts_more_lines_on_the_same_plate():
    loose = hatch_path(PlateSpec(0.3, 0.3), PathParams(overlap_percent=0.0))
    tight = hatch_path(PlateSpec(0.3, 0.3), PathParams(overlap_percent=75.0))
    assert len({w.line_index for w in tight.waypoints}) > len({w.line_index for w in loose.waypoints})


def test_swept_area_matches_the_cells_own_estimate():
    """platform reports true_surface_area = path length * tool_contact_width."""
    path = hatch_path(PlateSpec(0.3, 0.3), PathParams())
    assert path.swept_area() == pytest.approx(path.path_length * path.params.tool_contact_width)


def test_conventions_are_recorded_with_their_uncertainty():
    path = hatch_path()
    assert path.conventions["row_alternation_verified_against_tpp"] is False
    assert "vz = -normal" in path.conventions["frame"]


def test_margin_can_starve_the_path():
    with pytest.raises(ValueError, match="no area"):
        hatch_path(PlateSpec(0.1, 0.1), PathParams(margin=0.2))


def test_s2_defaults_are_the_part_class_values():
    params = PathParams()
    assert params.tool_contact_width == 0.1778        # 7 inches exactly
    assert params.pathgap_sanding_line == 0.0075
    assert math.isclose(params.pitch, 0.0889)          # at the default 50% overlap
