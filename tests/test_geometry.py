"""Mesh checks. Newton bakes an SDF from whatever it is given, so a non-watertight mesh fails
silently and plausibly — these run on CPU with no Isaac Sim.
"""

from __future__ import annotations

import math

import pytest

from grind_sim.analytic import PadSpec
from grind_sim.geometry import assert_watertight, make_pad, make_sheet

PAD = PadSpec()


def test_pad_is_watertight_with_analytic_volume():
    mesh = make_pad(PAD)
    assert mesh.is_watertight
    expected = math.pi * (PAD.outer_radius**2 - PAD.inner_radius**2) * PAD.thickness
    assert mesh.volume == pytest.approx(expected, rel=0.01)


def test_pad_spans_the_expected_bounds():
    mesh = make_pad(PAD)
    lo, hi = mesh.bounds
    assert lo[2] == pytest.approx(0.0, abs=1e-9)
    assert hi[2] == pytest.approx(PAD.thickness, abs=1e-9)
    assert hi[0] == pytest.approx(PAD.outer_radius, rel=1e-3)


def test_pad_keeps_the_arbor_hole_open():
    """The bore carries no load but must still exist, for mass and inertia."""
    mesh = make_pad(PAD)
    solid = math.pi * PAD.outer_radius**2 * PAD.thickness
    assert mesh.volume < solid * 0.99


def test_sheet_is_watertight_with_top_face_at_origin():
    mesh = make_sheet(0.3, 0.003)
    assert mesh.is_watertight
    lo, hi = mesh.bounds
    assert hi[2] == pytest.approx(0.0, abs=1e-9)
    assert lo[2] == pytest.approx(-0.003, abs=1e-9)
    assert mesh.volume == pytest.approx(0.3 * 0.3 * 0.003, rel=1e-6)


def test_sheet_rejects_a_narrow_band_deeper_than_half_its_thickness():
    with pytest.raises(ValueError, match="narrow band"):
        make_sheet(0.3, 0.003, narrow_band_inner=-0.002)
    make_sheet(0.3, 0.003, narrow_band_inner=-0.001)  # 1 mm into a 3 mm sheet is fine


def test_assert_watertight_catches_an_open_mesh():
    mesh = make_pad(PAD)
    mesh.faces = mesh.faces[:-4]  # punch a hole
    with pytest.raises(ValueError, match="watertight"):
        assert_watertight(mesh, "pad")


def test_coarser_revolution_is_still_watertight():
    assert make_pad(PAD, sections=32).is_watertight
