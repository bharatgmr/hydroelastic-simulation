"""The analytic Winkler reference is the gate's yardstick, so it gets checked against
closed-form answers and against physical monotonicity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grind_sim.analytic import (
    PadSpec,
    foundation_stiffness,
    k_effective,
    press_flat,
    press_tilted,
    sdf_voxel_budget,
)

PAD = PadSpec()


def test_face_area_matches_annulus_formula():
    assert PAD.face_area == pytest.approx(math.pi * (0.0889**2 - 0.0111**2))


def test_flat_press_is_uniform_and_closed_form():
    k = foundation_stiffness(20.0e6, PAD.thickness)
    res = press_flat(PAD, k, 25.0)
    assert res.pressure_mean == pytest.approx(25.0 / PAD.face_area)
    assert res.pressure_peak == pytest.approx(res.pressure_mean)
    assert res.penetration_max == pytest.approx(res.pressure_mean / k)
    assert res.contact_fraction == pytest.approx(1.0)


def test_series_stiffness_is_dominated_by_the_softer_layer():
    # A 20 MPa pad against 200 GPa steel: the metal contributes under 0.01%.
    k_pad = foundation_stiffness(20.0e6, 0.0127)
    k_steel = foundation_stiffness(200.0e9, 0.003)
    assert k_effective(k_pad, k_steel) == pytest.approx(k_pad, rel=1e-4)


def test_tilted_press_carries_the_target_force():
    k = foundation_stiffness(20.0e6, PAD.thickness)
    res = press_tilted(PAD, k, 25.0, 5.0)
    assert res.force == pytest.approx(25.0, rel=2e-3)


def test_tilt_concentrates_load_into_a_rim_crescent():
    k = foundation_stiffness(20.0e6, PAD.thickness)
    flat = press_flat(PAD, k, 25.0)
    tilted = press_tilted(PAD, k, 25.0, 5.0)
    # Less area, higher peak pressure, deeper penetration — the whole reason tilt matters here.
    assert tilted.contact_fraction < 0.05
    assert tilted.pressure_peak > 100.0 * flat.pressure_peak
    assert tilted.penetration_max > flat.penetration_max


def test_more_tilt_means_smaller_patch():
    k = foundation_stiffness(20.0e6, PAD.thickness)
    fractions = [press_tilted(PAD, k, 25.0, t).contact_fraction for t in (1.0, 2.0, 5.0)]
    assert fractions == sorted(fractions, reverse=True)


def test_zero_tilt_falls_back_to_the_flat_solution():
    k = foundation_stiffness(20.0e6, PAD.thickness)
    assert press_tilted(PAD, k, 25.0, 0.0).contact_area == pytest.approx(PAD.face_area)


def test_softer_pad_penetrates_further_at_the_same_force():
    soft = press_flat(PAD, foundation_stiffness(100.0e3, PAD.thickness), 25.0)
    hard = press_flat(PAD, foundation_stiffness(500.0e6, PAD.thickness), 25.0)
    assert soft.penetration_max > 1000.0 * hard.penetration_max


def test_voxel_budget_scales_as_inverse_cube():
    coarse = sdf_voxel_budget(PAD, 1.0e-3)[0]
    fine = sdf_voxel_budget(PAD, 0.5e-3)[0]
    assert fine / coarse == pytest.approx(8.0, rel=0.01)


def test_quadrature_is_converged_at_the_defaults():
    k = foundation_stiffness(20.0e6, PAD.thickness)
    coarse = press_tilted(PAD, k, 25.0, 5.0, n_radial=200, n_angular=360)
    fine = press_tilted(PAD, k, 25.0, 5.0, n_radial=800, n_angular=1440)
    assert coarse.contact_area == pytest.approx(fine.contact_area, rel=0.05)
    assert coarse.pressure_peak == pytest.approx(fine.pressure_peak, rel=0.05)


def test_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        foundation_stiffness(20.0e6, 0.0)
    with pytest.raises(ValueError):
        sdf_voxel_budget(PAD, 0.0)


def test_hard_pad_flat_penetration_is_sub_micron():
    """The headline feasibility result: a hard plate pressed flat barely moves."""
    k = foundation_stiffness(500.0e6, PAD.thickness)
    assert press_flat(PAD, k, 25.0).penetration_max < 1.0e-6
    assert np.isfinite(press_flat(PAD, k, 25.0).penetration_max)
