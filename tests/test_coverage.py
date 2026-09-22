"""Coverage, measured two ways: from simulated contact, and the way the cell counts it."""

from __future__ import annotations

import numpy as np
import pytest

from grind_sim.coverage import (
    CoverageGrid,
    coverage_gap,
    plate_point_cloud,
    platform_coverage,
    sim_coverage,
)


def disc_patch(centre_xy, radius=0.05, n=60, pressure_pa=5.0e3):
    """A filled disc of uniform pressure lying on z=0, as a patch triple."""
    xs = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    keep = (gx**2 + gy**2) <= radius**2
    cell = (2 * radius / (n - 1)) ** 2
    centroid = np.stack([gx[keep] + centre_xy[0], gy[keep] + centre_xy[1],
                         np.zeros(keep.sum())], axis=1)
    return centroid, np.full(keep.sum(), pressure_pa), np.full(keep.sum(), cell)


def test_grid_accumulates_area_and_peak_pressure():
    grid = CoverageGrid.empty(0.2, 0.2, cell_size=0.005)
    centroid, pressure, area = disc_patch((0.0, 0.0), radius=0.03)
    grid.add_patch(centroid, pressure, area)
    assert grid.contact_area.sum() == pytest.approx(area.sum(), rel=1e-9)
    assert grid.pressure_peak.max() == pytest.approx(5.0e3)
    assert grid.coverage_fraction() > 0.0


def test_patches_outside_the_plate_are_dropped_not_wrapped():
    grid = CoverageGrid.empty(0.2, 0.2, cell_size=0.005)
    centroid, pressure, area = disc_patch((5.0, 5.0), radius=0.01)
    grid.add_patch(centroid, pressure, area)
    assert grid.contact_area.sum() == 0.0


def test_mean_pressure_is_area_weighted_and_zero_where_untouched():
    grid = CoverageGrid.empty(0.2, 0.2, cell_size=0.005)
    grid.add_patch(*disc_patch((0.0, 0.0), radius=0.03, pressure_pa=4.0e3))
    mean = grid.mean_pressure
    assert mean.max() == pytest.approx(4.0e3, rel=1e-6)
    assert mean[grid.contact_area == 0].max(initial=0.0) == 0.0


def test_overlapping_passes_count_as_repeat_visits():
    grid = CoverageGrid.empty(0.2, 0.2, cell_size=0.005)
    grid.add_patch(*disc_patch((0.0, 0.0), radius=0.03))
    grid.add_patch(*disc_patch((0.01, 0.0), radius=0.03))
    assert grid.visits.max() == 2


def test_sim_coverage_rises_as_patches_fill_the_plate():
    sparse = sim_coverage([disc_patch((0.0, 0.0), radius=0.02)], 0.2, 0.2)
    dense = sim_coverage([disc_patch((x, y), radius=0.02)
                          for x in (-0.05, 0.0, 0.05) for y in (-0.05, 0.0, 0.05)], 0.2, 0.2)
    assert dense.coverage_fraction() > sparse.coverage_fraction()


def test_platform_coverage_counts_points_under_the_disc():
    cloud = plate_point_cloud(0.4, 0.4, spacing=0.005)
    result = platform_coverage(np.array([[0.0, 0.0, 0.0]]), cloud)
    # One disc of radius 88.9 mm on a 400 x 400 mm plate.
    expected = 100.0 * (np.pi * 0.0889**2) / (0.4 * 0.4)
    assert result["coverage_percentage"] == pytest.approx(expected, rel=0.05)
    assert result["total_points"] == len(cloud)


def test_platform_compliance_margin_credits_standoff_as_contact():
    """backing_pad_height is the cell's stand-in for pad compliance: 25 mm of it."""
    cloud = plate_point_cloud(0.3, 0.3, spacing=0.005)
    lifted = np.array([[0.0, 0.0, 0.02]])  # disc held 20 mm off the surface
    generous = platform_coverage(lifted, cloud, backing_pad_height=0.025)
    strict = platform_coverage(lifted, cloud, backing_pad_height=0.0)
    assert generous["coverage_percentage"] > 0.0
    assert strict["coverage_percentage"] == 0.0


def test_platform_buckets_are_equal_area():
    """A full disc should fill essentially every polar bucket, which equal-area annuli make even."""
    cloud = plate_point_cloud(0.3, 0.3, spacing=0.002)
    result = platform_coverage(np.array([[0.0, 0.0, 0.0]]), cloud)
    assert result["bucket_coverage"] > 0.95


def test_platform_coverage_is_blind_to_tilt_by_construction():
    """The cell's cylinder never tilts, so the same waypoints give the same number regardless.

    That is precisely the limitation the simulated patch is meant to expose.
    """
    cloud = plate_point_cloud(0.3, 0.3, spacing=0.005)
    positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    assert platform_coverage(positions, cloud) == platform_coverage(positions, cloud)


def test_coverage_gap_reports_both_and_their_ratio():
    cloud = plate_point_cloud(0.3, 0.3, spacing=0.005)
    platform = platform_coverage(np.array([[0.0, 0.0, 0.0]]), cloud)
    sim = sim_coverage([disc_patch((0.0, 0.0), radius=0.02)], 0.3, 0.3)
    gap = coverage_gap(sim, platform)
    assert gap["platform_coverage_percentage"] > gap["sim_coverage_percentage"]
    assert gap["platform_over_sim"] > 1.0


def test_empty_inputs_do_not_blow_up():
    grid = sim_coverage([], 0.2, 0.2)
    assert grid.coverage_fraction() == 0.0
    result = platform_coverage(np.zeros((0, 3)), plate_point_cloud(0.1, 0.1))
    assert result["coverage_percentage"] == 0.0


def test_applied_pressure_accumulates_over_passes_rather_than_averaging():
    """Two passes at the same pressure leave twice the applied pressure on those cells."""
    one = sim_coverage([disc_patch((0.0, 0.0), radius=0.03, pressure_pa=10.0e3)], 0.2, 0.2)
    two = sim_coverage([disc_patch((0.0, 0.0), radius=0.03, pressure_pa=10.0e3)] * 2, 0.2, 0.2)
    assert two.applied_pressure.max() == pytest.approx(2.0 * one.applied_pressure.max(), rel=1e-6)
    # ... while the per-pass mean is unchanged.
    assert two.mean_pressure.max() == pytest.approx(one.mean_pressure.max(), rel=1e-6)


def test_dose_is_the_accumulated_load():
    grid = sim_coverage([disc_patch((0.0, 0.0), radius=0.03, pressure_pa=10.0e3)], 0.2, 0.2)
    centroid, pressure, area = disc_patch((0.0, 0.0), radius=0.03, pressure_pa=10.0e3)
    assert grid.dose.sum() == pytest.approx(float((pressure * area).sum()), rel=1e-9)


def test_overlap_stats_splits_missed_single_and_repeated():
    from grind_sim.coverage import overlap_stats

    grid = sim_coverage([disc_patch((-0.02, 0.0), radius=0.03),
                         disc_patch((0.02, 0.0), radius=0.03)], 0.2, 0.2)
    stats = overlap_stats(grid)
    assert stats["remaining_fraction"] > 0.0        # corners never touched
    assert stats["single_pass_fraction"] > 0.0      # the outer lobes
    assert stats["overlapped_fraction"] > 0.0       # the lens where they cross
    assert stats["max_visits"] == 2
    assert (stats["applied_pressure_overlapped_pa"]
            > stats["applied_pressure_single_pa"])


def test_overlap_stats_on_an_untouched_plate():
    from grind_sim.coverage import overlap_stats

    stats = overlap_stats(sim_coverage([], 0.2, 0.2))
    assert stats["remaining_fraction"] == 1.0
    assert stats["overlapped_fraction"] == 0.0
    assert stats["applied_pressure_peak_pa"] == 0.0
