"""Two ways to say how much of a plate the tool touched, and the gap between them.

**Sim coverage** rasterises the simulated contact patches: a cell counts as covered when real
contact pressure landed on it. It knows about tilt, because the patch does.

**Platform coverage** reimplements ``TppUtils::generate_contact_analysis`` /
``compute_contact_percentage``: at each waypoint, drop a cylinder of radius
``tool_contact_width / 2`` along the surface normal, bin what falls inside into equal-area polar
sectors, and count a point as touched if it lies within ``backing_pad_height`` of the contact plane.
That margin (25 mm by default, against a pad the config comments say is really 10 mm) is a flat
geometric allowance standing in for pad compliance, and the metric is blind to tilt — the cylinder
never changes shape.

Reporting both is the point: the difference is how much the cell's coverage number is flattered by
that allowance, which is exactly what a contact simulation can say and the planner cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["CoverageGrid", "overlap_stats", "platform_coverage", "sim_coverage"]


@dataclass
class CoverageGrid:
    """A raster over the plate accumulating what the tool did to each cell."""

    cell_size: float
    origin: np.ndarray
    """World position of the lower-left cell corner [m]."""
    contact_area: np.ndarray
    """Contact area accumulated per cell [m^2]."""
    pressure_sum: np.ndarray
    """Sum of pressure x area per cell [N], for an area-weighted mean."""
    pressure_peak: np.ndarray
    """Highest pressure seen in each cell [Pa]."""
    visits: np.ndarray
    """How many waypoints put contact in each cell."""

    @classmethod
    def empty(cls, size_x: float, size_y: float, cell_size: float = 0.0025) -> "CoverageGrid":
        """Grid covering a plate centred on the origin. 2.5 mm cells match sanding-wm's maps."""
        nx = max(1, int(round(size_x / cell_size)))
        ny = max(1, int(round(size_y / cell_size)))
        shape = (ny, nx)
        return cls(
            cell_size=cell_size,
            origin=np.array([-size_x / 2.0, -size_y / 2.0]),
            contact_area=np.zeros(shape),
            pressure_sum=np.zeros(shape),
            pressure_peak=np.zeros(shape),
            visits=np.zeros(shape, dtype=np.int32),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.contact_area.shape

    def add_patch(self, centroid: np.ndarray, pressure: np.ndarray, area: np.ndarray) -> None:
        """Accumulate one contact patch onto the grid.

        Args:
            centroid: Face centroids [N, 3] in world coordinates.
            pressure: Face pressures [N] (Pa).
            area: Face areas [N] (m^2).
        """
        centroid = np.asarray(centroid, dtype=np.float64).reshape(-1, 3)
        if centroid.size == 0:
            return
        pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
        area = np.asarray(area, dtype=np.float64).reshape(-1)

        ny, nx = self.shape
        ix = np.floor((centroid[:, 0] - self.origin[0]) / self.cell_size).astype(int)
        iy = np.floor((centroid[:, 1] - self.origin[1]) / self.cell_size).astype(int)
        keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if not np.any(keep):
            return
        ix, iy = ix[keep], iy[keep]
        flat = iy * nx + ix

        np.add.at(self.contact_area.ravel(), flat, area[keep])
        np.add.at(self.pressure_sum.ravel(), flat, pressure[keep] * area[keep])
        np.maximum.at(self.pressure_peak.ravel(), flat, pressure[keep])
        touched = np.zeros(self.visits.size, dtype=np.int32)
        touched[np.unique(flat)] = 1
        self.visits += touched.reshape(self.shape)

    @property
    def applied_pressure(self) -> np.ndarray:
        """Pressure summed over every pass that touched a cell [Pa].

        Not an average: two passes at 10 kPa read 20 kPa, because what a cell has had applied to
        it is what drives removal. Divide by :attr:`visits` for the per-pass figure, or use
        :attr:`mean_pressure` for the area-weighted one.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            per_pass = np.where(self.contact_area > 0, self.pressure_sum / self.contact_area, 0.0)
        return per_pass * np.maximum(self.visits, 1) * (self.contact_area > 0)

    @property
    def dose(self) -> np.ndarray:
        """Accumulated normal load per cell [N] — the sum of pressure x area over all passes.

        This is the quantity a Preston removal model integrates (``k.P.|v|.dt``) once a traverse
        speed is attached, so it is the honest thing to carry forward rather than a pressure that
        has been averaged over passes.
        """
        return self.pressure_sum

    @property
    def mean_pressure(self) -> np.ndarray:
        """Area-weighted mean pressure per cell [Pa]; zero where nothing touched."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.contact_area > 0, self.pressure_sum / self.contact_area, 0.0)

    def coverage_fraction(self, min_area: float = 0.0) -> float:
        """Fraction of cells that received more than ``min_area`` of contact."""
        return float((self.contact_area > min_area).mean())


def sim_coverage(
    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    size_x: float,
    size_y: float,
    cell_size: float = 0.0025,
) -> CoverageGrid:
    """Rasterise simulated contact patches into a coverage grid.

    Args:
        patches: One ``(centroid, pressure, area)`` triple per waypoint.
        size_x: Plate extent in x [m].
        size_y: Plate extent in y [m].
        cell_size: Raster cell size [m].

    Returns:
        The accumulated grid.
    """
    grid = CoverageGrid.empty(size_x, size_y, cell_size)
    for centroid, pressure, area in patches:
        grid.add_patch(centroid, pressure, area)
    return grid


def platform_coverage(
    positions: np.ndarray,
    points: np.ndarray,
    *,
    tool_contact_width: float = 0.1778,
    backing_pad_height: float = 0.025,
    opposite_height: float = 0.1,
    radius_step_percent: float = 10.0,
    angle_step_deg: float = 10.0,
) -> dict:
    """Reimplement the cell's coverage metric so the sim can report the same number.

    Mirrors ``generate_contact_analysis``: a cylinder of radius ``tool_contact_width/2`` at each
    waypoint, aligned with the surface normal (+z here, since the plate is flat), extending
    ``backing_pad_height`` above the contact plane and ``opposite_height`` below it. Points inside
    are binned into **equal-area** annuli — radius bucket ``k`` spans
    ``R*sqrt(k/n) .. R*sqrt((k+1)/n)`` so every bucket has the same area, which is what the
    planner's ``step_rad_area[k] ∝ (k+1)^2 - k^2`` weighting amounts to — and into
    ``360/angle_step_deg`` angular sectors.

    Args:
        positions: Waypoint positions [W, 3].
        points: The part's point cloud [P, 3]; for a flat plate, a grid of surface samples.
        tool_contact_width: Disc diameter [m].
        backing_pad_height: Compliance allowance above the contact plane [m].
        opposite_height: Cylinder extent below the plane [m].
        radius_step_percent: Radial bucket width as a percentage, so 10 means 10 buckets.
        angle_step_deg: Angular bucket width [deg].

    Returns:
        ``{"coverage_percentage", "touched_points", "total_points", "bucket_coverage"}`` —
        ``bucket_coverage`` is the fraction of (radius, angle) buckets that any waypoint filled,
        which is the planner's per-waypoint number averaged over the path.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if positions.size == 0 or points.size == 0:
        return {"coverage_percentage": 0.0, "touched_points": 0, "total_points": len(points),
                "bucket_coverage": 0.0}

    radius = tool_contact_width / 2.0
    n_radial = max(1, int(round(100.0 / radius_step_percent)))
    n_angular = max(1, int(round(360.0 / angle_step_deg)))
    # Equal-area annuli: the k-th boundary is at R*sqrt(k/n).
    edges = radius * np.sqrt(np.arange(n_radial + 1) / n_radial)

    touched = np.zeros(len(points), dtype=bool)
    filled_fractions = []
    for position in positions:
        delta = points - position
        # Positive = the point sits below the waypoint, i.e. recessed away from the tool. That is
        # the direction ``backing_pad_height`` forgives: a compliant pad is credited with reaching
        # down into a hollow. ``opposite_height`` extends the cylinder the other way.
        axial = position[2] - points[:, 2]
        inside_axially = (axial <= backing_pad_height) & (axial >= -opposite_height)
        radial = np.hypot(delta[:, 0], delta[:, 1])
        inside = inside_axially & (radial <= radius)
        touched |= inside
        if not np.any(inside):
            filled_fractions.append(0.0)
            continue
        r_bucket = np.clip(np.searchsorted(edges, radial[inside], side="right") - 1, 0, n_radial - 1)
        a_bucket = (np.degrees(np.arctan2(delta[inside, 1], delta[inside, 0])) % 360.0)
        a_bucket = np.clip((a_bucket / angle_step_deg).astype(int), 0, n_angular - 1)
        filled = len(np.unique(r_bucket * n_angular + a_bucket))
        filled_fractions.append(filled / float(n_radial * n_angular))

    return {
        "coverage_percentage": 100.0 * float(touched.mean()),
        "touched_points": int(touched.sum()),
        "total_points": int(len(points)),
        "bucket_coverage": float(np.mean(filled_fractions)),
    }


def plate_point_cloud(size_x: float, size_y: float, spacing: float = 0.0025) -> np.ndarray:
    """A flat plate sampled as a point cloud, standing in for a scan."""
    xs = np.arange(-size_x / 2.0, size_x / 2.0 + 1e-12, spacing)
    ys = np.arange(-size_y / 2.0, size_y / 2.0 + 1e-12, spacing)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)


def coverage_gap(sim: CoverageGrid, platform: dict) -> dict:
    """Compare the two measures on the same path.

    Returns the two percentages and their ratio. Platform's number should be the higher one
    wherever tilt lifts part of the disc clear, because its cylinder does not tilt and its
    compliance allowance credits standoff as contact.
    """
    sim_percentage = 100.0 * sim.coverage_fraction()
    platform_percentage = float(platform["coverage_percentage"])
    ratio = platform_percentage / sim_percentage if sim_percentage > 0 else math.inf
    return {
        "sim_coverage_percentage": sim_percentage,
        "platform_coverage_percentage": platform_percentage,
        "platform_over_sim": ratio,
    }


def overlap_stats(grid: CoverageGrid) -> dict:
    """Split a plate into what was missed, what was hit once, and what was hit repeatedly.

    "Remaining" is the fraction of the plate no contact ever reached — on a tilted pass this is
    the band between hatch lines that the planner's diameter-based pitch assumes is covered.
    "Overlapped" is where passes stacked, which is where the applied pressure accumulates.

    Args:
        grid: An accumulated coverage grid.

    Returns:
        Fractions and pressures for each class, plus the peak accumulated load.
    """
    touched = grid.contact_area > 0
    once = touched & (grid.visits == 1)
    over = touched & (grid.visits >= 2)
    applied = grid.applied_pressure
    total_cells = float(grid.visits.size)

    def mean_of(mask: np.ndarray) -> float:
        return float(applied[mask].mean()) if mask.any() else 0.0

    return {
        "remaining_fraction": float((~touched).sum() / total_cells),
        "single_pass_fraction": float(once.sum() / total_cells),
        "overlapped_fraction": float(over.sum() / total_cells),
        "max_visits": int(grid.visits.max()),
        "applied_pressure_mean_pa": mean_of(touched),
        "applied_pressure_single_pa": mean_of(once),
        "applied_pressure_overlapped_pa": mean_of(over),
        "applied_pressure_peak_pa": float(applied.max()),
        "dose_total_n": float(grid.dose.sum()),
    }
