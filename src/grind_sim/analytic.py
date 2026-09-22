"""Analytic Winkler (elastic-foundation) reference for a rigid annular pad on a flat sheet.

This is the yardstick the hydroelastic sim is measured against in the v1 fidelity gate, and the
cheap way to ask whether a case is worth simulating at all.

Model: the pad is a rigid annulus pressed into a Winkler foundation of stiffness ``k`` [N/m^3],
so the pressure at a point is ``p = k * delta`` where ``delta`` is the local penetration and
only positive penetrations carry load. This is the same constitutive law Newton's hydroelastic
contact reduces to in the rigid-workpiece limit (``sdf_hydroelastic.py`` computes face pressure
as ``kh * |depth|``), which is exactly why it is a fair reference.

Series stiffness: pad and sheet act in series, ``k_eff = k_pad*k_work/(k_pad+k_work)``. Every
metal is thousands of times stiffer than any pad, so ``k_eff ~= k_pad`` — see
``k_effective`` and the note in the grind prompt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PadSpec",
    "WinklerResult",
    "foundation_stiffness",
    "k_effective",
    "press_flat",
    "press_tilted",
    "sdf_voxel_budget",
]


@dataclass(frozen=True)
class PadSpec:
    """Geometry of the annular pad face that touches the sheet.

    Defaults are the 3M 80514 7" face plate: 7 in OD, 7/8 in arbor hole, 1/2 in thick. The
    arbor hole never touches the sheet, so it is excluded from the contact area.
    """

    outer_radius: float = 0.0889  # 7 in OD
    inner_radius: float = 0.0111  # 7/8 in arbor hole
    thickness: float = 0.0127  # 1/2 in

    @property
    def face_area(self) -> float:
        """Annular face area [m^2]."""
        return math.pi * (self.outer_radius**2 - self.inner_radius**2)


@dataclass(frozen=True)
class WinklerResult:
    """Outcome of pressing a rigid pad into a Winkler foundation at a target force."""

    force: float
    """Normal force carried [N]."""
    penetration_max: float
    """Deepest penetration anywhere in the patch [m]."""
    axis_penetration: float
    """Penetration on the pad axis [m]; negative when a tilted pad lifts its centre clear.

    This is what a simulation prescribes: drop the pad's origin by this much (then tilt it) and
    the contact should reproduce the rest of this result.
    """
    pressure_peak: float
    """Peak contact pressure [Pa]."""
    pressure_mean: float
    """Force divided by the contact area actually carrying load [Pa]."""
    contact_area: float
    """Area with positive penetration [m^2]."""
    contact_fraction: float
    """Contact area as a fraction of the pad's full face area."""


def foundation_stiffness(modulus: float, thickness: float) -> float:
    """Winkler stiffness ``k = E / h`` [N/m^3] of a layer of modulus ``E`` and thickness ``h``.

    Args:
        modulus: Layer Young's modulus [Pa].
        thickness: Layer thickness [m].

    Returns:
        Foundation stiffness [N/m^3].
    """
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    return modulus / thickness


def k_effective(k_pad: float, k_work: float) -> float:
    """Series stiffness of pad and workpiece, ``k_pad*k_work/(k_pad+k_work)`` [N/m^3]."""
    total = k_pad + k_work
    if total <= 0.0:
        return 0.0
    return k_pad * k_work / total


def press_flat(pad: PadSpec, k: float, force: float) -> WinklerResult:
    """Press the pad flat (zero tilt) onto a rigid sheet.

    The whole annular face carries load, so the pressure is uniform: ``p = F/A``, and the
    penetration follows from the foundation law, ``delta = p/k``.

    Args:
        pad: Pad face geometry.
        k: Effective foundation stiffness [N/m^3].
        force: Target normal force [N].

    Returns:
        The resulting patch, which for zero tilt covers the full face.
    """
    area = pad.face_area
    pressure = force / area
    delta = pressure / k
    return WinklerResult(
        force=force,
        penetration_max=delta,
        axis_penetration=delta,
        pressure_peak=pressure,
        pressure_mean=pressure,
        contact_area=area,
        contact_fraction=1.0,
    )


def press_tilted(
    pad: PadSpec,
    k: float,
    force: float,
    tilt_deg: float,
    *,
    n_radial: int = 400,
    n_angular: int = 720,
) -> WinklerResult:
    """Press the pad onto a rigid sheet at a tilt, giving a crescent patch near the rim.

    Tilting lifts one side clear, so the load concentrates into a crescent: less area, higher
    pressure, deeper penetration. That matters for the gate because penetration is what the SDF
    has to resolve — see :func:`sdf_voxel_budget`.

    The pad is rigid, so the penetration field is a tilted plane,
    ``delta(x, y) = delta_0 - x*tan(tilt)``, clipped at zero. ``delta_0`` is found by bisection
    so the integrated pressure matches ``force``. Integration is a polar quadrature over the
    annulus; the defaults resolve the crescent to well under a percent in area.

    Args:
        pad: Pad face geometry.
        k: Effective foundation stiffness [N/m^3].
        force: Target normal force [N].
        tilt_deg: Tilt of the pad axis away from the sheet normal [deg].
        n_radial: Radial quadrature samples.
        n_angular: Angular quadrature samples.

    Returns:
        The resulting crescent patch.
    """
    if tilt_deg <= 0.0:
        return press_flat(pad, k, force)

    r_edges = np.linspace(pad.inner_radius, pad.outer_radius, n_radial + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    theta = (np.arange(n_angular) + 0.5) * (2.0 * math.pi / n_angular)
    d_theta = 2.0 * math.pi / n_angular

    rr, tt = np.meshgrid(r, theta, indexing="ij")
    cell_area = (rr * dr[:, None]) * d_theta
    x = rr * np.cos(tt)
    slope = math.tan(math.radians(tilt_deg))

    def carried(delta_0: float) -> tuple[float, np.ndarray]:
        depth = np.clip(delta_0 - x * slope, 0.0, None)
        return float(k * np.sum(depth * cell_area)), depth

    # delta_0 is the penetration on the pad axis and is free to go negative: a stiff pad at any
    # real tilt lifts its axis clear of the sheet and carries the load on a rim sliver alone.
    # Bracket from "only the rim edge touches" (no load) up to the flat-contact depth.
    lo = -pad.outer_radius * slope
    hi = force / (k * pad.face_area)
    while carried(hi)[0] < force:
        hi *= 2.0
        if hi > 1.0:  # 1 m of penetration: the case is nonsense, fail loudly
            raise RuntimeError("no bracketing penetration found; check k and force")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if carried(mid)[0] < force:
            lo = mid
        else:
            hi = mid
    delta_0 = 0.5 * (lo + hi)
    carried_force, depth = carried(delta_0)

    live = depth > 0.0
    contact_area = float(np.sum(cell_area[live]))
    pressure = k * depth
    return WinklerResult(
        force=carried_force,
        penetration_max=float(depth.max()),
        axis_penetration=delta_0,
        pressure_peak=float(pressure.max()),
        pressure_mean=carried_force / contact_area if contact_area > 0 else 0.0,
        contact_area=contact_area,
        contact_fraction=contact_area / pad.face_area,
    )


def sdf_memory_mib(voxel: float) -> float:
    """Measured GPU cost of Newton's SDF for the 7" pad at a given voxel size [MiB].

    :func:`sdf_voxel_budget` counts the narrow band and lands ~1000x low, because Newton
    allocates in 8^3 tiles across the whole shell and carries per-block textures. This is the
    measured curve instead, from ``scripts/grind/voxel_ladder.py`` on an RTX 5080 (2 mm narrow
    band, float32 texture): 1 mm -> 1109 MiB, 500 um -> 7357 MiB, i.e. the expected 1/voxel^3
    with a constant fitted at 1 mm.

    Args:
        voxel: Voxel edge length [m].

    Returns:
        Estimated GPU memory [MiB]. Trust it to a factor of ~1.2, not better.
    """
    if voxel <= 0.0:
        raise ValueError("voxel must be positive")
    return 1109.0 * (1.0e-3 / voxel) ** 3


def sdf_voxel_budget(
    pad: PadSpec,
    voxel: float,
    *,
    narrow_band: float = 0.002,
    bytes_per_voxel: int = 2,
) -> tuple[int, float]:
    """Estimate the narrow-band SDF cost of a pad at a given voxel size.

    Newton stores a sparse SDF in a band around the surface, so cost scales with surface area
    times band width, not with the bounding box. This is the estimate that decides whether a
    voxel size is affordable at all.

    Args:
        pad: Pad face geometry.
        voxel: Voxel edge length [m].
        narrow_band: Half-width of the band each side of the surface [m].
        bytes_per_voxel: 2 for ``uint16`` (Newton's default texture format), 4 for ``float32``.

    Returns:
        Tuple of (voxel count, gigabytes).
    """
    if voxel <= 0.0:
        raise ValueError("voxel must be positive")
    faces = 2.0 * pad.face_area
    rims = 2.0 * math.pi * (pad.outer_radius + pad.inner_radius) * pad.thickness
    band_volume = (faces + rims) * (2.0 * narrow_band)
    count = int(band_volume / voxel**3)
    return count, count * bytes_per_voxel / 1024**3
