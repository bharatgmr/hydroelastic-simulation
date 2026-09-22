"""Watertight pad and sheet meshes for hydroelastic contact.

Hydroelastic contact needs volumetric, watertight, closed meshes on both sides — Newton silently
drops planes and heightfields, and an open mesh produces a garbage SDF. Everything here is built
with trimesh alone: shapely and manifold3d are not installed in this venv, so no CSG is used. The
pad is a surface of revolution (an annulus needs no boolean), the sheet is a box.

v1 is the flat pad only; ribs are deferred (see the plan). The rib parameters are therefore not
modelled here at all rather than being faked with defaults nobody measured.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from grind_sim.analytic import PadSpec

__all__ = ["make_pad", "make_sheet", "assert_watertight", "export_usd"]


def make_pad(pad: PadSpec, *, sections: int = 256) -> trimesh.Trimesh:
    """Build the flat annular face plate as a watertight surface of revolution.

    Args:
        pad: Pad geometry (outer/inner radius, thickness).
        sections: Angular segments in the revolution. 256 keeps the rim faceting under
            ~0.007 mm of the true circle at 7", well below any voxel size we can afford.

    Returns:
        A watertight mesh whose base sits at z=0 and top at z=thickness.

    Raises:
        ValueError: If the resulting mesh is not watertight.
    """
    profile = np.array(
        [
            [pad.inner_radius, 0.0],
            [pad.outer_radius, 0.0],
            [pad.outer_radius, pad.thickness],
            [pad.inner_radius, pad.thickness],
            [pad.inner_radius, 0.0],  # close the profile, else revolve leaves the ends open
        ]
    )
    mesh = trimesh.creation.revolve(profile, sections=sections)
    assert_watertight(mesh, "pad")

    expected = math.pi * (pad.outer_radius**2 - pad.inner_radius**2) * pad.thickness
    if abs(mesh.volume - expected) / expected > 0.01:
        raise ValueError(f"pad volume {mesh.volume:.3e} m^3 differs from analytic {expected:.3e}")
    return mesh


def make_sheet(
    side: float = 0.300,
    thickness: float = 0.003,
    *,
    narrow_band_inner: float | None = None,
) -> trimesh.Trimesh:
    """Build the flat metal sheet as a box, with its top face at z=0.

    Args:
        side: Square side length [m].
        thickness: Sheet thickness [m].
        narrow_band_inner: If given, the inner (negative) SDF narrow-band distance [m] that will
            be authored on this shape. It must be shallower than half the sheet thickness, or the
            inner bands from the two faces meet and the SDF interior is nonsense.

    Returns:
        A watertight box mesh spanning z in [-thickness, 0].

    Raises:
        ValueError: If the narrow band would exceed half the thickness.
    """
    if narrow_band_inner is not None and abs(narrow_band_inner) >= thickness / 2.0:
        raise ValueError(
            f"narrow band inner {abs(narrow_band_inner)*1e3:.2f} mm must be under half the sheet "
            f"thickness ({thickness*1e3/2:.2f} mm)"
        )
    mesh = trimesh.creation.box(extents=(side, side, thickness))
    mesh.apply_translation((0.0, 0.0, -thickness / 2.0))
    assert_watertight(mesh, "sheet")
    return mesh


def assert_watertight(mesh: trimesh.Trimesh, name: str) -> None:
    """Raise unless the mesh is closed and consistently wound.

    Newton accepts an open mesh and bakes a meaningless SDF from it, so this check is the only
    thing standing between a typo and a plausible-looking but wrong contact patch.
    """
    if not mesh.is_watertight:
        raise ValueError(f"{name} mesh is not watertight")
    if not mesh.is_winding_consistent:
        raise ValueError(f"{name} mesh has inconsistent winding")
    if mesh.volume <= 0.0:
        raise ValueError(f"{name} mesh has non-positive volume {mesh.volume:.3e}")


def export_usd(mesh: trimesh.Trimesh, stage, prim_path: str):
    """Author a trimesh as a ``UsdGeom.Mesh`` prim on an open stage.

    Kept separate from mesh construction so the geometry module stays importable (and testable)
    without Isaac Sim running.

    Args:
        mesh: Watertight source mesh.
        stage: An open ``Usd.Stage``.
        prim_path: Path for the new mesh prim.

    Returns:
        The created ``UsdGeom.Mesh``.
    """
    from pxr import UsdGeom, Vt

    geom = UsdGeom.Mesh.Define(stage, prim_path)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.astype(np.int32).ravel()))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = mesh.bounds
    geom.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([lo, hi], dtype=np.float32)))
    return geom
