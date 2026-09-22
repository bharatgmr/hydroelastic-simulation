"""Shared stage authoring for the grind-contact scripts.

Imports from ``pxr`` at call time, not at module import, so this module stays importable (and
unit-testable) without a running Isaac Sim.
"""

from __future__ import annotations

__all__ = ["PAD_PATH", "PAD_BODY", "SHEET_PATH", "SHEET_BODY", "author_shape"]

PAD_PATH = "/World/pad"
SHEET_PATH = "/World/sheet"
PAD_BODY = f"{PAD_PATH}/body"
SHEET_BODY = f"{SHEET_PATH}/body"


def author_shape(
    prim,
    kh: float,
    voxel: float,
    narrow_inner: float,
    narrow_outer: float,
    *,
    texture_format: str = "float32",
) -> None:
    """Apply Newton's SDF + hydroelastic attributes to a collider prim.

    Args:
        prim: A ``UsdGeom.Mesh`` prim to turn into a hydroelastic collider.
        kh: Hydroelastic stiffness for this shape [N/m^3]. Newton's contact law takes the
            stiffness from shape B of each pair, so both shapes need a physical value.
        voxel: SDF target voxel size [m]. When authored this overrides ``sdfMaxResolution``.
        narrow_inner: Inner (negative) narrow-band distance [m]; must stay shallower than half
            the shape's thickness.
        narrow_outer: Outer (positive) narrow-band distance [m].
        texture_format: ``uint8``, ``uint16`` (Newton's default) or ``float32``.
    """
    from pxr import Sdf, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    if not prim.HasAPI("NewtonSDFCollisionAPI"):
        prim.ApplyAPI("NewtonSDFCollisionAPI")
    prim.GetAttribute("newton:hydroelasticEnabled").Set(True)
    prim.GetAttribute("newton:hydroelasticStiffness").Set(float(kh))
    prim.GetAttribute("newton:sdfTargetVoxelSize").Set(float(voxel))
    prim.GetAttribute("newton:sdfNarrowBandInner").Set(float(narrow_inner))
    prim.GetAttribute("newton:sdfNarrowBandOuter").Set(float(narrow_outer))
    prim.GetAttribute("newton:contactMargin").Set(0.0)
    prim.GetAttribute("newton:sdfTextureFormat").Set(texture_format)
    # "none" keeps the mesh exact: no convex decomposition behind our backs.
    approx = prim.GetAttribute("physics:approximation")
    if not approx.IsValid():
        approx = prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
    approx.Set("none")
