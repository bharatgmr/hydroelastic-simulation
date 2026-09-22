"""Bake a press run into a USD animation: pad, sheet, and the contact patch as coloured points.

Same idea as the nut-bolt recordings — the physics has already happened, so what gets written is
pure animation that replays instantly and identically. One frame per force step.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh

__all__ = ["PressFrame", "write_press_usd"]


class PressFrame:
    """One press step: where the pad sat, and the patch it produced."""

    __slots__ = ("translate", "orient_xyzw", "centroid", "pressure", "label")

    def __init__(
        self,
        translate: Sequence[float],
        orient_xyzw: Sequence[float],
        centroid: np.ndarray,
        pressure: np.ndarray,
        label: str = "",
    ) -> None:
        self.translate = tuple(float(v) for v in translate)
        self.orient_xyzw = tuple(float(v) for v in orient_xyzw)
        self.centroid = np.asarray(centroid, dtype=np.float64).reshape(-1, 3)
        self.pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
        self.label = label


def _colours(pressure: np.ndarray) -> np.ndarray:
    """Map pressure to RGB with matplotlib's inferno, normalised per frame."""
    import matplotlib.cm as cm

    if pressure.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    top = float(pressure.max())
    scaled = pressure / top if top > 0 else np.zeros_like(pressure)
    return cm.inferno(scaled)[:, :3].astype(np.float32)


def write_press_usd(
    path: str,
    pad_mesh: trimesh.Trimesh,
    sheet_mesh: trimesh.Trimesh,
    frames: Sequence[PressFrame],
    *,
    fps: float = 1.0,
    max_points: int = 40_000,
    point_width: float = 0.0015,
    marker_points: np.ndarray | None = None,
    pad_opacity: float = 0.4,
) -> None:
    """Write pad, sheet and per-frame contact points to a USD file.

    Args:
        path: Output ``.usda``/``.usd``.
        pad_mesh: Pad mesh in its own frame (base at z=0).
        sheet_mesh: Sheet mesh, already positioned (top face at z=0).
        frames: One entry per press step, in order.
        fps: Time codes per second; 1.0 means one press step per second on replay.
        max_points: Patches are subsampled to this many points for display. A flat press can
            produce 200k faces, which is slow to load and unreadable on screen.
        point_width: Display width of each contact point [m].
        pad_opacity: Display opacity of the disc; below 1 the contact patch shows through it.
        marker_points: Optional static points drawn in green — the scan's marked region, so it is
            visible in the replay whether contact is landing on it.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(len(frames) - 1, 0))

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # The recordings used to carry no lights at all, which is fine in the GUI (the viewport's
    # camera-light mode covers it) but renders black offscreen. Author them so a headless capture
    # and an interactive replay look the same.
    dome = UsdLux.DomeLight.Define(stage, "/World/lights/dome")
    dome.CreateIntensityAttr(400.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.89, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/lights/key")
    key.CreateIntensityAttr(1500.0)
    key.CreateAngleAttr(1.5)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
    fill = UsdLux.DistantLight.Define(stage, "/World/lights/fill")
    fill.CreateIntensityAttr(600.0)
    fill.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
    UsdGeom.Xformable(fill).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 0.0, -130.0))

    def author_mesh(prim_path: str, mesh: trimesh.Trimesh, colour: tuple[float, float, float]):
        geom = UsdGeom.Mesh.Define(stage, prim_path)
        geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
        geom.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(mesh.faces.astype(np.int32).ravel())
        )
        geom.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32))
        )
        geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        geom.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
        lo, hi = mesh.bounds
        geom.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([lo, hi], dtype=np.float32)))
        return geom

    author_mesh("/World/sheet", sheet_mesh, (0.34, 0.36, 0.40))

    pad_xform = UsdGeom.Xform.Define(stage, "/World/pad")
    pad_geom = author_mesh("/World/pad/mesh", pad_mesh, (0.75, 0.20, 0.18))
    # Semi-transparent, or the disc hides the contact patch underneath it — which is the one
    # thing a reviewer is looking for. displayOpacity alone is not enough: the RTX renderer
    # ignores it and draws the disc solid, so bind a UsdPreviewSurface that carries the opacity.
    pad_geom.CreateDisplayOpacityAttr(Vt.FloatArray([pad_opacity]))
    if pad_opacity < 1.0:
        material = UsdShade.Material.Define(stage, "/World/pad/material")
        shader = UsdShade.Shader.Define(stage, "/World/pad/material/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.78, 0.24, 0.20))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(pad_opacity)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(pad_geom.GetPrim()).Bind(material)
    translate_op = pad_xform.AddTranslateOp()
    orient_op = pad_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    if marker_points is not None and len(marker_points):
        marked = np.asarray(marker_points, dtype=np.float32)
        if len(marked) > max_points:
            marked = marked[np.random.default_rng(1).choice(len(marked), max_points, replace=False)]
        marker = UsdGeom.Points.Define(stage, "/World/marked")
        marker.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(marked))
        marker.CreateWidthsAttr(Vt.FloatArray.FromNumpy(
            np.full(len(marked), point_width, np.float32)))
        marker.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        marker.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.15, 0.70, 0.30)]))

    points = UsdGeom.Points.Define(stage, "/World/contact")
    points_attr = points.CreatePointsAttr()
    colour_attr = points.CreateDisplayColorAttr()
    width_attr = points.CreateWidthsAttr()
    points.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    colour_primvar = UsdGeom.PrimvarsAPI(points).GetPrimvar("displayColor")
    if colour_primvar:
        colour_primvar.SetInterpolation(UsdGeom.Tokens.vertex)

    rng = np.random.default_rng(0)
    for frame_index, frame in enumerate(frames):
        translate_op.Set(Gf.Vec3d(*frame.translate), time=frame_index)
        x, y, z, w = frame.orient_xyzw
        orient_op.Set(Gf.Quatd(w, x, y, z), time=frame_index)

        centroid, pressure = frame.centroid, frame.pressure
        if len(centroid) > max_points:
            keep = rng.choice(len(centroid), max_points, replace=False)
            centroid, pressure = centroid[keep], pressure[keep]

        points_attr.Set(Vt.Vec3fArray.FromNumpy(centroid.astype(np.float32)), time=frame_index)
        colour_attr.Set(Vt.Vec3fArray.FromNumpy(_colours(pressure)), time=frame_index)
        width_attr.Set(
            Vt.FloatArray.FromNumpy(np.full(len(centroid), point_width, np.float32)),
            time=frame_index,
        )
        if frame.label:
            stage.SetMetadata("comment", frame.label) if frame_index == 0 else None

    # Keep the replayed scene from being simulated again: it is animation, not physics.
    for prim_path in ("/World/pad", "/World/sheet"):
        prim = stage.GetPrimAtPath(prim_path)
        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool,
                             custom=False).Set(False)
    stage.GetRootLayer().Save()
