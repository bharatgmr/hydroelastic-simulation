"""Reduce a contact patch to the loads the machine actually carries.

The force device pushes along the **tool axis** — the spindle centreline through the disc — while
the contact reaction acts wherever the patch happens to be. On a tilted, rim-referenced TCP that
is tens of millimetres off the axis, so the same contact force also feeds a bending moment back
through the spindle and the wrist. Summing pressure times area alone misses that entirely, which
is why the reduction lives here rather than inline in a script.

Three reference points matter and they are not interchangeable:

* **tool origin** — on the spindle axis at the disc face. What the AFD holds is the component of
  the reaction along the tool axis through this point; the perpendicular offset of the centre of
  pressure from that axis is the moment arm.
* **TCP origin** — the commanded point on the surface. What the planner positions.
* **surface normal** — what the part feels, which on a tilted tool is neither of the above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ContactLoads", "reduce_loads"]


@dataclass(frozen=True)
class ContactLoads:
    """Contact patch reduced to forces, moments and the geometry that produced them."""

    force_world: np.ndarray
    """Total contact force on the tool, world frame [N]."""
    force_axial: float
    """Component along the tool axis — the load the force device regulates [N]."""
    force_normal_surface: float
    """Component along the surface normal — the load the part feels [N]."""
    force_lateral: float
    """Magnitude of the in-plane (sideways) force at the surface [N]."""
    centre_of_pressure: np.ndarray
    """Pressure-weighted centroid of the patch, world frame [m]."""
    moment_arm: float
    """Perpendicular distance from the tool axis to the centre of pressure [m]."""
    moment_tool: np.ndarray
    """Moment about the tool origin, world frame [N m]."""
    bending_moment_tool: float
    """Component of that moment perpendicular to the tool axis — what bends the spindle [N m]."""
    torque_about_tool_axis: float
    """Component along the tool axis; zero without friction, kept so it stays visible [N m]."""
    moment_tcp: np.ndarray
    """Moment about the TCP origin, world frame [N m]."""
    bending_moment_tcp: float
    """Magnitude of that moment [N m]."""
    patch_area: float
    """Total contact area [m^2]."""
    pressure_peak: float
    """Highest face pressure [Pa]."""
    pressure_mean: float
    """Area-weighted mean pressure [Pa]."""
    n_faces: int
    """Number of contact faces the patch was built from."""


def reduce_loads(
    centroid: np.ndarray,
    normal: np.ndarray,
    pressure: np.ndarray,
    area: np.ndarray,
    *,
    tool_origin: np.ndarray,
    tool_axis: np.ndarray,
    tcp_origin: np.ndarray,
    surface_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> ContactLoads:
    """Reduce per-face contact data to the loads at the tool and at the TCP.

    Args:
        centroid: Face centroids, world frame ``[N, 3]`` [m].
        normal: Unit face normals ``[N, 3]``.
        pressure: Face pressures ``[N]`` [Pa].
        area: Face areas ``[N]`` [m^2].
        tool_origin: Point on the spindle axis, at the disc face [m].
        tool_axis: Unit vector along the spindle axis, pointing away from the part.
        tcp_origin: The commanded point on the surface [m].
        surface_normal: Outward normal of the part surface.

    Returns:
        The reduced :class:`ContactLoads`.
    """
    centroid = np.asarray(centroid, dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
    area = np.asarray(area, dtype=np.float64).reshape(-1)
    axis = _unit(tool_axis)
    surface = _unit(surface_normal)

    if centroid.size == 0:
        zero3 = np.zeros(3)
        return ContactLoads(zero3, 0.0, 0.0, 0.0, zero3, 0.0, zero3, 0.0, 0.0, zero3, 0.0,
                            0.0, 0.0, 0.0, 0)

    # Per-face force on the tool. Newton's face normals are unsigned with respect to which body
    # they push, so orient them to push the tool away from the part before summing.
    face_force = (pressure * area)[:, None] * normal
    if float((face_force @ surface).sum()) < 0.0:
        face_force = -face_force

    total = face_force.sum(axis=0)
    weight = pressure * area
    total_weight = float(weight.sum())
    cop = ((centroid * weight[:, None]).sum(axis=0) / total_weight) if total_weight > 0 else (
        centroid.mean(axis=0))

    # Moment arm: how far the centre of pressure sits off the spindle centreline.
    offset = cop - np.asarray(tool_origin, dtype=np.float64)
    arm = float(np.linalg.norm(offset - np.dot(offset, axis) * axis))

    moment_tool = np.cross(centroid - np.asarray(tool_origin, dtype=np.float64),
                           face_force).sum(axis=0)
    along_axis = float(moment_tool @ axis)
    bending_tool = float(np.linalg.norm(moment_tool - along_axis * axis))

    moment_tcp = np.cross(centroid - np.asarray(tcp_origin, dtype=np.float64),
                          face_force).sum(axis=0)

    in_plane = total - float(total @ surface) * surface
    return ContactLoads(
        force_world=total,
        force_axial=float(total @ axis),
        force_normal_surface=float(total @ surface),
        force_lateral=float(np.linalg.norm(in_plane)),
        centre_of_pressure=cop,
        moment_arm=arm,
        moment_tool=moment_tool,
        bending_moment_tool=bending_tool,
        torque_about_tool_axis=along_axis,
        moment_tcp=moment_tcp,
        bending_moment_tcp=float(np.linalg.norm(moment_tcp)),
        patch_area=float(area.sum()),
        pressure_peak=float(pressure.max()),
        pressure_mean=(total_weight / float(area.sum())) if float(area.sum()) > 0 else 0.0,
        n_faces=int(len(pressure)),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction vector must be non-zero")
    return vector / norm
