"""Reconstruct the per-face contact field from Newton's hydroelastic surface.

VENDORED from sanding-wm (`src/sanding_wm/processes/sanding/engine/pressure.py`,
commit d5ff5a59d640228297704b84ba0635e09960709d, 2026-09-04), unmodified apart from this
header. The two repos have separate virtualenvs — sanding-wm is py3.12 + uv, this sandbox runs
Isaac Sim's own interpreter — so the file is copied rather than imported. It is pure NumPy with
no Newton or Isaac imports. Re-sync it if the upstream contact law changes.

Newton's :class:`~newton.geometry.HydroelasticSDF.ContactSurfaceData` exposes the
raw contact-surface triangles but, unlike Drake's ``ContactSurface``, it does
*not* expose per-face pressure, area, or normal. We reconstruct them here, in
pure NumPy, so the engine seam (``ContactField``) stays Newton-agnostic.

The reconstruction is faithful to the design spec rather than a hack:

* **Pressure** ``P_i = kh[shape_b] * |depth_i|`` is the elastic-foundation
  pressure of ``eq:efm`` (``p = E*phi/H``), with Newton's per-shape stiffness
  ``kh`` playing the role of ``E/H``. This is Newton's *own* contact law, not a
  reinterpretation: ``sdf_hydroelastic.py`` generates each face contact as
  ``face_pressure = pressure_func(pen_depth, shape_b, ...)`` with the default
  ``linear_pressure = -kh[shape_idx] * signed_depth``. Note the cross-pairing —
  **``depth`` belongs to shape A, the stiffness to shape B** — and it is the
  physically right one: the marching cubes samples shape A's SDF
  (``corner_sdf_vals[i] = valA``), so ``|depth_i|`` is how far *B* has
  penetrated into A, i.e. B's compression, and the pressure is B's stiffness
  times it.

  This previously used the series stiffness ``k_eff = (k_a*k_b)/(k_a + k_b)``.
  That agrees with ``kh[shape_b]`` to ~1% whenever ``k_a >> k_b`` — which holds
  for every primitive scene in this repo, because Newton returns those pairs
  workpiece-first — but it is **not** order-independent, and Newton does not
  guarantee the ordering. On a mesh workpiece the pair comes back pad-first at
  some ``sdf_target_voxel_size`` values, where ``k_eff`` under-reported the
  contact force by ~170x (0.116 N against an analytic 19.99 N) with a
  correct-looking patch. ``kh[shape_b]`` reproduces the analytic Winkler force in
  both orderings. ``effective_stiffness`` is kept for the disc engine's
  foam/paper/workpiece series and for the penetration inversion, which are
  genuinely series stiffnesses.
* **Area** ``A_i = 1/2 * ||(v1 - v0) x (v2 - v0)||`` is the triangle area from
  the three world-space face vertices.
* **Centroid** ``x_i`` is the mean of the three vertices.
* **Normal** ``n_i`` is the normalized cross product of two triangle edges.

These map onto ``docs/working-document/working_document.tex`` §2.3, Step 3
(``P_i``, ``A_i``, ``x_i``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FaceGeometry",
    "effective_stiffness",
    "face_geometry_from_vertices",
    "face_pressure",
    "reconstruct_faces",
]

# Numerical floor for degenerate (zero-area) triangles, mirroring Newton's
# MC_DEGENERATE_N_SQ_EPS treatment: such faces get a valid (non-NaN) normal and
# zero area, so they contribute nothing to the integrated force.
_DEGENERATE_AREA_EPS = 1.0e-18


class FaceGeometry:
    """Per-face geometry reconstructed from the three triangle vertices.

    All arrays share a leading face axis (``[N]`` scalars, ``[N, 3]`` vectors).
    """

    __slots__ = ("area", "centroid", "normal")

    def __init__(self, area: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> None:
        self.area = area
        self.centroid = centroid
        self.normal = normal


def effective_stiffness(k_a: np.ndarray, k_b: np.ndarray) -> np.ndarray:
    """Series (harmonic) effective stiffness ``k_eff = (k_a*k_b)/(k_a + k_b)``.

    This is ``eq:contact_force`` and matches Newton's ``get_effective_stiffness``
    ``@wp.func`` in ``sdf_hydroelastic.py`` byte-for-byte (including the
    non-positive-denominator guard, which returns ``0``).

    Args:
        k_a: Per-pair stiffness of shape ``a`` (Pa/m), any broadcastable shape.
        k_b: Per-pair stiffness of shape ``b`` (Pa/m), same shape as ``k_a``.

    Returns:
        The effective series stiffness, ``0`` wherever ``k_a + k_b <= 0``.
    """
    k_a = np.asarray(k_a, dtype=np.float64)
    k_b = np.asarray(k_b, dtype=np.float64)
    denom = k_a + k_b
    with np.errstate(divide="ignore", invalid="ignore"):
        k_eff = np.where(denom > 0.0, (k_a * k_b) / denom, 0.0)
    return k_eff


def face_geometry_from_vertices(verts: np.ndarray) -> FaceGeometry:
    """Reconstruct area, centroid, and unit normal from triangle vertices.

    Args:
        verts: World-space vertices, shape ``[N, 3, 3]`` — ``N`` faces, three
            vertices each, three coordinates each.

    Returns:
        A :class:`FaceGeometry` with ``area`` ``[N]``, ``centroid`` ``[N, 3]``,
        and unit ``normal`` ``[N, 3]``. Degenerate triangles get zero area and a
        ``+z`` placeholder normal (matching Newton's own convention).
    """
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim != 3 or verts.shape[1:] != (3, 3):
        raise ValueError(f"verts must be [N, 3, 3], got {verts.shape}")

    v0, v1, v2 = verts[:, 0, :], verts[:, 1, :], verts[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)  # [N, 3]
    cross_norm = np.linalg.norm(cross, axis=1)  # [N]
    area = 0.5 * cross_norm
    centroid = (v0 + v1 + v2) / 3.0

    degenerate = cross_norm <= _DEGENERATE_AREA_EPS
    safe_norm = np.where(degenerate, 1.0, cross_norm)
    normal = cross / safe_norm[:, None]
    # Degenerate faces: give a finite placeholder normal so downstream math
    # never sees NaN; their zero area zeroes any force contribution anyway.
    normal[degenerate] = np.array([0.0, 0.0, 1.0])
    return FaceGeometry(area=area, centroid=centroid, normal=normal)


def face_pressure(depth: np.ndarray, kh: np.ndarray) -> np.ndarray:
    """Elastic-foundation pressure ``P_i = kh_i * |depth_i|`` (``eq:efm``).

    Args:
        depth: Per-face centroid penetration depth (m); Newton reports this
            negative inside the overlap, so the magnitude is taken.
        kh: Per-face stiffness (Pa/m). For a Newton contact surface this is
            ``shape_kh[shape_b]`` — the *other* shape from the one the depth was
            sampled in; see the module docstring.

    Returns:
        Non-negative per-face pressure (Pa).
    """
    return np.asarray(kh, dtype=np.float64) * np.abs(np.asarray(depth, dtype=np.float64))


def reconstruct_faces(
    points: np.ndarray,
    depth: np.ndarray,
    shape_pair: np.ndarray,
    shape_kh: np.ndarray,
    n_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the full per-face contact field from a Newton surface dump.

    Args:
        points: Flat world-space vertex buffer, shape ``[>=3*n_faces, 3]``; the
            three vertices of face ``i`` are rows ``3*i, 3*i+1, 3*i+2`` (this is
            exactly Newton's ``contact_surface_point`` layout).
        depth: Per-face centroid penetration depth, shape ``[>=n_faces]``
            (``contact_surface_depth``).
        shape_pair: Per-face ``(shape_a, shape_b)`` indices, shape
            ``[>=n_faces, 2]`` (``contact_surface_shape_pair``).
        shape_kh: Per-shape hydroelastic stiffness ``kh`` (``E/H``, Pa/m), shape
            ``[n_shapes]`` (``model.shape_material_kh``).
        n_faces: Number of valid faces (``face_contact_count[0]``); the buffers
            are sliced to this length before reconstruction.

    Returns:
        Tuple ``(pressure, area, centroid, normal, shape_pair_valid)``:
        ``pressure`` ``[n_faces]`` (Pa), ``area`` ``[n_faces]`` (m^2),
        ``centroid`` ``[n_faces, 3]`` (m, world frame), ``normal`` ``[n_faces, 3]``
        (unit), ``shape_pair_valid`` ``[n_faces, 2]`` (int).
    """
    n = int(n_faces)
    if n <= 0:
        empty1 = np.zeros((0,), dtype=np.float64)
        empty3 = np.zeros((0, 3), dtype=np.float64)
        return empty1, empty1, empty3, empty3, np.zeros((0, 2), dtype=np.int64)

    # Slice to the valid face count *before* the float64 cast: Newton's vertex /
    # depth buffers are allocated with wp.empty, so the tail beyond n_faces holds
    # uninitialized (possibly NaN/inf) values that would otherwise warn on cast.
    verts = np.asarray(points)[: 3 * n].astype(np.float64).reshape(n, 3, 3)
    d = np.asarray(depth)[:n].astype(np.float64)
    pairs = np.asarray(shape_pair)[:n].astype(np.int64)  # [n, 2]
    kh = np.asarray(shape_kh, dtype=np.float64)

    geom = face_geometry_from_vertices(verts)
    # kh of shape B against a depth sampled in shape A — Newton's own pairing,
    # and order-independent. See the module docstring.
    pressure = face_pressure(d, kh[pairs[:, 1]])

    return pressure, geom.area, geom.centroid, geom.normal, pairs
