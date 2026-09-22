"""Turn a scanned profile cloud into something hydroelastic contact can actually run on.

The cell's scans are ``mergedPointcloud.ply`` files of ``PointXYZRGBNormal``, with the target
material painted pure green ``(0, 255, 0)``. Newton's hydroelastic contact needs a **watertight,
volumetric** mesh on both sides of the pair — it silently refuses planes and heightfields — so a
raw scan is not usable as-is. What works is what sanding-wm called a "baked closed-slab mesh":
resample the surface onto a regular grid in a local frame, then close it into a slab by offsetting
a back face and stitching the sides.

Everything here is pure NumPy/trimesh: no Isaac, no GPU, so it is testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

__all__ = ["Cloud", "LocalFrame", "band_centreline", "load_cloud", "resample_line",
           "surface_normals", "slab_mesh"]

GREEN = (0, 255, 0)

_PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
])


@dataclass(frozen=True)
class Cloud:
    """A scan: points, normals and colours, in the cell's world frame [m]."""

    points: np.ndarray
    normals: np.ndarray
    colours: np.ndarray

    def __len__(self) -> int:
        return len(self.points)

    def mask_colour(self, rgb: tuple[int, int, int] = GREEN) -> np.ndarray:
        """Boolean mask of points painted exactly this colour."""
        return np.all(self.colours == np.asarray(rgb, dtype=np.uint8), axis=1)


def load_cloud(path: str | Path) -> Cloud:
    """Read a binary-little-endian ``PointXYZRGBNormal`` PLY as written by the cell.

    Parsed directly rather than through a PLY library: the format is fixed, the files are large,
    and a single ``np.frombuffer`` avoids copying ~1.4M points through Python objects.

    Args:
        path: The ``mergedPointcloud.ply`` to read.

    Returns:
        The cloud.

    Raises:
        ValueError: If the header is not the layout this parser expects.
    """
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path}: no end_header")
    header = raw[:end].split(b"\n")
    if not any(line.startswith(b"format binary_little_endian") for line in header):
        raise ValueError(f"{path}: expected binary_little_endian")
    counts = [line for line in header if line.startswith(b"element vertex")]
    if not counts:
        raise ValueError(f"{path}: no vertex count")
    count = int(counts[0].split()[-1])

    data = np.frombuffer(raw, dtype=_PLY_DTYPE, count=count, offset=end + len(marker))
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float64)
    colours = np.stack([data["r"], data["g"], data["b"]], axis=1)
    return Cloud(points=points, normals=normals, colours=colours)


@dataclass(frozen=True)
class LocalFrame:
    """A right-handed frame fitted to a marked region: x along it, y across, z out of the part."""

    origin: np.ndarray
    rotation: np.ndarray
    """Columns are the local x, y, z axes in world coordinates."""

    @classmethod
    def fit(cls, points: np.ndarray, normals: np.ndarray) -> "LocalFrame":
        """Fit by PCA, with z taken from the scan normals so "out of the part" is unambiguous.

        The longest principal axis becomes x (along the band), and y completes the frame. Using
        the mean scan normal for z rather than the smallest principal axis matters on a curved
        band, where the third axis can be dominated by curvature instead of thickness.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        centre = points.mean(axis=0)
        _, _, basis = np.linalg.svd(points - centre, full_matrices=False)

        z_axis = np.asarray(normals, dtype=np.float64).reshape(-1, 3).mean(axis=0)
        norm = np.linalg.norm(z_axis)
        z_axis = z_axis / norm if norm > 1e-12 else basis[2]

        x_axis = basis[0] - np.dot(basis[0], z_axis) * z_axis  # longest axis, flattened into the surface
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:  # the band runs along the normal: fall back to the second axis
            x_axis = basis[1] - np.dot(basis[1], z_axis) * z_axis
            x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        return cls(origin=centre, rotation=np.column_stack([x_axis, y_axis, z_axis]))

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """World -> local."""
        return (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.origin) @ self.rotation

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """Local -> world."""
        return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ self.rotation.T + self.origin

    def direction_to_world(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate local directions into world, without translating."""
        return np.asarray(vectors, dtype=np.float64).reshape(-1, 3) @ self.rotation.T


def band_centreline(
    local_points: np.ndarray,
    *,
    n_stations: int = 60,
    smooth: int = 5,
) -> np.ndarray:
    """Trace the centre of a marked band, in the band's own local frame.

    The band is binned along its length; each bin contributes the midpoint of its cross-section
    (the mean of the 10th and 90th percentile across the band, which ignores the ragged paint edge
    that a median would follow into), and the surface height there. Empty bins are dropped, and
    the result is smoothed with a short moving average so the tool is not steered by scan noise.

    Args:
        local_points: Marked points in local coordinates ``[N, 3]``.
        n_stations: Bins along the band's length.
        smooth: Moving-average window in stations; 1 disables smoothing.

    Returns:
        ``[M, 3]`` centreline points in local coordinates, ordered along the band.

    Raises:
        ValueError: If no station has any points.
    """
    local_points = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
    edges = np.linspace(local_points[:, 0].min(), local_points[:, 0].max(), n_stations + 1)
    stations = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = local_points[(local_points[:, 0] >= lo) & (local_points[:, 0] < hi)]
        if len(inside) < 3:
            continue
        across = 0.5 * (np.percentile(inside[:, 1], 10) + np.percentile(inside[:, 1], 90))
        stations.append([0.5 * (lo + hi), across, np.median(inside[:, 2])])
    if not stations:
        raise ValueError("no station along the band contained enough points")

    line = np.array(stations)
    if smooth > 1 and len(line) >= smooth:
        kernel = np.ones(smooth) / smooth
        half = smooth // 2
        for column in (1, 2):
            # Reflect at the ends before smoothing. A 'same' convolution tapers into zero there,
            # and clamping the result instead leaves a step at the join — which shows up as a
            # waypoint whose tangent is nowhere near the surface.
            padded = np.pad(line[:, column], half, mode="reflect")
            line[:, column] = np.convolve(padded, kernel, mode="valid")[: len(line)]
    return line


def slab_mesh(
    local_points: np.ndarray,
    *,
    cell: float = 0.002,
    thickness: float = 0.02,
    margin: float = 0.12,
    bounds: tuple[float, float, float, float] | None = None,
) -> trimesh.Trimesh:
    """Resample scanned points into a watertight slab, in the region's local frame.

    The top face is the scanned surface on a regular grid; the bottom is that surface pushed down
    by ``thickness``; the sides are stitched. Closed by construction, which is what the SDF needs.
    Gaps in the scan are filled from the nearest samples, so the surface stays continuous rather
    than punching holes the contact would fall through.

    Args:
        local_points: Scan points in local coordinates ``[N, 3]``; pass the neighbourhood, not
            just the marked band, or the disc will overhang the mesh.
        cell: Grid spacing [m].
        thickness: Slab depth [m]; only needs to exceed the SDF's inner narrow band.
        margin: How far past the point cloud's extent to extend the grid [m].
        bounds: Optional ``(x_min, x_max, y_min, y_max)`` in local coordinates, overriding the
            extent-plus-margin default.

    Returns:
        A watertight mesh in local coordinates.

    Raises:
        ValueError: If the grid would be degenerate, or the result is not watertight.
    """
    local_points = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
    if bounds is None:
        x_min, y_min = local_points[:, :2].min(axis=0) - margin
        x_max, y_max = local_points[:, :2].max(axis=0) + margin
    else:
        x_min, x_max, y_min, y_max = bounds
    nx = int(np.floor((x_max - x_min) / cell)) + 1
    ny = int(np.floor((y_max - y_min) / cell)) + 1
    if nx < 2 or ny < 2:
        raise ValueError("grid is degenerate; check cell size against the region extent")

    xs = x_min + np.arange(nx) * cell
    ys = y_min + np.arange(ny) * cell
    heights = _grid_heights(local_points, xs, ys)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    top = np.stack([grid_x.ravel(), grid_y.ravel(), heights.ravel()], axis=1)
    bottom = top.copy()
    bottom[:, 2] -= thickness
    vertices = np.vstack([top, bottom])

    faces = []
    index = np.arange(nx * ny).reshape(nx, ny)
    offset = nx * ny
    a, b = index[:-1, :-1], index[1:, :-1]
    c, d = index[1:, 1:], index[:-1, 1:]
    faces.append(np.stack([a, b, c], axis=-1).reshape(-1, 3))
    faces.append(np.stack([a, c, d], axis=-1).reshape(-1, 3))
    faces.append(np.stack([a + offset, c + offset, b + offset], axis=-1).reshape(-1, 3))
    faces.append(np.stack([a + offset, d + offset, c + offset], axis=-1).reshape(-1, 3))

    def wall(edge: np.ndarray, flip: bool) -> np.ndarray:
        lower, upper = edge[:-1], edge[1:]
        quad = [lower, upper, upper + offset, lower + offset]
        first = np.stack([quad[0], quad[1], quad[2]], axis=-1)
        second = np.stack([quad[0], quad[2], quad[3]], axis=-1)
        pair = np.vstack([first, second])
        return pair[:, ::-1] if flip else pair

    faces.append(wall(index[0, :], flip=False))
    faces.append(wall(index[-1, :], flip=True))
    faces.append(wall(index[:, 0], flip=True))
    faces.append(wall(index[:, -1], flip=False))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.vstack(faces), process=True)
    mesh.fix_normals()
    if not mesh.is_watertight:
        raise ValueError("slab mesh is not watertight; the SDF would be meaningless")
    return mesh


def _grid_heights(points: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Median height per grid cell, with empty cells filled from their nearest filled neighbour."""
    cell_x = xs[1] - xs[0]
    cell_y = ys[1] - ys[0]
    ix = np.clip(((points[:, 0] - xs[0]) / cell_x).round().astype(int), 0, len(xs) - 1)
    iy = np.clip(((points[:, 1] - ys[0]) / cell_y).round().astype(int), 0, len(ys) - 1)
    flat = ix * len(ys) + iy

    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    z_sorted = points[order, 2]
    unique, starts = np.unique(flat_sorted, return_index=True)
    medians = np.array([np.median(chunk) for chunk in np.split(z_sorted, starts[1:])])

    heights = np.full(len(xs) * len(ys), np.nan)
    heights[unique] = medians
    heights = heights.reshape(len(xs), len(ys))

    missing = np.isnan(heights)
    if missing.any():
        from scipy.ndimage import distance_transform_edt

        _, (src_x, src_y) = distance_transform_edt(missing, return_indices=True)
        heights = heights[src_x, src_y]
    return heights


def resample_line(line: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at a fixed arc-length spacing, with unit tangents.

    The centreline comes out of binning at whatever station spacing the bins gave; the tool wants
    waypoints a fixed distance apart, which is what ``pathgap_sanding_line`` means in the planner.

    Args:
        line: Polyline ``[M, 3]``.
        spacing: Desired spacing along the curve [m].

    Returns:
        ``(points [K, 3], tangents [K, 3])``.

    Raises:
        ValueError: If the line is shorter than one spacing.
    """
    line = np.asarray(line, dtype=np.float64).reshape(-1, 3)
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] < spacing:
        raise ValueError(f"line is {arc[-1]*1e3:.1f} mm long, shorter than the {spacing*1e3:.1f} mm spacing")

    wanted = np.arange(0.0, arc[-1] + 1e-12, spacing)
    points = np.column_stack([np.interp(wanted, arc, line[:, axis]) for axis in range(3)])
    tangents = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    return points, tangents / np.maximum(lengths, 1e-12)


def surface_normals(query: np.ndarray, points: np.ndarray, *, radius: float = 0.015,
                    reference: np.ndarray | None = None, smooth: int = 5) -> np.ndarray:
    """Estimate outward surface normals at ``query`` by fitting a plane to nearby scan points.

    Taken from the scan rather than from the resampled grid so the tool is oriented by the surface
    the part actually has. Normals are flipped to agree with ``reference`` (the region's own
    outward axis), because a plane fit has no inherent side.

    Args:
        query: Points to estimate at ``[K, 3]``.
        points: The scan neighbourhood ``[N, 3]``, same frame.
        radius: Neighbourhood radius for the fit [m].
        reference: Outward direction to agree with; defaults to +z of the frame.
        smooth: Moving-average window along the query sequence; 1 disables it. The query points
            are consecutive waypoints, so smoothing here stops one noisy plane fit from jerking
            the tool at a single station.

    Returns:
        Unit normals ``[K, 3]``.
    """
    from scipy.spatial import cKDTree

    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    reference = np.array([0.0, 0.0, 1.0]) if reference is None else np.asarray(reference, float)

    tree = cKDTree(points)
    normals = np.zeros_like(query)
    for i, neighbours in enumerate(tree.query_ball_point(query, radius)):
        if len(neighbours) < 8:  # too sparse to fit; fall back to the region's own axis
            normals[i] = reference
            continue
        local = points[neighbours]
        _, _, basis = np.linalg.svd(local - local.mean(axis=0), full_matrices=False)
        normal = basis[2]
        normals[i] = normal if np.dot(normal, reference) >= 0.0 else -normal
    return normals
