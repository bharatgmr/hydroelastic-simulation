"""A planner-shaped hatch toolpath over a flat plate.

This reproduces the spacing arithmetic the cell's tool path planner does, so the waypoints the sim
presses at are the waypoints the cell would plan — not an invented raster. From
``TppUtils::calculateInferredParameters``::

    max_displacement      = tool_contact_width                       # the disc DIAMETER
    requested_displacement = max_displacement * (1 - overlap_percent/100)   # hatch-line pitch

and points along a line are spaced at ``pathgap_sanding_line``. Each waypoint carries the planner's
own frame (``grind_sim.tcp.waypoint_frame``): ``vz = -normal``, ``vx = direction x normal``,
travel along ``-vy``.

Two conventions the planner's source did not settle, recorded per path so a run says which it used:

* **Row alternation.** Whether TPP serpentines alternate rows was not established; default here is
  serpentine (``alternate=True``), which is what minimises air moves.
* **Tilt sign.** ``tilt_toolpath`` can negate lead/ortho tilt and ``tcp_offset`` together via
  ``flipTiltDirectionToMaintainWithWorld`` depending on traversal direction. Nothing here tilts the
  path itself — tilt arrives through the TCP frame — but a comparison against real cell output has
  to account for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from grind_sim.tcp import Transform, travel_direction, waypoint_frame

__all__ = ["PlateSpec", "PathParams", "Waypoint", "hatch_path"]


@dataclass(frozen=True)
class PlateSpec:
    """A flat rectangular plate lying in the z=0 plane, outward normal +z."""

    size_x: float = 0.300
    size_y: float = 0.300
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class PathParams:
    """Path spacing, in the planner's own terms.

    Defaults are the S2 - FR Corner - Front Top part class, except ``overlap_percent``, which the
    planner takes from the pass (operator-authored), not the part class.
    """

    tool_contact_width: float = 0.1778
    """Disc diameter [m] — the planner's ``tpp_tool_characteristic.tool_contact_width``."""
    overlap_percent: float = 50.0
    """Overlap between adjacent hatch lines [%]; from the pass, typical cell values 50-75."""
    pathgap_sanding_line: float = 0.0075
    """Spacing between waypoints along a hatch line [m]."""
    hatch_angle_deg: float = 0.0
    """Rotation of the hatch pattern within the surface plane [deg]."""
    margin: float = 0.0
    """Inset from the plate edge [m]; 0 runs the disc centre right to the boundary."""
    alternate: bool = True
    """Serpentine rows (reverse every other line) rather than always travelling the same way."""

    @property
    def pitch(self) -> float:
        """Hatch-line pitch [m]: ``tool_contact_width * (1 - overlap/100)``."""
        if self.tool_contact_width <= 0.0:
            raise ValueError("tool_contact_width must be positive")
        if not 0.0 <= self.overlap_percent < 100.0:
            raise ValueError("overlap_percent must be in [0, 100)")
        return self.tool_contact_width * (1.0 - self.overlap_percent / 100.0)


@dataclass(frozen=True)
class Waypoint:
    """One planned point: where the tool goes and how it is held there."""

    position: np.ndarray
    """Surface point [m], world frame."""
    frame: Transform
    """Planner triad; columns are ``vx, vy, vz``."""
    line_index: int
    """Which hatch line this point belongs to."""
    index_in_line: int

    @property
    def travel(self) -> np.ndarray:
        """Unit travel direction, ``-vy``."""
        return travel_direction(self.frame)

    @property
    def normal(self) -> np.ndarray:
        """Outward surface normal, ``-vz``."""
        return -self.frame.rotation[:, 2]


@dataclass
class ToolPath:
    """A planned path plus the parameters and conventions that produced it."""

    waypoints: list[Waypoint] = field(default_factory=list)
    params: PathParams = field(default_factory=PathParams)
    plate: PlateSpec = field(default_factory=PlateSpec)
    conventions: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def positions(self) -> np.ndarray:
        """All waypoint positions as ``[N, 3]``."""
        return np.array([w.position for w in self.waypoints]) if self.waypoints else np.zeros((0, 3))

    @property
    def path_length(self) -> float:
        """Total travelled distance [m], including the step across between lines."""
        points = self.positions
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def swept_area(self) -> float:
        """The planner's own coverage estimate: ``path length * tool_contact_width`` [m^2].

        ``tool_path_planner.cpp`` reports this as ``true_surface_area``. It is a rectangle-sweep
        approximation that ignores tilt, overlap double-counting and the disc's round ends, so it
        is useful only as the number the cell would quote.
        """
        return self.path_length * self.params.tool_contact_width


def hatch_path(plate: PlateSpec = PlateSpec(), params: PathParams = PathParams()) -> ToolPath:
    """Lay out hatch lines over a flat plate, planner-style.

    Lines run along the hatch direction, spaced by :attr:`PathParams.pitch` across it, with points
    every ``pathgap_sanding_line`` along each line. The first and last lines are placed half a pitch
    inside the plate so the swept band stays on the part.

    Args:
        plate: The plate to cover.
        params: Spacing parameters.

    Returns:
        The path, with each waypoint carrying the planner's frame.
    """
    angle = math.radians(params.hatch_angle_deg)
    along = np.array([math.cos(angle), math.sin(angle), 0.0])   # hatch-line direction
    across = np.array([-math.sin(angle), math.cos(angle), 0.0])  # line-to-line step
    normal = np.array(plate.normal, dtype=np.float64)

    # Extent of the plate along each of those two directions, from its corners.
    corners = np.array([[sx * plate.size_x / 2.0, sy * plate.size_y / 2.0, 0.0]
                        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
    along_span = corners @ along
    across_span = corners @ across
    along_lo, along_hi = along_span.min() + params.margin, along_span.max() - params.margin
    across_lo, across_hi = across_span.min() + params.margin, across_span.max() - params.margin
    if along_hi <= along_lo or across_hi <= across_lo:
        raise ValueError("margin leaves no area to cover")

    pitch = params.pitch
    width = across_hi - across_lo
    n_lines = max(1, int(math.floor(width / pitch)) + 1)
    # Centre the set of lines in the available width rather than hugging one edge.
    used = pitch * (n_lines - 1)
    first = across_lo + (width - used) / 2.0

    n_points = max(2, int(math.floor((along_hi - along_lo) / params.pathgap_sanding_line)) + 1)
    offsets = along_lo + np.arange(n_points) * params.pathgap_sanding_line

    waypoints: list[Waypoint] = []
    for line in range(n_lines):
        sequence = offsets if (not params.alternate or line % 2 == 0) else offsets[::-1]
        # Travel direction reverses on alternate lines, and the frame follows it.
        direction = along if (not params.alternate or line % 2 == 0) else -along
        frame = waypoint_frame(normal=normal, direction=direction)
        for k, s in enumerate(sequence):
            position = along * s + across * (first + line * pitch)
            waypoints.append(Waypoint(position=position, frame=frame, line_index=line,
                                      index_in_line=k))

    return ToolPath(
        waypoints=waypoints,
        params=params,
        plate=plate,
        conventions={
            "row_alternation": "serpentine" if params.alternate else "unidirectional",
            "row_alternation_verified_against_tpp": False,
            "pitch_formula": "tool_contact_width * (1 - overlap_percent/100)",
            "frame": "vz = -normal, vx = direction x normal, travel = -vy",
            "tilt_sign_flip_accounted": False,
        },
    )
