"""Tool centre point frames, read from the cell's own TCP calibration files.

Why this exists: the pad's pose is not "tilt about the disc centre". The planner holds a *TCP*
against the surface — origin on the surface, z along the surface normal — and the pad hangs off
that frame. A nominal TCP sits at the disc face centre, so the face lands parallel to the part.
A tilted TCP sits out near the rim and is rotated, so the same approach motion drives the rim in
at an angle. Same commanded travel, different pad pose, different patch, and a moment about the
TCP that a centre-referenced model does not produce at all.

TCP files live in the settings repo, e.g.
``ss/src/settings/tools/tcp/TCP_7in_Nominal.csv``, one line of
``x, y, z, qw, qx, qy, qz`` giving the flange -> TCP transform in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Transform", "TcpFrame", "read_tcp_csv", "waypoint_frame", "travel_direction"]


@dataclass(frozen=True)
class Transform:
    """A rigid transform: rotation matrix plus translation."""

    rotation: np.ndarray  # [3, 3]
    translation: np.ndarray  # [3]

    def inverse(self) -> "Transform":
        rot_t = self.rotation.T
        return Transform(rot_t, -rot_t @ self.translation)

    def __matmul__(self, other: "Transform") -> "Transform":
        return Transform(self.rotation @ other.rotation,
                         self.rotation @ other.translation + self.translation)

    def apply(self, point: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=np.float64) + self.translation

    @property
    def quaternion_xyzw(self) -> tuple[float, float, float, float]:
        """Rotation as (x, y, z, w), the order Newton's body_q uses."""
        matrix = self.rotation
        trace = matrix.trace()
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
        return (x, y, z, w)


def _quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_tcp_csv(path: str | Path) -> Transform:
    """Read one of the cell's TCP files as a flange -> TCP transform.

    Column order is ``X, Y, Z, Qx, Qy, Qz, Qw`` — **vector first, scalar last**, metres and a unit
    quaternion. This is the ROS convention and is what the cell's own loaders assume:
    ``trajectory_planner/src/utils.cpp:generateTCP`` builds
    ``Eigen::Quaterniond(v[6], v[3], v[4], v[5])`` (Eigen's constructor takes w first), and
    ``settings/tools_old/tcp/TCP.txt`` states "quaternion = xyzw (ROS convention)".

    Reading it scalar-first instead is silent and plausible — it yields a valid rotation, just the
    wrong one — so this is pinned by tests against the real files.
    """
    values = [float(v) for v in Path(path).read_text().strip().split(",")]
    if len(values) != 7:
        raise ValueError(f"{path}: expected 7 values, got {len(values)}")
    translation = np.array(values[0:3])
    x, y, z, w = values[3:7]
    return Transform(_quaternion_to_matrix(w, x, y, z), translation)


@dataclass(frozen=True)
class TcpFrame:
    """Where a TCP sits on the pad, and how to pose the pad from it.

    The pad frame here is the *nominal* TCP frame: origin at the disc face centre, z along the
    disc axis pointing into the part. ``pad_from_tcp`` is the fixed transform from that frame to
    this TCP, which is what distinguishes a rim-referenced tilted TCP from a centre one.
    """

    pad_from_tcp: Transform
    name: str = ""

    @classmethod
    def from_csv_pair(cls, nominal: str | Path, tilted: str | Path) -> "TcpFrame":
        """Derive the pad -> TCP transform from the cell's own calibration files.

        Both files give flange -> TCP, so the nominal one inverted and composed with the tilted
        one yields exactly the offset and rotation the tilted TCP carries relative to the disc
        face centre.
        """
        nominal_t = read_tcp_csv(nominal)
        tilted_t = read_tcp_csv(tilted)
        return cls(nominal_t.inverse() @ tilted_t, name=Path(tilted).stem)

    @classmethod
    def from_parameters(
        cls,
        *,
        radial_offset: float = 0.0,
        axial_offset: float = 0.0,
        tilt_deg: float = 0.0,
        azimuth_deg: float = 0.0,
        name: str = "",
    ) -> "TcpFrame":
        """Build a TCP by hand: out ``radial_offset`` at ``azimuth_deg``, tilted ``tilt_deg``.

        The disc leans **toward** the side the TCP sits on, so the rim under the TCP is the one
        that touches — that is what makes a tilted TCP a contact point rather than an arbitrary
        frame, and it is what the cell's own files do: ``TCP_7in_2.5deg_0000_Disk`` offsets to
        -x and leans about -y, ``..._0600_`` offsets to +x and leans about +y. The tilt therefore
        rotates about the in-plane axis perpendicular to the azimuth, signed so the offset side
        drops.

        The 4-digit code in the cell's filenames is the clock azimuth of that offset
        (``settings/tools_old/tcp/TCP.txt``): ``0000`` = +x, ``0300`` = +y, ``0600`` = -x,
        ``0900`` = -y.
        """
        azimuth = math.radians(azimuth_deg)
        tilt = math.radians(tilt_deg)
        direction = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
        axis = np.array([math.sin(azimuth), -math.cos(azimuth), 0.0])  # perpendicular, in-plane
        rotation = _axis_angle(axis, tilt)
        translation = direction * radial_offset + np.array([0.0, 0.0, axial_offset])
        return cls(Transform(rotation, translation), name=name)

    def spun(self, angle_deg: float) -> "TcpFrame":
        """Rotate the whole TCP about the tool axis — the clock position of the lean.

        This is the cell's own ``Rz(theta)`` term: ``T = T_nom . Rz(theta) . Tx(radius) . Ry(beta)``
        (``settings/tools_old/tcp/TCP.txt``), which is what separates the ``0000``/``0300``/``0600``
        /``0900`` files of one tool. It carries the radial offset and the tilt axis round together,
        so the disc still leans onto the rim where its TCP sits.

        Against a path, this is the lead/side distinction: at 0 the disc leans across the direction
        of travel (side tilt), at 90 it leans along it (lead tilt) — the planner's own
        ``tilt_tcp_orth_dir`` versus ``tilt_tcp_path_dir``.

        Args:
            angle_deg: Rotation about the tool axis [deg].

        Returns:
            A new frame; this one is unchanged.
        """
        # Post-multiplied, so the rotation is about the TCP's own axis: placing that TCP at a
        # fixed pose then swings the disc (and its lean) around it by -angle. Pre-multiplying
        # instead spins the disc about its own centre, which for an axisymmetric disc changes
        # nothing at all — the contact stays exactly where it was.
        spin = Transform(_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(angle_deg)),
                         np.zeros(3))
        suffix = f"+{angle_deg:g}deg" if angle_deg else ""
        return TcpFrame(self.pad_from_tcp @ spin, name=f"{self.name}{suffix}")

    @property
    def tilt_deg(self) -> float:
        """Angle between the pad axis and the TCP axis [deg]."""
        cosine = float(np.clip(self.pad_from_tcp.rotation[2, 2], -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    @property
    def radial_offset(self) -> float:
        """How far the TCP sits from the pad axis [m]."""
        return float(np.hypot(*self.pad_from_tcp.translation[:2]))

    @property
    def axial_offset(self) -> float:
        """How far the TCP sits along the pad axis [m]."""
        return float(self.pad_from_tcp.translation[2])

    @property
    def azimuth_deg(self) -> float:
        """Clock position of the TCP offset [deg]; 0 is +x, matching the ``0000`` naming."""
        x, y = self.pad_from_tcp.translation[:2]
        return math.degrees(math.atan2(y, x))

    def pad_pose(self, depth: float) -> tuple[tuple[float, float, float],
                                              tuple[float, float, float, float]]:
        """Pose the pad so the TCP sits ``depth`` into a sheet whose top face is z=0.

        This is what the machine does: the TCP origin goes to the commanded point on the surface
        (or ``depth`` past it), with the TCP axis along **minus** the surface normal. Where the pad
        ends up follows from the TCP, not the other way round.

        The sign is the planner's, not a choice: ``eigen_tools.cpp:compute_TCP_wrt_Vector`` sets
        ``tool_z = -normal``, and the taught-normal path records ``-tool1_Z`` so that ``vz`` comes
        back as ``+tool1_Z``. TCP +Z is also the disc face's outward normal — the 7" sphere model
        puts the whole pad at flange ``x = -0.3525`` with the nominal TCP translation at exactly
        that point, and the TCP quaternion maps its +Z onto flange -X.

        Args:
            depth: Travel past the surface along the TCP axis [m]; positive presses in.

        Returns:
            ``((x, y, z), (qx, qy, qz, qw))`` for the pad body, in world coordinates, with the
            pad mesh built base-at-z=0 and its axis along +z.
        """
        # A flat plate at the origin: the TCP axis is world -Z, which is the waypoint frame
        # vz = -normal for normal = +Z.
        tcp_in_world = Transform(_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi),
                                 np.array([0.0, 0.0, -depth]))
        return self.pad_pose_at(tcp_in_world)

    def pad_pose_at(self, tcp_in_world: "Transform") -> tuple[tuple[float, float, float],
                                                              tuple[float, float, float, float]]:
        """Pose the pad for an arbitrary TCP pose — a waypoint anywhere on any surface.

        The caller supplies where the TCP goes and how it is oriented; a waypoint frame from
        :func:`waypoint_frame` is exactly that, since its ``vz`` is the TCP axis. Travel into the
        surface is a translation along that frame's ``vz``.

        Args:
            tcp_in_world: World pose of the TCP frame.

        Returns:
            ``((x, y, z), (qx, qy, qz, qw))`` for the pad body.
        """
        pad_in_world = tcp_in_world @ self.pad_from_tcp.inverse()
        # The pad *frame* has +z through the disc face (into the part); the pad *mesh* is built
        # face-at-z=0 with its body along +z, so the mesh is the frame turned over. Flip about x
        # so a TCP offset at azimuth 0 stays on the mesh's +x side.
        body = pad_in_world @ Transform(_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi),
                                        np.zeros(3))
        return tuple(body.translation), body.quaternion_xyzw

    def describe(self) -> str:
        """One line summarising the frame, for run logs."""
        return (f"{self.name or 'tcp'}: tilt {self.tilt_deg:.2f} deg, radial "
                f"{self.radial_offset*1e3:.1f} mm at azimuth {self.azimuth_deg:.0f} deg, axial "
                f"{self.axial_offset*1e3:.1f} mm")


def waypoint_frame(normal: np.ndarray, direction: np.ndarray) -> Transform:
    """Build the planner's waypoint triad at a surface point.

    Mirrors ``eigen_tools.cpp:compute_TCP_wrt_Vector`` plus the outer column negation in
    ``TppUtils::compute_TCP``, which together come out as::

        vz = -normal              # into the part
        vx = direction x normal   # in-plane SIDE axis, not the travel axis
        vy = vz x vx              # travel runs along -vy, which equals +direction

    The cross product takes the **raw outward normal**, not ``vz``; using ``vz`` there flips ``vx``
    and sends the tool down the path backwards. The planner's own worked example
    (``tool_path_planner/README.md``) pins it: a flat panel with ``direction_vec = +Y`` gives
    ``vx = +X``.

    ``direction`` is the segment's azimuth vector (the planner's ``direction_vec``, chosen per cube
    face and spun by ``tcp_rotation_angle_<face>``), not the travel direction — the travel
    direction comes out as ``-vy``. Getting that backwards rotates every waypoint by 90 degrees.

    Args:
        normal: Outward surface normal at the waypoint (away from the material).
        direction: The segment's azimuth vector; need not be perpendicular to ``normal``.

    Returns:
        A rotation-only :class:`Transform` whose columns are ``vx, vy, vz``.

    Raises:
        ValueError: If ``direction`` is parallel to ``normal``, which leaves ``vx`` undefined.
    """
    outward = _unit(normal)
    vz = -outward
    cross = np.cross(_unit(direction), outward)
    if float(np.linalg.norm(cross)) < 1e-9:
        raise ValueError("direction is parallel to the normal; the waypoint frame is undefined")
    vx = cross / np.linalg.norm(cross)
    vy = np.cross(vz, vx)
    return Transform(np.column_stack([vx, vy, vz]), np.zeros(3))


def travel_direction(frame: Transform) -> np.ndarray:
    """Travel direction of a waypoint frame: ``-vy``, per the planner's triad."""
    return -frame.rotation[:, 1]


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalise, refusing a zero vector rather than returning NaNs."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction vector must be non-zero")
    return vector / norm


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for ``angle`` radians about a unit ``axis`` (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3)
    axis = axis / norm
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
