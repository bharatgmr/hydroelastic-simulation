"""TCP frame maths, checked against the cell's real calibration files.

The distinction under test: a nominal TCP sits at the disc face centre so the face lands parallel
to the part, while a tilted TCP sits out near the rim, so the same commanded travel produces a
different pad pose entirely.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from grind_sim.tcp import (
    TcpFrame,
    Transform,
    read_tcp_csv,
    travel_direction,
    waypoint_frame,
)

TCP_DIR = Path("/home/gmr/dev-docker-CAT-GRIND_v7/ss/src/settings/tools/tcp")
NOMINAL = TCP_DIR / "TCP_7in_Nominal.csv"
TILTED = TCP_DIR / "TCP_7in_2.5deg_0000_Disk.csv"

real_tcps = pytest.mark.skipif(not NOMINAL.exists(), reason="settings repo not checked out here")


def test_transform_inverse_round_trips():
    frame = TcpFrame.from_parameters(radial_offset=0.05, tilt_deg=7.0, azimuth_deg=30.0)
    combined = frame.pad_from_tcp @ frame.pad_from_tcp.inverse()
    assert np.allclose(combined.rotation, np.eye(3), atol=1e-12)
    assert np.allclose(combined.translation, np.zeros(3), atol=1e-12)


def test_quaternion_matches_the_rotation_it_came_from():
    for angle in (0.0, 0.3, math.pi / 2, 2.0):
        frame = TcpFrame.from_parameters(tilt_deg=math.degrees(angle))
        x, y, z, w = frame.pad_from_tcp.quaternion_xyzw
        assert pytest.approx(1.0, abs=1e-12) == x * x + y * y + z * z + w * w
        rebuilt = Transform(
            np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]),
            np.zeros(3),
        )
        assert np.allclose(rebuilt.rotation, frame.pad_from_tcp.rotation, atol=1e-12)


def test_parameters_round_trip_through_the_properties():
    frame = TcpFrame.from_parameters(radial_offset=0.0762, axial_offset=0.0225,
                                     tilt_deg=7.5, azimuth_deg=0.0)
    assert frame.radial_offset == pytest.approx(0.0762)
    assert frame.axial_offset == pytest.approx(0.0225)
    assert frame.tilt_deg == pytest.approx(7.5)
    assert frame.azimuth_deg == pytest.approx(0.0)


def rim_points(frame: TcpFrame, depth: float = 0.0, radius: float = 0.0889) -> np.ndarray:
    """World-space points around the disc rim at a given TCP travel."""
    (x, y, z), (qx, qy, qz, qw) = frame.pad_pose(depth)
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    angles = np.linspace(0.0, 2.0 * math.pi, 721)
    rim = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], 1)
    return rim @ rotation.T + np.array([x, y, z])


def test_nominal_tcp_puts_the_face_flat_on_the_surface():
    """Zero tilt, zero offset: the whole rim lands on the surface at the commanded depth."""
    frame = TcpFrame.from_parameters()
    heights = rim_points(frame, depth=0.001)[:, 2]
    assert heights.max() - heights.min() == pytest.approx(0.0, abs=1e-9)  # face is parallel
    assert heights.mean() == pytest.approx(-0.001, abs=1e-9)              # and at the depth


def test_tcp_axis_opposes_the_surface_normal():
    """The planner's convention: vz = -normal, so the tool axis points INTO the part.

    This is the bug the user caught — the sim had the TCP axis along +normal, pointing out of
    the part, which silently mirrors every tilted frame.
    """
    frame = TcpFrame.from_parameters()
    (_, _, z_at_zero), _ = frame.pad_pose(0.0)
    (_, _, z_deeper), _ = frame.pad_pose(0.004)
    assert z_deeper < z_at_zero  # pressing in moves the disc down, into the sheet
    # And the disc face, not its back, is what descends: the rim sits at the commanded depth.
    assert rim_points(frame, depth=0.004)[:, 2].mean() == pytest.approx(-0.004, abs=1e-9)


def test_tilted_tcp_moves_the_pad_centre_off_the_commanded_point():
    """The whole point: the pad hangs off the TCP, so its centre is not under the contact."""
    frame = TcpFrame.from_parameters(radial_offset=0.0762, tilt_deg=7.5)
    (x, _, _), _ = frame.pad_pose(0.0)
    assert abs(x) > 0.05  # centre sits ~76 mm away from where the TCP touches


@real_tcps
@pytest.mark.parametrize("tcp_name", ["TCP_7in_2.5deg_0000_Disk.csv", "TCP_7in_2.5deg_0600_Disk.csv"])
def test_real_tilted_tcp_touches_next_to_its_own_axis(tcp_name):
    """A TCP is the contact point, so the tilted disc must touch near it — not on the far rim.

    This is the check that caught both frame bugs: reading the quaternion scalar-first, or
    pointing the TCP axis along +normal, each put the contact on the opposite rim, ~166 mm away.
    """
    frame = TcpFrame.from_csv_pair(NOMINAL, TCP_DIR / tcp_name)
    world = rim_points(frame)
    lowest = world[np.argmin(world[:, 2])]
    assert math.hypot(lowest[0], lowest[1]) < 0.02


@real_tcps
def test_real_tilted_tcps_lean_toward_their_own_offset():
    """Offset direction and lean direction agree — the rim under the TCP is the one that drops."""
    for tcp_name in ("TCP_7in_2.5deg_0000_Disk.csv", "TCP_7in_2.5deg_0600_Disk.csv"):
        frame = TcpFrame.from_csv_pair(NOMINAL, TCP_DIR / tcp_name)
        world = rim_points(frame)
        lowest = world[np.argmin(world[:, 2])]
        # Everything in world: the TCP sits at the origin, the disc centre hangs off to one side,
        # and the rim that drops must be the one on the TCP's side of that centre.
        (cx, cy, _), _ = frame.pad_pose(0.0)
        centre_to_contact = np.array([lowest[0] - cx, lowest[1] - cy])
        centre_to_tcp = np.array([-cx, -cy])
        assert float(np.dot(centre_to_contact, centre_to_tcp)) > 0.0


def test_depth_moves_the_pad_along_the_surface_normal():
    frame = TcpFrame.from_parameters(radial_offset=0.0762, tilt_deg=7.5)
    (x0, y0, z0), q0 = frame.pad_pose(0.0)
    (x1, y1, z1), q1 = frame.pad_pose(0.002)
    assert (x1, y1) == pytest.approx((x0, y0), abs=1e-12)
    assert z1 - z0 == pytest.approx(-0.002)
    assert q1 == pytest.approx(q0, abs=1e-12)  # travel is pure translation, pose unchanged


def test_azimuth_rotates_which_side_of_the_rim_touches():
    front = TcpFrame.from_parameters(radial_offset=0.0762, tilt_deg=7.5, azimuth_deg=0.0)
    back = TcpFrame.from_parameters(radial_offset=0.0762, tilt_deg=7.5, azimuth_deg=180.0)
    (xf, _, _), _ = front.pad_pose(0.0)
    (xb, _, _), _ = back.pad_pose(0.0)
    assert xf == pytest.approx(-xb, abs=1e-9)


@real_tcps
def test_real_nominal_tcp_is_centred_and_untilted():
    frame = TcpFrame.from_csv_pair(NOMINAL, NOMINAL)
    assert frame.radial_offset == pytest.approx(0.0, abs=1e-9)
    assert frame.tilt_deg == pytest.approx(0.0, abs=1e-9)


@real_tcps
def test_real_tilted_tcp_sits_near_the_rim():
    """TCP_7in_2.5deg_0000_Disk: 76.2 mm out (3 inches) on an 88.9 mm disc, and 7.5 deg tilted.

    Note the filename says 2.5 deg while the file encodes 7.5 — flagged to the user, not fixed
    here, since the settings repo is the source of truth. Decoded with the correct xyzw
    quaternion order; scalar-first parsing gives a different, plausible-looking answer.
    """
    frame = TcpFrame.from_csv_pair(NOMINAL, TILTED)
    assert frame.radial_offset == pytest.approx(0.0762, abs=1e-4)
    assert frame.axial_offset == pytest.approx(0.0225, abs=1e-4)
    assert frame.tilt_deg == pytest.approx(7.5, abs=0.01)
    assert frame.azimuth_deg == pytest.approx(180.0, abs=0.1)


@real_tcps
def test_the_other_encoded_tcp_leans_the_opposite_way():
    """0600 is the 6 o'clock clock-position: same 76.2 mm offset, opposite side, 2.5 deg."""
    frame = TcpFrame.from_csv_pair(NOMINAL, TCP_DIR / "TCP_7in_2.5deg_0600_Disk.csv")
    assert frame.radial_offset == pytest.approx(0.0762, abs=1e-4)
    assert frame.tilt_deg == pytest.approx(2.5, abs=0.01)
    assert frame.azimuth_deg == pytest.approx(0.0, abs=0.1)


@real_tcps
def test_nominal_tcp_translation_is_the_disc_face_centre():
    """The sphere model puts all 21 pad spheres at flange x = -0.3525; the TCP sits there too."""
    flange_to_tcp = read_tcp_csv(NOMINAL)
    assert flange_to_tcp.translation == pytest.approx([-0.3525, 0.0, 0.2625], abs=1e-6)
    # TCP +Z maps onto flange -X, the direction the abrasive face points.
    assert flange_to_tcp.rotation[:, 2] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-6)


def test_waypoint_frame_matches_the_planners_worked_example():
    """Flat panel, direction_vec = +Y: the planner's README says vx comes out as world +X."""
    frame = waypoint_frame(normal=[0.0, 0.0, 1.0], direction=[0.0, 1.0, 0.0])
    assert frame.rotation[:, 0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert frame.rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)  # vz = -normal
    assert travel_direction(frame) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_waypoint_frame_is_right_handed_and_into_the_part():
    for direction in ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.3, -0.7, 0.2]):
        frame = waypoint_frame(normal=[0.0, 0.0, 1.0], direction=direction)
        vx, vy, vz = frame.rotation.T
        assert np.cross(vx, vy) == pytest.approx(vz, abs=1e-12)
        assert float(np.dot(vz, [0.0, 0.0, 1.0])) == pytest.approx(-1.0, abs=1e-12)
        # vx is the side axis: perpendicular to travel, in the surface plane.
        assert float(np.dot(vx, travel_direction(frame))) == pytest.approx(0.0, abs=1e-12)


def test_waypoint_frame_rejects_a_direction_along_the_normal():
    with pytest.raises(ValueError, match="parallel"):
        waypoint_frame(normal=[0.0, 0.0, 1.0], direction=[0.0, 0.0, 1.0])


@real_tcps
def test_real_tcp_reading_rejects_a_malformed_file(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("1, 2, 3")
    with pytest.raises(ValueError, match="expected 7 values"):
        read_tcp_csv(bad)


@real_tcps
def test_spin_keeps_the_lean_angle_and_offset_but_moves_where_it_points():
    """Rotating about the tool axis is the cell's own Rz(theta) between clock positions.

    The disc is axisymmetric, so this has to rotate the TCP frame (post-multiply). Rotating the
    disc about its own centre instead is a no-op: same lean, same contact, nothing moved.
    """
    zero = TcpFrame.from_csv_pair(NOMINAL, TILTED)
    spun = zero.spun(180.0)
    assert spun.tilt_deg == pytest.approx(zero.tilt_deg, abs=1e-9)
    assert spun.radial_offset == pytest.approx(zero.radial_offset, abs=1e-9)
    # Placed at the same TCP pose, the disc now hangs off the opposite side.
    (x0, y0, _), _ = zero.pad_pose(0.0)
    (x1, y1, _), _ = spun.pad_pose(0.0)
    assert (x1, y1) == pytest.approx((-x0, -y0), abs=1e-6)


@real_tcps
def test_spun_frame_still_touches_at_its_own_tcp():
    """Spinning must carry the tilt axis with the offset, or the disc leans off its own TCP."""
    for angle in (0.0, 45.0, 90.0, 180.0, 270.0):
        frame = TcpFrame.from_csv_pair(NOMINAL, TILTED).spun(angle)
        world = rim_points(frame)
        lowest = world[np.argmin(world[:, 2])]
        assert math.hypot(lowest[0], lowest[1]) < 0.02


@real_tcps
def test_ninety_degrees_swaps_lead_tilt_for_side_tilt():
    """With travel along +x, a 90 deg spin swings the lean from along the path to across it."""
    lead = TcpFrame.from_csv_pair(NOMINAL, TILTED)          # leans along x
    side = lead.spun(90.0)                                   # leans along y
    lead_low = rim_points(lead)[np.argmin(rim_points(lead)[:, 2])]
    side_low = rim_points(side)[np.argmin(rim_points(side)[:, 2])]
    assert abs(lead_low[0]) > 10.0 * abs(lead_low[1])
    assert abs(side_low[1]) > 10.0 * abs(side_low[0])


def test_spin_is_a_no_op_for_a_centred_frame():
    frame = TcpFrame.from_parameters()
    assert frame.spun(90.0).radial_offset == pytest.approx(0.0, abs=1e-12)
    assert frame.spun(90.0).tilt_deg == pytest.approx(0.0, abs=1e-9)
