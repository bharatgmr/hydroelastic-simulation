"""Walk a planner-shaped toolpath over a flat plate, pressing at every waypoint.

One Isaac session, one stage, then for each waypoint: place the disc through the chosen TCP at that
waypoint's frame, find the travel that holds the pass force, and record the contact patch and the
loads the machine would carry. Output feeds ``path_report.py``.

    source env.sh
    python scripts/grind/press_path.py --tcp-csv <settings>/tools/tcp/TCP_7in_2.5deg_0000_Disk.csv \
        --force 25 --every 4 --out runs/path_rim
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import isaacsim  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402

SETTINGS_TCP = Path("/home/gmr/dev-docker-CAT-GRIND_v7/ss/src/settings/tools/tcp")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tcp-csv", type=Path, default=None,
                    help="tilted TCP to drive with; omit for the nominal (centred) frame")
parser.add_argument("--tcp-nominal", type=Path, default=SETTINGS_TCP / "TCP_7in_Nominal.csv")
parser.add_argument("--tcp-spin-deg", type=float, default=0.0,
                    help="rotate the TCP about the tool axis [deg]: 0 leaves the lean where the "
                         "file puts it, 90 swings it a quarter turn (lead tilt <-> side tilt)")
parser.add_argument("--force", type=float, default=25.0,
                    help="pass force setpoint [N], held along the tool axis")
parser.add_argument("--plate", type=float, nargs=2, default=[0.300, 0.300],
                    help="plate size [m]")
parser.add_argument("--overlap-percent", type=float, default=50.0)
parser.add_argument("--pathgap", type=float, default=0.0075, help="S2 pathgap_sanding_line [m]")
parser.add_argument("--tool-contact-width", type=float, default=0.1778, help="disc diameter [m]")
parser.add_argument("--hatch-angle-deg", type=float, default=0.0)
parser.add_argument("--every", type=int, default=4,
                    help="press every Nth waypoint; the path is still planned in full")
parser.add_argument("--pad-modulus", type=float, default=100.0e3,
                    help="pad Young's modulus [Pa]; foam-soft, since the gate showed a hard pad "
                         "pressed flat cannot be resolved")
parser.add_argument("--pad-thickness", type=float, default=0.0127)
parser.add_argument("--sheet-modulus", type=float, default=200.0e9)
parser.add_argument("--sheet-poisson", type=float, default=0.26)
parser.add_argument("--sheet-thickness", type=float, default=0.003)
parser.add_argument("--voxel", type=float, default=1.0e-3)
parser.add_argument("--narrow-band", type=float, default=0.002)
parser.add_argument("--force-tolerance", type=float, default=0.03)
parser.add_argument("--max-bisection", type=int, default=14)
parser.add_argument("--record", action="store_true", help="bake the walk for replay")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

experience = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.newton.kit")
simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "extra_args": [
            "--/exts/isaacsim.core.throttling/enable_async=false",
            "--/app/asyncRendering=false",
            "--/app/asyncRenderingLowLatency=false",
            "--/isaac/startup/ros_bridge_extension=",
            "--/crashreporter/enabled=false",
        ],
    },
    experience=experience,
)

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import warp as wp  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.physics.newton", True
)
# Unused here, and its timeline-stop callback segfaults in _clear_overlay — which is what has
# been crashing every run at shutdown, and kills a run outright when the stage is recreated.
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot_setup.virtual_gantry", False
)
import isaacsim.physics.newton as newton_ext  # noqa: E402
from isaacsim.physics.newton import (  # noqa: E402
    CollisionConfig,
    HydroelasticConfig,
    MuJoCoSolverConfig,
    NewtonConfig,
    configure_newton,
)

from grind_sim.analytic import PadSpec, foundation_stiffness, k_effective  # noqa: E402
from grind_sim.geometry import export_usd, make_pad, make_sheet  # noqa: E402
from grind_sim.loads import reduce_loads  # noqa: E402
from grind_sim.pressure import reconstruct_faces  # noqa: E402
from grind_sim.scene import PAD_BODY, PAD_PATH, SHEET_BODY, SHEET_PATH, author_shape  # noqa: E402
from grind_sim.tcp import TcpFrame, Transform  # noqa: E402
from grind_sim.toolpath import PathParams, PlateSpec, hatch_path  # noqa: E402
from grind_sim.viz import PressFrame, write_press_usd  # noqa: E402

K_WORK_CAP_RATIO = 100.0


def rotation_of(quat_xyzw) -> np.ndarray:
    """Rotation matrix from a Newton (x, y, z, w) quaternion."""
    x, y, z, w = quat_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    frame = (TcpFrame.from_csv_pair(args.tcp_nominal, args.tcp_csv) if args.tcp_csv
             else TcpFrame.from_parameters(name="nominal"))
    if args.tcp_spin_deg:
        frame = frame.spun(args.tcp_spin_deg)
    plate = PlateSpec(size_x=args.plate[0], size_y=args.plate[1])
    params = PathParams(
        tool_contact_width=args.tool_contact_width,
        overlap_percent=args.overlap_percent,
        pathgap_sanding_line=args.pathgap,
        hatch_angle_deg=args.hatch_angle_deg,
    )
    path = hatch_path(plate, params)
    selected = path.waypoints[:: max(1, args.every)]

    pad_spec = PadSpec(thickness=args.pad_thickness)
    k_pad = foundation_stiffness(args.pad_modulus, args.pad_thickness)
    k_work_raw = args.sheet_modulus / (args.sheet_thickness * (1.0 - args.sheet_poisson**2))
    k_work = min(k_work_raw, K_WORK_CAP_RATIO * k_pad)
    k_eff = k_effective(k_pad, k_work)
    sheet_voxel = min(4.0 * args.voxel, args.sheet_thickness / 4.0)
    narrow_inner = -min(args.narrow_band, 0.4 * args.sheet_thickness)

    print(frame.describe())
    print(f"path: {len(path)} waypoints on {len({w.line_index for w in path.waypoints})} lines, "
          f"pitch {params.pitch*1e3:.1f} mm, spacing {params.pathgap_sanding_line*1e3:.1f} mm, "
          f"pressing every {args.every} -> {len(selected)} presses")
    print(f"k_eff {k_eff:.3e} N/m^3, pad voxel {args.voxel*1e6:.0f} um, "
          f"sheet voxel {sheet_voxel*1e6:.0f} um", flush=True)

    # --- stage: the plate is sized to the path, with a margin for the overhanging disc ---------
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage(backend="usd")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    SimulationManager.set_default_physics_scene("/World/PhysicsScene")

    sheet_side = max(plate.size_x, plate.size_y) + args.tool_contact_width
    sheet_mesh = make_sheet(sheet_side, args.sheet_thickness, narrow_band_inner=narrow_inner)
    pad_mesh = make_pad(pad_spec)
    UsdGeom.Xform.Define(stage, SHEET_PATH)
    export_usd(sheet_mesh, stage, SHEET_BODY)
    author_shape(stage.GetPrimAtPath(SHEET_BODY), k_work, sheet_voxel, narrow_inner,
                 args.narrow_band)
    pad_prim = UsdGeom.Xform.Define(stage, PAD_PATH).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(pad_prim)
    UsdPhysics.MassAPI.Apply(pad_prim).CreateMassAttr(0.37)
    export_usd(pad_mesh, stage, PAD_BODY)
    author_shape(stage.GetPrimAtPath(PAD_BODY), k_pad, args.voxel, -args.narrow_band,
                 args.narrow_band)
    simulation_app.update()

    if args.gui:
        import omni.kit.actions.core as kit_actions
        from isaacsim.core.rendering_manager import ViewportManager

        kit_actions.get_action_registry().get_action(
            "omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera"
        ).execute()
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0.4, 0.4, 0.3],
                                        target=[0.0, 0.0, 0.0])

    SimulationManager.switch_physics_engine("newton")
    SimulationManager.setup_simulation(dt=1.0 / 240.0, device="cuda")
    solver_cfg = MuJoCoSolverConfig(njmax=200_000, nconmax=200_000)
    solver_cfg.cone = "elliptic"
    configure_newton(
        NewtonConfig(
            num_substeps=2,
            solver_cfg=solver_cfg,
            collision_cfg=CollisionConfig(
                rigid_contact_max=200_000,
                hydroelastic=HydroelasticConfig(
                    enabled=True, reduce_contacts=False, output_contact_surface=True,
                    mc_edge_clamp_min=0.0, buffer_mult_iso=2,
                ),
            ),
        )
    )

    omni.timeline.get_timeline_interface().play()
    ns = newton_ext.acquire_stage()
    began = time.perf_counter()
    while not ns.initialized:
        simulation_app.update()
        if time.perf_counter() - began > 600.0:
            raise RuntimeError("Newton stage never initialized")
    print(f"initialized in {time.perf_counter() - began:.1f}s", flush=True)

    model = ns.model
    pipeline = ns.collision_pipeline
    hydro = pipeline.narrow_phase.hydroelastic_sdf
    shape_kh = model.shape_material_kh.numpy()
    body_q0 = model.body_q.numpy().copy()

    def probe(waypoint, depth: float) -> dict:
        """Place the disc at this waypoint and travel, and read what it touches."""
        # The waypoint frame IS the TCP pose: its vz is the tool axis (-normal), and travel into
        # the surface slides along that axis. Composing pad_pose's own flat-plate pose on top of
        # it would apply the -normal flip twice.
        tcp_in_world = Transform(waypoint.frame.rotation,
                                 waypoint.position + waypoint.frame.rotation[:, 2] * depth)
        translate, quat = frame.pad_pose_at(tcp_in_world)
        pad = Transform(rotation_of(quat), np.asarray(translate, dtype=np.float64))

        state = model.state()
        q = body_q0.copy()
        q[0, 0:3] = pad.translation
        q[0, 3:7] = quat
        state.body_q.assign(q)
        pipeline.collide(state, ns.contacts)
        wp.synchronize()

        tool_origin = pad.translation
        # reduce_loads wants the spindle axis pointing AWAY from the part, which is the pad mesh's
        # +z: the mesh is built face-at-z=0 with its body behind the face.
        tool_axis = pad.rotation @ np.array([0.0, 0.0, 1.0])
        tcp_origin = tcp_in_world.translation

        surface = hydro.get_contact_surface()
        n = min(int(surface.face_contact_count.numpy()[0]), len(surface.contact_surface_depth))
        if n <= 0:
            empty = reduce_loads(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                                 tool_origin=tool_origin, tool_axis=tool_axis,
                                 tcp_origin=tcp_origin)
            return {"loads": empty, "points": np.zeros((0, 3)), "pressure": np.zeros(0),
                    "area": np.zeros(0), "pad": pad, "depth": depth}

        pressure, area, centroid, normal, _ = reconstruct_faces(
            surface.contact_surface_point[: 3 * n].numpy(),
            surface.contact_surface_depth[:n].numpy(),
            surface.contact_surface_shape_pair[:n].numpy(),
            shape_kh, n,
        )
        loads = reduce_loads(centroid, normal, pressure, area, tool_origin=tool_origin,
                             tool_axis=tool_axis, tcp_origin=tcp_origin)
        return {"loads": loads, "points": centroid, "pressure": pressure, "area": area,
                "pad": pad, "depth": depth}

    def seek(waypoint, target: float) -> dict:
        """Bisect travel until the axial (force-device) load matches the setpoint."""
        low = 0.0
        for _ in range(40):
            if probe(waypoint, low)["loads"].force_axial <= 0.0:
                break
            low -= max(0.002, pad_spec.outer_radius * math.sin(math.radians(frame.tilt_deg)))
            if low < -0.3:
                raise RuntimeError("retracted 300 mm and still in contact")
        high = low + max(2.0 * target / (k_eff * pad_spec.face_area), 1.0e-4)
        for _ in range(40):
            if probe(waypoint, high)["loads"].force_axial >= target:
                break
            high += max(high - low, 1.0e-4)
            if high - low > 0.1:
                raise RuntimeError("100 mm of travel without reaching the setpoint")
        result = probe(waypoint, high)
        for _ in range(args.max_bisection):
            mid = 0.5 * (low + high)
            result = probe(waypoint, mid)
            if abs(result["loads"].force_axial - target) <= args.force_tolerance * target:
                return result
            if result["loads"].force_axial < target:
                low = mid
            else:
                high = mid
        return result

    patches_dir = args.out / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    rows, frames_out = [], []
    began = time.perf_counter()
    for i, waypoint in enumerate(selected):
        result = seek(waypoint, args.force)
        loads = result["loads"]
        np.savez_compressed(
            patches_dir / f"wp_{i:04d}.npz",
            centroid=result["points"], pressure=result["pressure"], area=result["area"],
            position=waypoint.position, travel=waypoint.travel, depth=result["depth"],
        )
        rows.append({
            "waypoint": i,
            "line_index": waypoint.line_index,
            "x_m": float(waypoint.position[0]),
            "y_m": float(waypoint.position[1]),
            "travel_x": float(waypoint.travel[0]),
            "travel_y": float(waypoint.travel[1]),
            "tcp_depth_m": float(result["depth"]),
            "force_axial_n": loads.force_axial,
            "force_normal_surface_n": loads.force_normal_surface,
            "force_lateral_n": loads.force_lateral,
            "moment_arm_mm": loads.moment_arm * 1e3,
            "bending_moment_tool_nm": loads.bending_moment_tool,
            "moment_about_tcp_nm": loads.bending_moment_tcp,
            "patch_area_m2": loads.patch_area,
            "pressure_peak_pa": loads.pressure_peak,
            "pressure_mean_pa": loads.pressure_mean,
            "cop_x_m": float(loads.centre_of_pressure[0]),
            "cop_y_m": float(loads.centre_of_pressure[1]),
            "n_faces": loads.n_faces,
        })
        if args.record:
            pad = result["pad"]
            frames_out.append(PressFrame(tuple(pad.translation), pad.quaternion_xyzw,
                                         result["points"], result["pressure"],
                                         label=f"wp {i}"))
        if i % 10 == 0 or i == len(selected) - 1:
            rate = (time.perf_counter() - began) / (i + 1)
            print(f"  [{i+1:3d}/{len(selected)}] ({waypoint.position[0]*1e3:+7.1f},"
                  f"{waypoint.position[1]*1e3:+7.1f}) mm  axial {loads.force_axial:6.2f} N  "
                  f"area {loads.patch_area*1e4:6.2f} cm2  peak {loads.pressure_peak/1e3:6.2f} kPa  "
                  f"arm {loads.moment_arm*1e3:5.1f} mm  ({rate:.2f} s/wp)", flush=True)

    if args.record and frames_out:
        usd_path = args.out / "press.usda"
        if usd_path.exists():
            usd_path.unlink()
        write_press_usd(str(usd_path), pad_mesh, sheet_mesh, frames_out, fps=12.0)
        print(f"recorded {len(frames_out)} frames to {usd_path}")

    with (args.out / "waypoints.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": _git(),
        "tcp": {"name": frame.name, "source_csv": str(args.tcp_csv) if args.tcp_csv else None,
                "spin_deg": args.tcp_spin_deg,
                "tilt_deg": frame.tilt_deg, "radial_offset_m": frame.radial_offset,
                "axial_offset_m": frame.axial_offset, "azimuth_deg": frame.azimuth_deg},
        "path": {"plate_x_m": plate.size_x, "plate_y_m": plate.size_y,
                 "tool_contact_width_m": params.tool_contact_width,
                 "overlap_percent": params.overlap_percent, "pitch_m": params.pitch,
                 "pathgap_m": params.pathgap_sanding_line,
                 "hatch_angle_deg": params.hatch_angle_deg,
                 "waypoints_planned": len(path), "waypoints_pressed": len(selected),
                 "every": args.every, "path_length_m": path.path_length,
                 "swept_area_m2": path.swept_area(), "conventions": path.conventions},
        "config": {"force_setpoint_n": args.force, "pad_modulus_pa": args.pad_modulus,
                   "voxel_m": args.voxel, "sheet_voxel_m": sheet_voxel, "k_eff": k_eff},
        "rows": rows,
    }, indent=2))
    print(f"wrote {args.out}/waypoints.csv and summary.json")
    return 0


def _git() -> dict:
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"],
                                        text=True).strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception as exc:
        return {"error": str(exc)}


try:
    code = main()
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    code = 1
finally:
    simulation_app.close()
raise SystemExit(code)
