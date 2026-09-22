"""Press the pad onto the sheet through a TCP frame, the way the cell actually commands it.

The difference from ``press_flat.py``: there the pad was tilted about its own centre and driven
along the surface normal. Here the pad hangs off a TCP — origin on the surface, axis along the
surface normal — exactly as the planner holds it. With a nominal TCP that is the same thing; with
one of the cell's tilted TCPs (origin out near the rim) it is not, and the pad pose, the patch
position and the moment about the TCP all differ.

Depth is found by bisection against the target force, because once the frame is rim-referenced
the analytic Winkler shortcut no longer gives the pose directly.

    source env.sh
    python scripts/grind/press_tcp.py --tcp-csv <settings>/tools/tcp/TCP_7in_2.5deg_0000_Disk.csv \
        --forces 25 --out runs/tcp_7in_tilted
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
                    help="the cell's TCP file to drive with; omit for a centre-referenced frame")
parser.add_argument("--tcp-nominal", type=Path, default=SETTINGS_TCP / "TCP_7in_Nominal.csv",
                    help="nominal TCP for the same tool; the pad frame is defined by this one")
parser.add_argument("--tcp-tilt-deg", type=float, default=0.0,
                    help="hand-built frame: tilt [deg] (ignored when --tcp-csv is given)")
parser.add_argument("--tcp-radial", type=float, default=0.0,
                    help="hand-built frame: radial offset [m]; 0 tilts about the pad centre")
parser.add_argument("--tcp-axial", type=float, default=0.0)
parser.add_argument("--tcp-azimuth-deg", type=float, default=0.0)
parser.add_argument("--tcp-spin-deg", type=float, default=0.0,
                    help="rotate the TCP about the tool axis [deg]: 0 leaves the lean where the "
                         "file puts it, 90 swings it a quarter turn (lead tilt <-> side tilt)")
parser.add_argument("--forces", type=float, nargs="+", default=[25.0],
                    help="force-device setpoints [N], held along the TOOL axis (not the surface "
                         "normal): that is the axis the AFD actuates")
parser.add_argument("--pad-modulus", type=float, default=100.0e3,
                    help="pad Young's modulus [Pa]; the gate showed only foam-soft pads resolve")
parser.add_argument("--pad-thickness", type=float, default=0.0127)
parser.add_argument("--sheet-modulus", type=float, default=200.0e9)
parser.add_argument("--sheet-poisson", type=float, default=0.26)
parser.add_argument("--sheet-thickness", type=float, default=0.003)
parser.add_argument("--sheet-side", type=float, default=0.260)
parser.add_argument("--voxel", type=float, default=1.0e-3)
parser.add_argument("--sheet-voxel", type=float, default=None)
parser.add_argument("--narrow-band", type=float, default=0.002)
parser.add_argument("--force-tolerance", type=float, default=0.02, help="relative")
parser.add_argument("--max-bisection", type=int, default=18)
parser.add_argument("--record", action="store_true")
parser.add_argument("--record-steps", type=int, default=24)
parser.add_argument("--record-fps", type=float, default=24.0)
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
from grind_sim.tcp import TcpFrame  # noqa: E402
from grind_sim.viz import PressFrame, write_press_usd  # noqa: E402

K_WORK_CAP_RATIO = 100.0


def build_frame() -> TcpFrame:
    """The TCP to drive with: a real calibration pair, or a hand-built frame."""
    if args.tcp_csv:
        return TcpFrame.from_csv_pair(args.tcp_nominal, args.tcp_csv)
    return TcpFrame.from_parameters(
        radial_offset=args.tcp_radial,
        axial_offset=args.tcp_axial,
        tilt_deg=args.tcp_tilt_deg,
        azimuth_deg=args.tcp_azimuth_deg,
        name=f"synthetic_{args.tcp_tilt_deg:.1f}deg_r{args.tcp_radial*1e3:.0f}mm",
    )


def main() -> int:
    frame = build_frame()
    if args.tcp_spin_deg:
        frame = frame.spun(args.tcp_spin_deg)
    pad_spec = PadSpec(thickness=args.pad_thickness)
    k_pad = foundation_stiffness(args.pad_modulus, args.pad_thickness)
    k_work_raw = args.sheet_modulus / (args.sheet_thickness * (1.0 - args.sheet_poisson**2))
    k_work = min(k_work_raw, K_WORK_CAP_RATIO * k_pad)
    k_eff = k_effective(k_pad, k_work)

    sheet_voxel = args.sheet_voxel or min(4.0 * args.voxel, args.sheet_thickness / 4.0)
    sheet_voxel = min(sheet_voxel, args.sheet_thickness / 4.0)
    narrow_inner = -min(args.narrow_band, 0.4 * args.sheet_thickness)

    print(frame.describe())
    print(f"k_pad {k_pad:.3e}  k_eff {k_eff:.3e} N/m^3  pad voxel {args.voxel*1e6:.0f} um  "
          f"sheet voxel {sheet_voxel*1e6:.0f} um")

    # --- stage -------------------------------------------------------------------
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage(backend="usd")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    SimulationManager.set_default_physics_scene("/World/PhysicsScene")

    sheet_mesh = make_sheet(args.sheet_side, args.sheet_thickness, narrow_band_inner=narrow_inner)
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
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0.3, 0.3, 0.2],
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

    def probe(depth: float) -> dict:
        """Collide at one TCP depth and reduce the patch to the loads the machine carries."""
        translate, quat = frame.pad_pose(depth)
        state = model.state()
        q = body_q0.copy()
        q[0, 0:3] = translate
        q[0, 3:7] = quat
        state.body_q.assign(q)
        pipeline.collide(state, ns.contacts)
        wp.synchronize()

        # The spindle axis is the pad's own axis through its face centre; both move with the pad,
        # so they come from the same pose the contact was computed at.
        rotation = _rotation(quat)
        tool_origin = np.asarray(translate, dtype=np.float64)
        tool_axis = rotation @ np.array([0.0, 0.0, 1.0])
        tcp_origin = np.array([0.0, 0.0, -depth])

        surface = hydro.get_contact_surface()
        n = min(int(surface.face_contact_count.numpy()[0]), len(surface.contact_surface_depth))
        if n <= 0:
            empty = reduce_loads(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                                 tool_origin=tool_origin, tool_axis=tool_axis,
                                 tcp_origin=tcp_origin)
            return {"depth": depth, "loads": empty, "points": np.zeros((0, 3)),
                    "pressure": np.zeros(0), "tool_origin": tool_origin, "tool_axis": tool_axis}

        pressure, area, centroid, normal, _ = reconstruct_faces(
            surface.contact_surface_point[: 3 * n].numpy(),
            surface.contact_surface_depth[:n].numpy(),
            surface.contact_surface_shape_pair[:n].numpy(),
            shape_kh, n,
        )
        loads = reduce_loads(centroid, normal, pressure, area, tool_origin=tool_origin,
                             tool_axis=tool_axis, tcp_origin=tcp_origin)
        return {"depth": depth, "loads": loads, "points": centroid, "pressure": pressure,
                "tool_origin": tool_origin, "tool_axis": tool_axis}

    def seek_depth(target: float) -> tuple[float, dict]:
        """Bisect TCP depth until the normal force matches the target.

        Depth zero is not "just touching": a tilted frame can already have a rim buried at zero
        travel (the cell's rim TCPs carry an axial offset), so the search first retracts until
        contact is clear, then closes in from there.
        """
        low = 0.0
        for _ in range(40):
            if probe(low)["loads"].force_axial <= 0.0:
                break
            low -= max(0.002, pad_spec.outer_radius * math.sin(math.radians(frame.tilt_deg)))
            if low < -0.3:
                raise RuntimeError("retracted 300 mm and still in contact; check the TCP frame")
        high = low + max(2.0 * target / (k_eff * pad_spec.face_area), 1.0e-4)
        for _ in range(40):
            if probe(high)["loads"].force_axial >= target:
                break
            high += max(high - low, 1.0e-4)
            if high - low > 0.1:
                raise RuntimeError("100 mm of travel without reaching the target force")
        result = probe(high)
        for _ in range(args.max_bisection):
            mid = 0.5 * (low + high)
            result = probe(mid)
            if abs(result["loads"].force_axial - target) <= args.force_tolerance * target:
                return mid, result
            if result["loads"].force_axial < target:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high), result

    patches = args.out / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    rows, frames = [], []
    for force in args.forces:
        depth, result = seek_depth(force)
        loads = result["loads"]
        np.savez_compressed(
            patches / f"force_{force:.0f}N.npz",
            centroid=result["points"], pressure=result["pressure"],
            force_target_n=force, tcp_tilt_deg=frame.tilt_deg,
            tcp_radial_m=frame.radial_offset, tcp_name=frame.name,
            pad_modulus_pa=args.pad_modulus, tilt_deg=frame.tilt_deg, voxel_m=args.voxel,
            tool_origin=result["tool_origin"], tool_axis=result["tool_axis"],
            centre_of_pressure=loads.centre_of_pressure,
        )
        rows.append({
            "force_target_n": force,
            "tcp_depth_m": depth,
            "force_axial_n": loads.force_axial,
            "force_normal_surface_n": loads.force_normal_surface,
            "force_lateral_n": loads.force_lateral,
            "moment_arm_mm": loads.moment_arm * 1e3,
            "bending_moment_tool_nm": loads.bending_moment_tool,
            "moment_about_tcp_nm": loads.bending_moment_tcp,
            "patch_area_m2": loads.patch_area,
            "pressure_peak_pa": loads.pressure_peak,
            "pressure_mean_pa": loads.pressure_mean,
            "patch_offset_from_tcp_mm": float(
                np.hypot(*(loads.centre_of_pressure[:2] - np.zeros(2)))) * 1e3,
            "n_faces": loads.n_faces,
            "tcp_tilt_deg": frame.tilt_deg,
            "tcp_radial_mm": frame.radial_offset * 1e3,
        })
        print(f"F={force:5.1f} N: travel {depth*1e3:7.3f} mm | axial {loads.force_axial:6.2f} N, "
              f"normal {loads.force_normal_surface:6.2f} N, lateral {loads.force_lateral:5.2f} N | "
              f"arm {loads.moment_arm*1e3:5.1f} mm, bending {loads.bending_moment_tool:6.3f} Nm "
              f"(TCP {loads.bending_moment_tcp:6.3f} Nm) | area {loads.patch_area*1e4:6.2f} cm2, "
              f"peak {loads.pressure_peak/1e3:7.2f} kPa", flush=True)

        if args.record:
            for step in range(1, args.record_steps + 1):
                step_depth = depth * step / args.record_steps
                step_result = result if step == args.record_steps else probe(step_depth)
                translate, quat = frame.pad_pose(step_depth)
                frames.append(PressFrame(translate, quat, step_result["points"],
                                         step_result["pressure"],
                                         label=f"{force:.0f} N, {frame.name}"))

    if args.record and frames:
        usd_path = args.out / "press.usda"
        if usd_path.exists():
            usd_path.unlink()
        write_press_usd(str(usd_path), pad_mesh, sheet_mesh, frames, fps=args.record_fps)
        print(f"recorded {len(frames)} frames to {usd_path}")

    if rows:
        with (args.out / "timeseries.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": _git(),
        "tcp": {
            "name": frame.name,
            "source_csv": str(args.tcp_csv) if args.tcp_csv else None,
            "nominal_csv": str(args.tcp_nominal) if args.tcp_csv else None,
            "spin_deg": args.tcp_spin_deg,
            "tilt_deg": frame.tilt_deg,
            "radial_offset_m": frame.radial_offset,
            "axial_offset_m": frame.axial_offset,
            "azimuth_deg": frame.azimuth_deg,
        },
        "config": {
            "pad_modulus_pa": args.pad_modulus, "pad_thickness_m": args.pad_thickness,
            "sheet_thickness_m": args.sheet_thickness, "sheet_side_m": args.sheet_side,
            "voxel_m": args.voxel, "sheet_voxel_m": sheet_voxel,
            "k_pad": k_pad, "k_eff": k_eff,
        },
        "rows": rows,
    }, indent=2))
    print(f"wrote {args.out}/summary.json")
    return 0


def _rotation(quat_xyzw) -> np.ndarray:
    """Rotation matrix from a Newton (x, y, z, w) quaternion."""
    x, y, z, w = quat_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


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
