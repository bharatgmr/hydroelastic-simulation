"""Run a single pass along the centre of a scan's marked region, on the real scanned surface.

Given one of the cell's ``mergedPointcloud.ply`` profiles, this takes the green-painted target
region, traces its centreline, and drives the disc along it through a chosen TCP — so the contact
lands where the cell would put it, on the surface the part actually has.

The part is meshed as a watertight slab (hydroelastic contact refuses open surfaces) and rebuilt in
chunks along the line, because one SDF covering a 600 mm weld plus the disc's reach does not fit in
GPU memory at a voxel fine enough to resolve the penetration.

    source env.sh
    python scripts/grind/press_part.py \\
        --profile /home/gmr/dev-docker-CAT-GRIND/persistent_data/profile_handler/data/profiles/weld_t1_eigen \\
        --tcp-csv <settings>/tools/tcp/TCP_7in_2.5deg_0000_Disk.csv --tcp-spin-deg 90 \\
        --force 25 --out runs/part_weld_t1
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
parser.add_argument("--profile", type=Path, required=True,
                    help="profile directory, or a mergedPointcloud.ply directly")
parser.add_argument("--colour", type=int, nargs=3, default=[0, 255, 0],
                    help="RGB of the marked target region")
parser.add_argument("--tcp-csv", type=Path, default=None,
                    help="tilted TCP to drive with; omit for the nominal (centred) frame")
parser.add_argument("--tcp-nominal", type=Path, default=SETTINGS_TCP / "TCP_7in_Nominal.csv")
parser.add_argument("--tcp-spin-deg", type=float, default=90.0,
                    help="rotate the TCP about the tool axis [deg]; 90 puts the lean along travel, "
                         "so the leading edge of the disc follows the line")
parser.add_argument("--force", type=float, default=25.0, help="setpoint along the tool axis [N]")
parser.add_argument("--spacing", type=float, default=0.0075, help="waypoint spacing [m]")
parser.add_argument("--every", type=int, default=1, help="press every Nth waypoint")
parser.add_argument("--chunk-length", type=float, default=0.15,
                    help="length of line covered by one SDF bake [m]")
parser.add_argument("--reach", type=float, default=0.10,
                    help="how far past the line the part mesh extends [m]; the disc overhangs")
parser.add_argument("--pad-modulus", type=float, default=100.0e3)
parser.add_argument("--pad-thickness", type=float, default=0.0127)
parser.add_argument("--part-modulus", type=float, default=200.0e9)
parser.add_argument("--part-poisson", type=float, default=0.26)
parser.add_argument("--slab-thickness", type=float, default=0.02)
parser.add_argument("--slab-cell", type=float, default=0.002, help="surface resample spacing [m]")
parser.add_argument("--voxel", type=float, default=1.0e-3, help="pad SDF voxel [m]")
parser.add_argument("--part-voxel", type=float, default=None,
                    help="part SDF voxel [m]; defaults to the pad voxel")
parser.add_argument("--narrow-band", type=float, default=0.002)
parser.add_argument("--force-tolerance", type=float, default=0.03)
parser.add_argument("--max-bisection", type=int, default=14)
parser.add_argument("--record", action="store_true")
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
from grind_sim.geometry import export_usd, make_pad  # noqa: E402
from grind_sim.loads import reduce_loads  # noqa: E402
from grind_sim.partmesh import (  # noqa: E402
    LocalFrame,
    band_centreline,
    load_cloud,
    resample_line,
    slab_mesh,
    surface_normals,
)
from grind_sim.pressure import reconstruct_faces  # noqa: E402
from grind_sim.scene import PAD_BODY, PAD_PATH, SHEET_BODY, SHEET_PATH, author_shape  # noqa: E402
from grind_sim.tcp import TcpFrame, Transform, waypoint_frame  # noqa: E402
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
    profile = args.profile
    ply = profile if profile.suffix == ".ply" else profile / "mergedPointcloud.ply"
    if not ply.exists():
        raise SystemExit(f"no cloud at {ply}")

    # --- the part, in its own frame --------------------------------------------------------
    # Everything downstream runs in the marked region's local frame: the sim's "world" is the
    # part. Keeps the geometry near the origin instead of out at the cell's metre-scale coords.
    cloud = load_cloud(ply)
    marked = cloud.mask_colour(tuple(args.colour))
    if marked.sum() < 50:
        raise SystemExit(f"only {marked.sum()} points painted {tuple(args.colour)}")
    frame = LocalFrame.fit(cloud.points[marked], cloud.normals[marked])
    local_marked = frame.to_local(cloud.points[marked])
    local_all = frame.to_local(cloud.points)

    line = band_centreline(local_marked)
    positions, tangents = resample_line(line, args.spacing)
    window = (
        (local_all[:, 0] > local_marked[:, 0].min() - args.reach - 0.05)
        & (local_all[:, 0] < local_marked[:, 0].max() + args.reach + 0.05)
        & (local_all[:, 1] > local_marked[:, 1].min() - args.reach - 0.05)
        & (local_all[:, 1] < local_marked[:, 1].max() + args.reach + 0.05)
        & (np.abs(local_all[:, 2] - np.median(local_marked[:, 2])) < 0.2)
    )
    neighbourhood = local_all[window]
    normals = surface_normals(positions, neighbourhood)

    selected = list(range(0, len(positions), max(1, args.every)))
    band_width = float(np.percentile(local_marked[:, 1], 90) - np.percentile(local_marked[:, 1], 10))
    print(f"{profile.name}: {len(cloud)} points, {marked.sum()} marked")
    print(f"  band {(local_marked[:,0].max()-local_marked[:,0].min())*1e3:.0f} x "
          f"{band_width*1e3:.0f} mm, centreline {np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()*1e3:.0f} mm, "
          f"{len(selected)} of {len(positions)} waypoints pressed")

    tcp = (TcpFrame.from_csv_pair(args.tcp_nominal, args.tcp_csv) if args.tcp_csv
           else TcpFrame.from_parameters(name="nominal"))
    if args.tcp_spin_deg:
        tcp = tcp.spun(args.tcp_spin_deg)
    print(f"  {tcp.describe()}")

    pad_spec = PadSpec(thickness=args.pad_thickness)
    k_pad = foundation_stiffness(args.pad_modulus, args.pad_thickness)
    k_work_raw = args.part_modulus / (args.slab_thickness * (1.0 - args.part_poisson**2))
    k_work = min(k_work_raw, K_WORK_CAP_RATIO * k_pad)
    k_eff = k_effective(k_pad, k_work)
    part_voxel = args.part_voxel or args.voxel
    narrow_inner = -min(args.narrow_band, 0.4 * args.slab_thickness)
    pad_mesh = make_pad(pad_spec)

    # --- chunks: one SDF bake each ----------------------------------------------------------
    chunk_of = {}
    for index in selected:
        chunk_of.setdefault(int(positions[index, 0] // args.chunk_length), []).append(index)
    print(f"  {len(chunk_of)} chunks of {args.chunk_length*1e3:.0f} mm\n", flush=True)

    patches_dir = args.out / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    rows, frames_out = [], []
    began_all = time.perf_counter()

    for chunk_number, (key, indices) in enumerate(sorted(chunk_of.items()), start=1):
        chunk_positions = positions[indices]
        bounds = (
            float(chunk_positions[:, 0].min() - args.reach),
            float(chunk_positions[:, 0].max() + args.reach),
            float(chunk_positions[:, 1].min() - args.reach),
            float(chunk_positions[:, 1].max() + args.reach),
        )
        inside = neighbourhood[
            (neighbourhood[:, 0] > bounds[0] - 0.02) & (neighbourhood[:, 0] < bounds[1] + 0.02)
            & (neighbourhood[:, 1] > bounds[2] - 0.02) & (neighbourhood[:, 1] < bounds[3] + 0.02)
        ]
        if len(inside) < 200:
            print(f"  chunk {chunk_number}: only {len(inside)} scan points, skipped")
            continue
        part_mesh = slab_mesh(inside, cell=args.slab_cell, thickness=args.slab_thickness,
                              bounds=bounds)

        stage_utils.create_new_stage()
        stage = stage_utils.get_current_stage(backend="usd")
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        SimulationManager.set_default_physics_scene("/World/PhysicsScene")

        UsdGeom.Xform.Define(stage, SHEET_PATH)
        export_usd(part_mesh, stage, SHEET_BODY)
        author_shape(stage.GetPrimAtPath(SHEET_BODY), k_work, part_voxel, narrow_inner,
                     args.narrow_band)
        pad_prim = UsdGeom.Xform.Define(stage, PAD_PATH).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(pad_prim)
        UsdPhysics.MassAPI.Apply(pad_prim).CreateMassAttr(0.37)
        export_usd(pad_mesh, stage, PAD_BODY)
        author_shape(stage.GetPrimAtPath(PAD_BODY), k_pad, args.voxel, -args.narrow_band,
                     args.narrow_band)
        simulation_app.update()

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
        bake = time.perf_counter() - began

        model = ns.model
        pipeline = ns.collision_pipeline
        hydro = pipeline.narrow_phase.hydroelastic_sdf
        shape_kh = model.shape_material_kh.numpy()
        body_q0 = model.body_q.numpy().copy()

        def probe(index: int, depth: float) -> dict:
            """Place the disc at this waypoint and travel, and read what it touches."""
            triad = waypoint_frame(normal=normals[index], direction=tangents[index])
            tcp_in_world = Transform(triad.rotation,
                                     positions[index] + triad.rotation[:, 2] * depth)
            translate, quat = tcp.pad_pose_at(tcp_in_world)
            pad = Transform(rotation_of(quat), np.asarray(translate, dtype=np.float64))

            state = model.state()
            q = body_q0.copy()
            q[0, 0:3] = pad.translation
            q[0, 3:7] = quat
            state.body_q.assign(q)
            pipeline.collide(state, ns.contacts)
            wp.synchronize()

            tool_axis = pad.rotation @ np.array([0.0, 0.0, 1.0])
            surface = hydro.get_contact_surface()
            n = min(int(surface.face_contact_count.numpy()[0]), len(surface.contact_surface_depth))
            if n <= 0:
                empty = reduce_loads(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                                     tool_origin=pad.translation, tool_axis=tool_axis,
                                     tcp_origin=tcp_in_world.translation,
                                     surface_normal=normals[index])
                return {"loads": empty, "points": np.zeros((0, 3)), "pressure": np.zeros(0),
                        "area": np.zeros(0), "pad": pad, "depth": depth}

            pressure, area, centroid, normal, _ = reconstruct_faces(
                surface.contact_surface_point[: 3 * n].numpy(),
                surface.contact_surface_depth[:n].numpy(),
                surface.contact_surface_shape_pair[:n].numpy(),
                shape_kh, n,
            )
            loads = reduce_loads(centroid, normal, pressure, area, tool_origin=pad.translation,
                                 tool_axis=tool_axis, tcp_origin=tcp_in_world.translation,
                                 surface_normal=normals[index])
            return {"loads": loads, "points": centroid, "pressure": pressure, "area": area,
                    "pad": pad, "depth": depth}

        def seek(index: int, target: float) -> dict:
            """Bisect travel until the axial load matches the setpoint."""
            low = 0.0
            for _ in range(40):
                if probe(index, low)["loads"].force_axial <= 0.0:
                    break
                low -= max(0.002, pad_spec.outer_radius * math.sin(math.radians(tcp.tilt_deg)))
                if low < -0.3:
                    raise RuntimeError("retracted 300 mm and still in contact")
            high = low + max(2.0 * target / (k_eff * pad_spec.face_area), 1.0e-4)
            for _ in range(40):
                if probe(index, high)["loads"].force_axial >= target:
                    break
                high += max(high - low, 1.0e-4)
                if high - low > 0.1:
                    return probe(index, high)  # never reaches the setpoint: report what it did
            result = probe(index, high)
            for _ in range(args.max_bisection):
                mid = 0.5 * (low + high)
                result = probe(index, mid)
                if abs(result["loads"].force_axial - target) <= args.force_tolerance * target:
                    return result
                if result["loads"].force_axial < target:
                    low = mid
                else:
                    high = mid
            return result

        for index in indices:
            result = seek(index, args.force)
            loads = result["loads"]
            np.savez_compressed(
                patches_dir / f"wp_{index:04d}.npz",
                centroid=result["points"], pressure=result["pressure"], area=result["area"],
                position=positions[index], normal=normals[index], tangent=tangents[index],
                depth=result["depth"],
            )
            rows.append({
                "waypoint": int(index),
                "chunk": chunk_number,
                "x_m": float(positions[index, 0]),
                "y_m": float(positions[index, 1]),
                "z_m": float(positions[index, 2]),
                "tcp_depth_m": float(result["depth"]),
                "force_axial_n": loads.force_axial,
                "force_normal_surface_n": loads.force_normal_surface,
                "force_lateral_n": loads.force_lateral,
                "moment_arm_mm": loads.moment_arm * 1e3,
                "bending_moment_tool_nm": loads.bending_moment_tool,
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
                                             label=f"wp {index}"))

        done = len(rows)
        print(f"  chunk {chunk_number}/{len(chunk_of)}: bake {bake:4.1f}s, {len(indices):3d} presses"
              f" | last: axial {rows[-1]['force_axial_n']:5.2f} N, area "
              f"{rows[-1]['patch_area_m2']*1e4:5.2f} cm2, peak "
              f"{rows[-1]['pressure_peak_pa']/1e3:6.2f} kPa | {done}/{len(selected)} total, "
              f"{time.perf_counter()-began_all:.0f}s elapsed", flush=True)

    if not rows:
        raise SystemExit("no waypoints produced contact")

    if args.record and frames_out:
        usd_path = args.out / "press.usda"
        if usd_path.exists():
            usd_path.unlink()
        # The part changes per chunk, so the recording carries the last chunk's slab as a stand-in.
        write_press_usd(str(usd_path), pad_mesh, part_mesh, frames_out, fps=12.0)
        print(f"recorded {len(frames_out)} frames to {usd_path}")

    with (args.out / "waypoints.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.out / "geometry.npz",
        frame_origin=frame.origin, frame_rotation=frame.rotation,
        centreline=positions, tangents=tangents, normals=normals,
        marked_local=local_marked.astype(np.float32),
        neighbourhood_local=neighbourhood[::4].astype(np.float32),
    )
    (args.out / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": _git(),
        "profile": str(profile),
        "marked_points": int(marked.sum()),
        "band": {"length_m": float(local_marked[:, 0].max() - local_marked[:, 0].min()),
                 "width_m": band_width},
        "tcp": {"name": tcp.name, "source_csv": str(args.tcp_csv) if args.tcp_csv else None,
                "spin_deg": args.tcp_spin_deg, "tilt_deg": tcp.tilt_deg,
                "radial_offset_m": tcp.radial_offset},
        "config": {"force_setpoint_n": args.force, "spacing_m": args.spacing,
                   "pad_modulus_pa": args.pad_modulus, "voxel_m": args.voxel,
                   "part_voxel_m": part_voxel, "slab_cell_m": args.slab_cell,
                   "slab_thickness_m": args.slab_thickness, "chunk_length_m": args.chunk_length,
                   "k_eff": k_eff},
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}/waypoints.csv, geometry.npz, summary.json "
          f"({len(rows)} waypoints, {time.perf_counter()-began_all:.0f}s)")
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
