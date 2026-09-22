"""Step 1 of the fidelity gate: press the pad onto the sheet and read the contact patch.

Quasi-static by construction. For each target force the analytic Winkler solution says where the
pad must sit; the pad is placed there and Newton's collision pipeline is asked for the contact
surface directly (``pipeline.collide``), with no solver, no dynamics and no controller in the
loop. That isolates the contact model, which is the only thing the gate is about.

    source env.sh
    python scripts/grind/press_flat.py --out runs/gate                       # flat, hard pad
    python scripts/grind/press_flat.py --tilt-deg 5 --pad-modulus 20e6 --out runs/gate_tilt

Then: python scripts/grind/gate_report.py <run dir>
"""

from __future__ import annotations

import argparse
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--forces", type=float, nargs="+", default=[10.0, 25.0, 30.0],
                    help="target normal forces [N] (sanding-wm envelope by default)")
parser.add_argument("--tilt-deg", type=float, default=0.0)
parser.add_argument("--pad-modulus", type=float, default=20.0e6,
                    help="pad Young's modulus [Pa]; TODO_MEASURE, swept 20-500 MPa in the prompt")
parser.add_argument("--pad-thickness", type=float, default=0.0127)
parser.add_argument("--sheet-modulus", type=float, default=200.0e9, help="A36 carbon steel")
parser.add_argument("--sheet-poisson", type=float, default=0.26)
parser.add_argument("--sheet-thickness", type=float, default=0.003)
parser.add_argument("--sheet-side", type=float, default=0.220,
                    help="square sheet side [m]; keep it just larger than the pad, since the "
                         "sheet SDF costs memory in proportion to its area")
parser.add_argument("--voxel", type=float, default=None,
                    help="SDF target voxel size [m]; default: half the analytic penetration")
parser.add_argument("--sheet-voxel", type=float, default=None,
                    help="SDF voxel for the sheet [m]; default 4x the pad voxel, min 1 mm. The "
                         "sheet face is planar, so a fine SDF there costs memory and buys nothing")
parser.add_argument("--narrow-band", type=float, default=0.002)
parser.add_argument("--init-timeout-s", type=float, default=600.0,
                    help="how long to wait for Newton to bake the SDFs and initialize")
parser.add_argument("--memory-budget-gb", type=float, default=8.0,
                    help="clamp the pad voxel so its SDF fits this much GPU memory")
parser.add_argument("--sections", type=int, default=256, help="pad revolution segments")
parser.add_argument("--gui", action="store_true", help="show the window (physics still quasi-static)")
parser.add_argument("--record", action="store_true",
                    help="bake the presses into <out>/press.usda for replay_recording.py")
parser.add_argument("--record-steps", type=int, default=24,
                    help="intermediate poses recorded per force step, from first touch to the "
                         "target depth; 1 records only the final pose")
parser.add_argument("--record-fps", type=float, default=24.0)
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
            # Isaac segfaults in a shutdown callback; without this the crash reporter then
            # spends ~30 s trying to upload a dump after every run.
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
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

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

from grind_sim.analytic import (  # noqa: E402
    PadSpec,
    foundation_stiffness,
    k_effective,
    press_flat,
    press_tilted,
    sdf_memory_mib,
)
from grind_sim.geometry import export_usd, make_pad, make_sheet  # noqa: E402
from grind_sim.scene import PAD_BODY, PAD_PATH, SHEET_BODY, SHEET_PATH, author_shape  # noqa: E402
from grind_sim.viz import PressFrame, write_press_usd  # noqa: E402
from grind_sim.pressure import reconstruct_faces  # noqa: E402

# Prompt: cap the workpiece stiffness for conditioning, and report the error this introduces.
K_WORK_CAP_RATIO = 100.0


def affordable_voxel(budget_gb: float) -> float:
    """Smallest pad voxel whose SDF fits the budget, from the measured cost curve."""
    voxel = 1.0e-5
    while sdf_memory_mib(voxel) > budget_gb * 1024.0:
        voxel *= 1.02
    return voxel


def main() -> int:
    pad_spec = PadSpec(thickness=args.pad_thickness)
    k_pad = foundation_stiffness(args.pad_modulus, args.pad_thickness)
    # Prompt's workpiece stiffness: k = E/(h*(1-nu^2)), capped for conditioning.
    k_work_raw = args.sheet_modulus / (args.sheet_thickness * (1.0 - args.sheet_poisson**2))
    k_work = min(k_work_raw, K_WORK_CAP_RATIO * k_pad)
    k_eff = k_effective(k_pad, k_work)
    k_eff_uncapped = k_effective(k_pad, k_work_raw)
    cap_error = abs(k_eff - k_eff_uncapped) / k_eff_uncapped

    def analytic(force: float):
        if args.tilt_deg > 0.0:
            return press_tilted(pad_spec, k_eff, force, args.tilt_deg)
        return press_flat(pad_spec, k_eff, force)

    reference = analytic(max(args.forces))
    # Two voxels per penetration is the resolution rule; the memory floor usually binds first.
    floor = affordable_voxel(args.memory_budget_gb)
    voxel = args.voxel if args.voxel else max(reference.penetration_max / 2.0, floor)
    if voxel < floor:
        print(f"WARNING: {voxel*1e6:.0f} um needs ~{sdf_memory_mib(voxel)/1024:.1f} GB, over the "
              f"{args.memory_budget_gb:.0f} GB budget — expect a stall or an OOM")
    voxels_deep = reference.penetration_max / voxel
    if voxels_deep < 2.0:
        print(f"WARNING: deepest penetration {reference.penetration_max*1e6:.2f} um is only "
              f"{voxels_deep:.2f} voxels at {voxel*1e6:.1f} um — the SDF cannot resolve it")

    # The sheet SDF has to resolve the slab itself: at fewer than ~4 voxels through the
    # thickness the baked band is degenerate, and whether contact is detected at all then
    # depends on how the grid lands (measured: 1.2 mm pad voxel found 4614 faces, 1.35 mm found
    # none, 1.457 mm found 3155 — with the sheet at ~0.5 voxels thick throughout).
    sheet_voxel_cap = args.sheet_thickness / 4.0
    sheet_voxel = args.sheet_voxel if args.sheet_voxel else max(4.0 * voxel, 1.0e-3)
    if sheet_voxel > sheet_voxel_cap:
        print(f"sheet voxel {sheet_voxel*1e3:.2f} mm would leave the {args.sheet_thickness*1e3:.1f} mm "
              f"sheet {args.sheet_thickness/sheet_voxel:.2f} voxels thick; clamping to "
              f"{sheet_voxel_cap*1e3:.2f} mm")
        sheet_voxel = sheet_voxel_cap
    narrow_inner = -min(args.narrow_band, 0.4 * args.sheet_thickness)
    print(f"k_pad {k_pad:.3e}  k_work {k_work:.3e} (uncapped {k_work_raw:.3e})  "
          f"k_eff {k_eff:.3e} N/m^3  cap error {cap_error*100:.3f}%")
    print(f"pad voxel {voxel*1e6:.1f} um, sheet voxel {sheet_voxel*1e6:.1f} um, "
          f"narrow band inner {narrow_inner*1e3:.2f} mm")
    print(f"pad SDF ~{sdf_memory_mib(voxel)/1024:.2f} GB (measured curve)")

    # --- stage -------------------------------------------------------------------
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage(backend="usd")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    SimulationManager.set_default_physics_scene("/World/PhysicsScene")

    sheet_mesh = make_sheet(args.sheet_side, args.sheet_thickness, narrow_band_inner=narrow_inner)
    pad_mesh = make_pad(pad_spec, sections=args.sections)

    UsdGeom.Xform.Define(stage, SHEET_PATH)
    export_usd(sheet_mesh, stage, SHEET_BODY)
    author_shape(stage.GetPrimAtPath(SHEET_BODY), k_work, sheet_voxel, narrow_inner,
                 args.narrow_band)

    pad_xform = UsdGeom.Xform.Define(stage, PAD_PATH)
    pad_prim = pad_xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(pad_prim)
    # Mass is TODO_MEASURE; it never enters a quasi-static collide, but Newton needs one.
    UsdPhysics.MassAPI.Apply(pad_prim).CreateMassAttr(0.37)
    export_usd(pad_mesh, stage, PAD_BODY)
    author_shape(stage.GetPrimAtPath(PAD_BODY), k_pad, voxel, -args.narrow_band, args.narrow_band)
    simulation_app.update()

    if args.gui:
        # Aliased: a plain "import omni.kit.actions.core" here would rebind the module-level
        # `omni` name as a local for the whole function.
        import omni.kit.actions.core as kit_actions
        from isaacsim.core.rendering_manager import ViewportManager

        kit_actions.get_action_registry().get_action(
            "omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera"
        ).execute()
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0.25, 0.25, 0.15],
                                        target=[0.0, 0.0, 0.0])

    # --- physics config ----------------------------------------------------------
    SimulationManager.switch_physics_engine("newton")
    SimulationManager.setup_simulation(dt=1.0 / 240.0, device="cuda")
    solver_cfg = MuJoCoSolverConfig(njmax=200_000, nconmax=200_000)
    solver_cfg.cone = "elliptic"  # pyramidal goes NaN on mesh-mesh contact in 6.1.0-rc.26
    configure_newton(
        NewtonConfig(
            num_substeps=2,
            solver_cfg=solver_cfg,
            collision_cfg=CollisionConfig(
                rigid_contact_max=200_000,
                hydroelastic=HydroelasticConfig(
                    enabled=True,
                    # Reduction saturates at 50-100 contacts regardless of resolution, so it
                    # cannot carry a pressure map (sanding-wm docs/newton_issue_reduction_cap).
                    reduce_contacts=False,
                    output_contact_surface=True,
                    mc_edge_clamp_min=0.0,
                    buffer_mult_iso=2,
                ),
            ),
        )
    )

    omni.timeline.get_timeline_interface().play()
    ns = newton_ext.acquire_stage()
    print("baking SDFs...", flush=True)
    began = time.perf_counter()
    next_note = 15.0
    while not ns.initialized:
        simulation_app.update()
        elapsed = time.perf_counter() - began
        if elapsed > next_note:
            print(f"  ... still baking after {elapsed:.0f}s", flush=True)
            next_note += 15.0
        if elapsed > args.init_timeout_s:
            raise RuntimeError(
                f"Newton stage not initialized after {elapsed:.0f}s — the SDF is probably too "
                f"fine; try a larger --voxel/--sheet-voxel"
            )
    print(f"initialized in {time.perf_counter() - began:.1f}s", flush=True)

    model = ns.model
    pipeline = ns.collision_pipeline
    hydro = pipeline.narrow_phase.hydroelastic_sdf
    if hydro is None:
        raise RuntimeError("no hydroelastic pipeline — check hydroelasticEnabled on both shapes")
    shape_kh = model.shape_material_kh.numpy()
    body_q0 = model.body_q.numpy().copy()
    print(f"shapes {model.shape_count}, bodies {model.body_count}, kh {shape_kh}")

    # --- press ladder ------------------------------------------------------------
    patches = args.out / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    rows = []
    frames: list[PressFrame] = []
    half = math.radians(args.tilt_deg) / 2.0
    quat = (0.0, math.sin(half), 0.0, math.cos(half))  # (x, y, z, w), tilt about +y

    for force in args.forces:
        ref = analytic(force)
        state = model.state()
        q = body_q0.copy()
        q[0, 0:3] = (0.0, 0.0, -ref.axis_penetration)
        q[0, 3:7] = quat
        state.body_q.assign(q)

        pipeline.collide(state, ns.contacts)
        wp.synchronize()

        surface = hydro.get_contact_surface()
        n = int(surface.face_contact_count.numpy()[0])
        n = min(n, len(surface.contact_surface_depth))
        if n <= 0:
            print(f"F={force:5.1f} N: NO CONTACT at penetration "
                  f"{ref.axis_penetration*1e6:.2f} um")
            rows.append({"force_target_n": force, "n_faces": 0})
            continue

        points = surface.contact_surface_point[: 3 * n].numpy()
        depth = surface.contact_surface_depth[:n].numpy()
        pair = surface.contact_surface_shape_pair[:n].numpy()
        pressure, area, centroid, normal, pairs = reconstruct_faces(points, depth, pair,
                                                                    shape_kh, n)

        force_sim = float(np.sum(pressure * area))
        area_sim = float(np.sum(area))
        peak_sim = float(pressure.max())
        mean_sim = force_sim / area_sim if area_sim > 0 else 0.0
        # Which shape Newton put first decides how well the peak resolves: with the compliant
        # shape first the depths are its own near-zero SDF values, quantised by the voxel.
        soft_first = int(np.count_nonzero(shape_kh[pairs[:, 0]] < shape_kh[pairs[:, 1]]))

        np.savez_compressed(
            patches / f"force_{force:.0f}N.npz",
            centroid=centroid, normal=normal, depth=depth[:n], area=area, pressure=pressure,
            shape_pair=pairs, force_target_n=force, tilt_deg=args.tilt_deg,
            k_eff=k_eff, pad_modulus_pa=args.pad_modulus, voxel_m=voxel,
        )
        if args.record:
            # Walk the pad down to the pose we just measured, so the replay shows a press
            # rather than a still. Each extra pose is one collide: milliseconds.
            first_touch = ref.axis_penetration - ref.penetration_max
            for step in range(1, args.record_steps + 1):
                fraction = step / args.record_steps
                axis = first_touch + fraction * (ref.axis_penetration - first_touch)
                if step == args.record_steps:
                    step_centroid, step_pressure = centroid, pressure
                else:
                    step_state = model.state()
                    step_q = body_q0.copy()
                    step_q[0, 0:3] = (0.0, 0.0, -axis)
                    step_q[0, 3:7] = quat
                    step_state.body_q.assign(step_q)
                    pipeline.collide(step_state, ns.contacts)
                    wp.synchronize()
                    step_surface = hydro.get_contact_surface()
                    step_n = min(int(step_surface.face_contact_count.numpy()[0]),
                                 len(step_surface.contact_surface_depth))
                    if step_n <= 0:
                        step_centroid = np.zeros((0, 3))
                        step_pressure = np.zeros(0)
                    else:
                        step_pressure, _, step_centroid, _, _ = reconstruct_faces(
                            step_surface.contact_surface_point[: 3 * step_n].numpy(),
                            step_surface.contact_surface_depth[:step_n].numpy(),
                            step_surface.contact_surface_shape_pair[:step_n].numpy(),
                            shape_kh, step_n,
                        )
                frames.append(PressFrame(
                    translate=(0.0, 0.0, -axis),
                    orient_xyzw=quat,
                    centroid=step_centroid,
                    pressure=step_pressure,
                    label=f"{force:.0f} N target, tilt {args.tilt_deg:.0f} deg",
                ))
        rows.append({
            "force_target_n": force,
            "force_analytic_n": ref.force,
            "force_sim_n": force_sim,
            "force_ratio": force_sim / ref.force,
            "area_analytic_m2": ref.contact_area,
            "area_sim_m2": area_sim,
            "area_ratio": area_sim / ref.contact_area,
            "pressure_peak_analytic_pa": ref.pressure_peak,
            "pressure_peak_sim_pa": peak_sim,
            "peak_ratio": peak_sim / ref.pressure_peak,
            "pressure_mean_analytic_pa": ref.pressure_mean,
            "pressure_mean_sim_pa": mean_sim,
            "penetration_max_analytic_m": ref.penetration_max,
            "penetration_max_sim_m": float(np.abs(depth[:n]).max()),
            "axis_penetration_m": ref.axis_penetration,
            "penetration_voxels": ref.penetration_max / voxel,
            "n_faces": n,
            "soft_shape_first_faces": soft_first,
        })
        print(f"F={force:5.1f} N: sim {force_sim:8.2f} N ({force_sim/ref.force:5.2f}x), "
              f"area {area_sim*1e4:7.2f} cm2 ({area_sim/ref.contact_area:5.2f}x), "
              f"peak {peak_sim/1e3:8.2f} kPa ({peak_sim/ref.pressure_peak:5.2f}x), "
              f"{n} faces, soft-first {soft_first}")

    if args.record and frames:
        usd_path = args.out / "press.usda"
        if usd_path.exists():
            usd_path.unlink()  # USD refuses to CreateNew over an existing layer
        write_press_usd(str(usd_path), pad_mesh, sheet_mesh, frames, fps=args.record_fps)
        print(f"recorded {len(frames)} press frames to {usd_path}")

    # --- outputs -----------------------------------------------------------------
    import csv

    if rows:
        fields = sorted({k for row in rows for k in row})
        with (args.out / "timeseries.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": _git_provenance(),
        "config": {
            "forces_n": args.forces,
            "tilt_deg": args.tilt_deg,
            "pad_modulus_pa": args.pad_modulus,
            "pad_thickness_m": args.pad_thickness,
            "sheet_modulus_pa": args.sheet_modulus,
            "sheet_poisson": args.sheet_poisson,
            "sheet_thickness_m": args.sheet_thickness,
            "sheet_side_m": args.sheet_side,
            "voxel_m": voxel,
            "sheet_voxel_m": sheet_voxel,
            "narrow_band_inner_m": narrow_inner,
            "narrow_band_outer_m": args.narrow_band,
            "sections": args.sections,
        },
        "stiffness": {
            "k_pad": k_pad,
            "k_work": k_work,
            "k_work_uncapped": k_work_raw,
            "k_eff": k_eff,
            "k_eff_uncapped": k_eff_uncapped,
            "cap_error_fraction": cap_error,
        },
        "rows": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/summary.json and {len(rows)} rows")
    return 0


def _git_provenance() -> dict:
    """Record the commit and whether the tree was dirty, so a run can be traced back."""
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"],
                                        text=True).strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception as exc:  # a missing git must not lose the run
        return {"error": str(exc)}


try:
    code = main()
except BaseException:  # Isaac's shutdown (and its segfault) eats tracebacks; print ours first
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    code = 1
finally:
    simulation_app.close()
raise SystemExit(code)
