"""Measure what an SDF voxel size actually costs: bake time and GPU memory, per voxel size.

The analytic band-volume estimate in ``grind_sim.analytic.sdf_voxel_budget`` turned out to be
optimistic — Newton's sparse SDF allocates in tiles and carries per-block textures, so the real
cost is measured here rather than predicted. The gate needs this to know which voxel sizes (and
therefore which penetrations, and therefore which pad stiffnesses) are reachable at all.

    source env.sh
    python scripts/grind/voxel_ladder.py --voxels 2e-3 1e-3 5e-4 --out runs/voxel_ladder
"""

from __future__ import annotations

import argparse
import json
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
parser.add_argument("--voxels", type=float, nargs="+", default=[2.0e-3, 1.0e-3, 5.0e-4])
parser.add_argument("--timeout-s", type=float, default=240.0, help="give up on one voxel size")
parser.add_argument("--sheet-voxel", type=float, default=2.0e-3)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

experience = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.newton.kit")
simulation_app = SimulationApp(
    {
        "headless": True,
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

import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.physics.newton", True
)
import isaacsim.physics.newton as newton_ext  # noqa: E402
from isaacsim.physics.newton import (  # noqa: E402
    CollisionConfig,
    HydroelasticConfig,
    MuJoCoSolverConfig,
    NewtonConfig,
    configure_newton,
)

from grind_sim.analytic import PadSpec, sdf_voxel_budget  # noqa: E402
from grind_sim.geometry import export_usd, make_pad, make_sheet  # noqa: E402

from grind_sim.scene import PAD_BODY, PAD_PATH, SHEET_BODY, SHEET_PATH, author_shape  # noqa: E402


def gpu_mib() -> float:
    """Current GPU memory use of this process [MiB], or 0 if nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.strip().splitlines():
            pid, mem = (part.strip() for part in line.split(","))
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:
        pass
    return 0.0


def bake_once(voxel: float, pad_spec: PadSpec) -> dict:
    """Build a fresh stage at this voxel size and time how long Newton takes to initialize."""
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage(backend="usd")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    SimulationManager.set_default_physics_scene("/World/PhysicsScene")

    UsdGeom.Xform.Define(stage, SHEET_PATH)
    export_usd(make_sheet(0.300, 0.003, narrow_band_inner=-0.0012), stage, SHEET_BODY)
    author_shape(stage.GetPrimAtPath(SHEET_BODY), 1.0e11, args.sheet_voxel, -0.0012, 0.002)

    pad_prim = UsdGeom.Xform.Define(stage, PAD_PATH).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(pad_prim)
    UsdPhysics.MassAPI.Apply(pad_prim).CreateMassAttr(0.37)
    export_usd(make_pad(pad_spec), stage, PAD_BODY)
    author_shape(stage.GetPrimAtPath(PAD_BODY), 1.6e9, voxel, -0.002, 0.002)
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

    before = gpu_mib()
    began = time.perf_counter()
    omni.timeline.get_timeline_interface().play()
    ns = newton_ext.acquire_stage()
    while not ns.initialized:
        simulation_app.update()
        if time.perf_counter() - began > args.timeout_s:
            omni.timeline.get_timeline_interface().stop()
            simulation_app.update()
            return {"voxel_m": voxel, "timed_out": True,
                    "bake_s": time.perf_counter() - began, "gpu_mib": gpu_mib() - before}
    elapsed = time.perf_counter() - began
    used = gpu_mib() - before
    omni.timeline.get_timeline_interface().stop()
    simulation_app.update()
    predicted, predicted_gb = sdf_voxel_budget(pad_spec, voxel, bytes_per_voxel=4)
    return {
        "voxel_m": voxel,
        "timed_out": False,
        "bake_s": elapsed,
        "gpu_mib": used,
        "predicted_band_voxels": predicted,
        "predicted_gib": predicted_gb,
    }


def main() -> int:
    pad_spec = PadSpec()
    rows = []
    print(f"{'voxel':>10} {'bake':>9} {'GPU delta':>11} {'predicted':>11}")
    print("-" * 45)
    for voxel in sorted(args.voxels, reverse=True):
        row = bake_once(voxel, pad_spec)
        rows.append(row)
        status = "TIMEOUT" if row["timed_out"] else f"{row['bake_s']:7.1f}s"
        pred = "" if row["timed_out"] else f"{row['predicted_gib']*1024:9.0f} MiB"
        print(f"{voxel*1e6:8.0f}um {status:>9} {row['gpu_mib']:9.0f} MiB {pred:>11}", flush=True)
        if row["timed_out"]:
            print("  (stopping the ladder: finer voxels only get worse)")
            break

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "sheet_voxel_m": args.sheet_voxel,
        "timeout_s": args.timeout_s,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}/summary.json")
    return 0


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
