"""Standalone nut-and-bolt hydroelastic contact demo (Isaac Sim 6.1, Newton backend).

Assembled from the "Configure Hydroelastic Contact for a Nut-and-Bolt Assembly"
walkthrough. Run:
    source env.sh
    python scripts/nut_bolt_hydroelastic.py [--headless] [--steps N]
"""

import argparse
import math
import os

import isaacsim
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=2400, help="app updates to run (0 = run until window closed)")
parser.add_argument("--record", metavar="FILE.usda", help="bake the nut motion into a USD animation for replay")
args = parser.parse_args()

# Newton-enabled app profile shipped with the pip package.
experience = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.newton.kit")
# Keep async rendering off for the whole run. With it on, isaacsim.core.throttling turns it off one
# update after Play, and on first run that toggle hangs the main thread (100% CPU, no further output).
simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "extra_args": [
            "--/exts/isaacsim.core.throttling/enable_async=false",
            "--/app/asyncRendering=false",
            "--/app/asyncRenderingLowLatency=false",
            # The ROS 2 bridge fails to load outside a ROS environment; its late failure stops the timeline.
            "--/isaac/startup/ros_bridge_extension=",
        ],
    },
    experience=experience,
)

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.physics.tensors as physics_tensors
import omni.timeline
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdPhysics, UsdShade

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.physics.newton", True)
from isaacsim.physics.newton import (
    CollisionConfig,
    HydroelasticConfig,
    MuJoCoSolverConfig,
    NewtonConfig,
    configure_newton,
)

BOLT_PATH = "/World/bolt"
NUT_PATH = "/World/nut"
COLLIDER_PATHS = (
    f"{BOLT_PATH}/factory_bolt_loose/collisions",
    f"{NUT_PATH}/factory_nut_loose/collisions",
)
BOLT_TIP_HEIGHT = 0.035
NUT_MESH_BASE_OFFSET = 0.010
NUT_ENGAGEMENT = 0.0005
NUT_YAW = math.pi / 8.0
MAX_CONTACTS = 40_000
UPDATES_PER_SECOND = 60.0  # app updates per wall second; physics runs 4 steps (1/240 s) per update


def write_recording(path: str, factory_directory: str, frames: list) -> None:
    """Bake the recorded nut poses into a standalone USD animation (no physics on replay).

    Args:
        path: Output .usd/.usda file.
        factory_directory: Asset root holding the Factory nut/bolt USD files.
        frames: One [x, y, z, qx, qy, qz, qw] world transform of the nut body per app update.
    """
    from pxr import Usd, UsdGeom

    out = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1.0)
    out.SetTimeCodesPerSecond(UPDATES_PER_SECOND)
    out.SetStartTimeCode(0)
    out.SetEndTimeCode(max(len(frames) - 1, 0))

    world = UsdGeom.Xform.Define(out, "/World")
    out.SetDefaultPrim(world.GetPrim())

    bolt = UsdGeom.Xform.Define(out, BOLT_PATH)
    bolt.GetPrim().GetReferences().AddReference(f"{factory_directory}/factory_bolt_m16.usd")

    nut = UsdGeom.Xform.Define(out, NUT_PATH)
    nut.GetPrim().GetReferences().AddReference(f"{factory_directory}/factory_nut_m16.usd")
    nut.ClearXformOpOrder()  # the referenced asset already authors translate/orient/scale ops
    translate = nut.AddTranslateOp()
    orient = nut.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)  # the asset's own op is quatd
    for frame, (x, y, z, qx, qy, qz, qw) in enumerate(frames):
        translate.Set(Gf.Vec3d(float(x), float(y), float(z)), time=frame)
        orient.Set(Gf.Quatd(float(qw), float(qx), float(qy), float(qz)), time=frame)

    # Replay is pure animation: keep the referenced rigid bodies from being simulated again.
    for body_path in (f"{BOLT_PATH}/factory_bolt_loose", f"{NUT_PATH}/factory_nut_loose"):
        body = out.OverridePrim(body_path)
        body.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
    out.OverridePrim(f"{BOLT_PATH}/root_joint").SetActive(False)
    out.OverridePrim(f"{NUT_PATH}/root_joint").SetActive(False)
    out.GetRootLayer().Save()

# --- Step 1: load assets ------------------------------------------------------
assets_root = get_assets_root_path()
if assets_root is None:
    raise RuntimeError("Could not find the Isaac Sim assets root.")
factory_dir = f"{assets_root}/Isaac/IsaacLab/Factory"

stage_utils.create_new_stage()
stage_utils.add_reference_to_stage(usd_path=f"{factory_dir}/factory_bolt_m16.usd", path=BOLT_PATH)
XformPrim(BOLT_PATH, reset_xform_op_properties=True).set_local_poses(
    translations=[[0.0, 0.0, 0.0]], orientations=[[1.0, 0.0, 0.0, 0.0]]
)
stage = stage_utils.get_current_stage(backend="usd")
bolt_body = stage.GetPrimAtPath(f"{BOLT_PATH}/factory_bolt_loose")
bolt_body.RemoveAPI(UsdPhysics.RigidBodyAPI)  # bolt is static
bolt_body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
stage.GetPrimAtPath(f"{BOLT_PATH}/root_joint").SetActive(False)

stage_utils.add_reference_to_stage(usd_path=f"{factory_dir}/factory_nut_m16.usd", path=NUT_PATH)
XformPrim(NUT_PATH, reset_xform_op_properties=True).set_local_poses(
    translations=[[0.0, 0.0, BOLT_TIP_HEIGHT - NUT_MESH_BASE_OFFSET - NUT_ENGAGEMENT]],
    orientations=[[math.cos(NUT_YAW * 0.5), 0.0, 0.0, math.sin(NUT_YAW * 0.5)]],
)
simulation_app.update()
if not args.headless:
    import omni.kit.actions.core
    from isaacsim.core.rendering_manager import ViewportManager

    # Without a light the stage renders black: follow the walkthrough and light from the camera.
    omni.kit.actions.core.get_action_registry().get_action(
        "omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera"
    ).execute()
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0.12, 0.12, 0.08], target=[0.0, 0.0, 0.02])

# --- Steps 2-3: SDF colliders with hydroelastic contact ---------------------------
for path in COLLIDER_PATHS:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Collider prim does not exist: {path}")
    if not prim.HasAPI("NewtonSDFCollisionAPI"):
        prim.ApplyAPI("NewtonSDFCollisionAPI")
    prim.GetAttribute("newton:hydroelasticEnabled").Set(True)
    # Walkthrough uses 1e10; on this setup the nut then slips through the threads (1e11 still slips ~7%).
    prim.GetAttribute("newton:hydroelasticStiffness").Set(1.0e12)
    prim.GetAttribute("newton:sdfMaxResolution").Set(128)
    prim.GetAttribute("newton:sdfNarrowBandInner").Set(-0.005)
    prim.GetAttribute("newton:sdfNarrowBandOuter").Set(0.005)
    prim.GetAttribute("newton:contactMargin").Set(0.0)
    prim.GetAttribute("newton:contactGap").Set(0.005)
    approx = prim.GetAttribute("physics:approximation")
    if not approx.IsValid():
        approx = prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
    approx.Set("none")

# --- Step 4: low-friction contact material --------------------------------------
material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/fastener")
mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
mat_api.CreateStaticFrictionAttr().Set(0.01)
mat_api.CreateDynamicFrictionAttr().Set(0.01)
mat_api.CreateRestitutionAttr().Set(0.0)
mat_prim = material.GetPrim()
if not mat_prim.HasAPI("NewtonMaterialAPI"):
    mat_prim.ApplyAPI("NewtonMaterialAPI")
mat_prim.GetAttribute("newton:torsionalFriction").Set(0.0)
mat_prim.GetAttribute("newton:rollingFriction").Set(0.0)
mat_prim.GetAttribute("newton:contactStiffness").Set(1.0e7)
mat_prim.GetAttribute("newton:contactDamping").Set(1.0e4)
for path in COLLIDER_PATHS:
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )

# --- Step 5: solver / pipeline config (must precede Play) ---------------------------
SimulationManager.switch_physics_engine("newton", verbose=True)
SimulationManager.setup_simulation(dt=1.0 / 240.0, device="cuda")
solver_cfg = MuJoCoSolverConfig(njmax=MAX_CONTACTS, nconmax=MAX_CONTACTS)
# The default pyramidal friction cone goes NaN on the first nut-bolt thread contact with MuJoCo-Warp
# (GPU, float32). The elliptic cone is stable. MuJoCoSolverConfig has no field for it, but every
# attribute on the config instance is forwarded to newton.solvers.SolverMuJoCo(**kwargs).
solver_cfg.cone = "elliptic"
configure_newton(
    NewtonConfig(
        num_substeps=2,
        solver_cfg=solver_cfg,
        collision_cfg=CollisionConfig(
            rigid_contact_max=MAX_CONTACTS,
            hydroelastic=HydroelasticConfig(enabled=True, mc_edge_clamp_min=0.0, buffer_mult_iso=2),
        ),
    )
)

# --- Run -----------------------------------------------------------------------
# Read the nut pose from Newton's physics state, not USD: Newton writes simulated poses to Fabric,
# so XformPrim/USD reads return the stale authored pose.
omni.timeline.get_timeline_interface().play()
simulation_app.update()
sim_view = physics_tensors.create_simulation_view("warp", backend="newton", stage_id=-1)
nut_body = sim_view.create_rigid_body_view(f"{NUT_PATH}/factory_nut_loose")
step = 1
recorded = []
while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
    simulation_app.update()
    step += 1
    transform = nut_body.get_transforms().numpy()[0]
    if args.record:
        recorded.append(transform.copy())
    if step % 60 == 0:
        x, y, z, qx, qy, qz, qw = transform
        yaw_deg = (math.degrees(2.0 * math.atan2(qz, qw)) + 180.0) % 360.0 - 180.0
        print(f"[t={SimulationManager.get_simulation_time():.2f}s] nut z={z:.5f} m  yaw={yaw_deg:.1f} deg  xy=({x:.5f}, {y:.5f})")

if args.record:
    write_recording(args.record, factory_dir, recorded)
    print(f"wrote {len(recorded)} frames to {args.record}")

simulation_app.close()
