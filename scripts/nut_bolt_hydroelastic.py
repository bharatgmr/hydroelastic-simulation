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
parser.add_argument("--steps", type=int, default=2400, help="physics steps at 240 Hz (0 = run until window closed)")
args = parser.parse_args()

# Newton-enabled app profile shipped with the pip package.
experience = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.newton.kit")
simulation_app = SimulationApp({"headless": args.headless}, experience=experience)

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.timeline
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, UsdPhysics, UsdShade

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

# --- Steps 2-3: SDF colliders with hydroelastic contact ---------------------------
for path in COLLIDER_PATHS:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Collider prim does not exist: {path}")
    if not prim.HasAPI("NewtonSDFCollisionAPI"):
        prim.ApplyAPI("NewtonSDFCollisionAPI")
    prim.GetAttribute("newton:hydroelasticEnabled").Set(True)
    prim.GetAttribute("newton:hydroelasticStiffness").Set(1.0e10)
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
configure_newton(
    NewtonConfig(
        num_substeps=2,
        solver_cfg=MuJoCoSolverConfig(njmax=MAX_CONTACTS, nconmax=MAX_CONTACTS),
        collision_cfg=CollisionConfig(
            rigid_contact_max=MAX_CONTACTS,
            hydroelastic=HydroelasticConfig(enabled=True, mc_edge_clamp_min=0.0, buffer_mult_iso=2),
        ),
    )
)

# --- Run -----------------------------------------------------------------------
nut = XformPrim(NUT_PATH)
omni.timeline.get_timeline_interface().play()
step = 0
while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
    simulation_app.update()
    step += 1
    if step % 240 == 0:
        pos, quat = nut.get_world_poses()
        print(f"[t={step / 240:.1f}s] nut z={float(pos.numpy()[0][2]):.5f} m  quat={quat.numpy()[0].round(4)}")

simulation_app.close()
