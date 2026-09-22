# CLAUDE.md

Guidance for Claude Code working in this folder.

## What this is

A standalone sandbox for **NVIDIA Isaac Sim 6.1.0 hydroelastic contact** simulation
(Newton physics backend), with the eventual goal of simulating grind/sand tool-to-part contact
for GrayMatter's Scan&Sand/Grind cell.

It lives under `dev-docker-CAT-GRIND_v7/ss/src/` but is **not** part of the ROS 2 stack: no
Docker, no colcon, no `/etc/ss` mounts. It runs directly on the host. `ss/.gitignore` ignores
everything under `ss/src/`, so this folder is its own (local) git repo.

## Layout

```
env.sh                    source this: activates .venv + sets OMNI_KIT_ACCEPT_EULA=YES
.venv/                    Python 3.12 venv (uv), isaacsim[all,extscache]==6.1.0.0 — gitignored, ~25 GB
requirements.lock.txt     exact frozen package set; rebuild the venv from this
scripts/
  nut_bolt_hydroelastic.py  NVIDIA's nut-and-bolt walkthrough as one standalone script
  replay_recording.py       plays back a baked recording (USD animation, no physics)
  grind/                    the grind-contact fidelity gate (see docs/gate_findings.md)
    feasibility.py            step 0: can the SDF resolve this case? (no sim, seconds)
    voxel_ladder.py           measured bake time + GPU cost per voxel size
    press_flat.py             step 1: quasi-static press, reads the contact patch
    gate_report.py            step 3: scores runs against analytic, writes the verdict
    press_tcp.py              press through one of the cell's TCP frames
    press_path.py             walk a planner-shaped path over a flat plate
    press_part.py             walk the centreline of a scan's marked region
    part_report.py            per-part + comparison figures from part runs
    rebake_part_usd.py        rebuild a part run's replay USD with the WHOLE part (no Isaac)
    render_views.py           headless mp4 per part per viewpoint (needs imageio-ffmpeg)
    path_report.py            coverage maps + both coverage metrics
    plot_patch.py             pressure maps from saved patches
src/grind_sim/
  analytic.py               Winkler reference (flat + tilted) and the SDF cost curves
  geometry.py               watertight pad/sheet meshes (trimesh) + USD export
  scene.py                  shared SDF/hydroelastic prim authoring
  pressure.py               VENDORED from sanding-wm: per-face pressure reconstruction
  loads.py                  patch -> axial force, moment arm, spindle bending
  tcp.py                    the cell's TCP frames + the planner's waypoint triad
  toolpath.py               planner-shaped hatch path over a flat plate
  coverage.py               sim raster coverage + the cell's own coverage metric
  config.py                 pydantic config; warns about every unmeasured field
configs/                  grind_v1.yaml, materials.yaml (TODO_MEASURE fields are null)
tests/                    CPU-only, no Isaac Sim: pytest
recordings/, runs/        baked recordings and run outputs (gitignored)
```

## Environment

- Host: Ubuntu 22.04, glibc 2.35, NVIDIA GPU (PCI id 2c02, Blackwell), driver 595.91 (open), 123 GB RAM.
- Python 3.12 comes from `uv` (`~/.local/bin/uv`); system Python is 3.10 and cannot run Isaac Sim 6.x.
- Key packages: `isaacsim 6.1.0.0`, `newton 1.5.0`, `newton-usd-schemas 0.4.1`, `warp-lang 1.17.0`,
  `torch 2.11.0+cu130`.
- Rebuild the venv (fast; uses the uv cache in `~/.cache/uv`):
  ```bash
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.lock.txt \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  ```
- Never move `.venv` with `mv`: its activate script and entry-point shebangs hold absolute paths. Rebuild it instead.

## Running

### Grind-contact gate (current work)

```bash
python scripts/grind/feasibility.py --memory-budget-gb 8 --out runs/feasibility   # seconds, no GPU
python scripts/grind/press_flat.py --pad-modulus 100e3 --tilt-deg 5 --out runs/gate_foam_tilt5
python scripts/grind/gate_report.py runs/gate_* runs/conv_* --out runs/gate_report
pytest                                                                            # CPU only
```

Read `docs/gate_findings.md` before changing any contact parameter — it holds the measured
resolution limits and what the failure looks like on either side of them.

**The workflow is simulate headless, then replay** — never debug physics through the GUI. A windowed
physics run costs minutes of startup per attempt, can diverge, and tells you less than the printed poses.

```bash
source env.sh

# 1. simulate, print poses, bake the motion (no window)
python scripts/nut_bolt_hydroelastic.py --headless --steps 480 --record recordings/nut_bolt.usda

# 2. watch it — pure USD animation, no physics, identical every time
python scripts/replay_recording.py recordings/nut_bolt.usda --loop   # --speed 0.5 to slow down
```

Same two steps for new geometry: iterate headless until the printed numbers are right, record, then replay.
Other entry points: `--steps 0` (GUI, live physics, runs until the window closes) and `isaacsim` (full editor).

The first launch compiles shaders (5–10 min); later runs start in ~25 s, which dominates both steps above.
Assets stream from NVIDIA's S3 asset root. Put other generated files in `outputs/` (gitignored).

There is no test suite or linter here; "it works" means the script runs and the printed poses are physically sensible.

## Script structure (order matters)

Standalone Isaac Sim scripts must follow this order, as `scripts/nut_bolt_hydroelastic.py` does:
1. `from isaacsim import SimulationApp` and construct it **before** importing anything from `omni.*`,
   `pxr`, or other `isaacsim.*` modules — those only become importable once the Kit app is running.
2. Enable the `isaacsim.physics.newton` extension (`set_extension_enabled_immediate`) **before**
   `from isaacsim.physics.newton import ...`.
3. Build/author the stage (references, colliders, materials), then `simulation_app.update()`.
4. `switch_physics_engine("newton")` → `setup_simulation(dt, device)` → `configure_newton(...)`.
5. Play the timeline and step with `simulation_app.update()`; end with `simulation_app.close()`.

## Hydroelastic essentials (Isaac Sim 6.1 / Newton)

- Newton must be the active engine. The pip package ships `isaacsim/apps/isaacsim.exp.full.newton.kit`;
  pass it as `SimulationApp(..., experience=...)`, then call `SimulationManager.switch_physics_engine("newton")`.
- Per collider: apply `NewtonSDFCollisionAPI`, set `newton:hydroelasticEnabled=True`,
  `newton:hydroelasticStiffness` (default 1e10 N/m^3), `newton:sdfMaxResolution`,
  `newton:sdfNarrowBandInner/Outer`, `newton:contactMargin`, `newton:contactGap`,
  and `physics:approximation = "none"`.
- **Both** shapes in a pair need hydroelastic + SDF. Shapes must be volumetric and watertight:
  planes, heightfields and open meshes are unsupported.
- Runtime: `configure_newton(NewtonConfig(collision_cfg=CollisionConfig(hydroelastic=HydroelasticConfig(enabled=True, ...))))`
  **after** the stage is loaded and **before** Play. Config classes are in
  `.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.physics.newton/isaacsim/physics/newton/impl/collision_config.py`.
  NVIDIA's own tests are the best API reference: `.../isaacsim.physics.newton/.../tests/test_hydroelastic.py`.
- Docs: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/hydroelastic_contact.html and
  `.../6.1.0/physics/hydroelastic_contact_walkthrough.html`.

## Cell conventions this sim must match

Read these before touching `tcp.py` — each was got wrong once, and each failure looked plausible.

- **Tool Z points INTO the part**: `vz = -surface_normal`
  (`platform/src/gmr_utilities/src/eigen_tools.cpp:compute_TCP_wrt_Vector`). The taught-normal path
  records `-tool1_Z`, so `vz = +tool1_Z`. A sim frame with TCP +Z along the outward normal mirrors
  every tilted TCP.
- **The waypoint triad** is `vz = -n`, `vx = direction_vec x n` (the raw OUTWARD normal),
  `vy = vz x vx`, and **travel is `-vy`**, not `vx`. `grind_sim.tcp.waypoint_frame` implements it;
  the planner README's worked example (flat panel, `direction_vec = +Y` -> `vx = +X`) pins the signs.
- **TCP CSVs are `X, Y, Z, Qx, Qy, Qz, Qw`** — vector first
  (`trajectory_planner/src/utils.cpp:generateTCP`, `settings/tools_old/tcp/TCP.txt`). Reading them
  scalar-first yields a valid but wrong rotation.
- **A TCP sits at the contact point.** The nominal TCP translation equals the disc face centre
  (flange `x = -0.3525`, matching all 21 pad spheres in `STC1503_lza_7inch_p2p.yaml`), and a tilted
  TCP sits ~76 mm out with the disc leaning toward it. If a tilted disc touches far from its own
  TCP axis the frame is wrong — `tests/test_tcp.py` asserts within 20 mm.
- **The 4-digit filename code is a clock azimuth**: `0000` = +x, `0300` = +y, `0600` = -x,
  `0900` = -y, per `TCP.txt`'s `T = T_nom . Rz(theta) . Tx(radius) . Ry(beta)`.
- **`TCP_7in_2.5deg_0000_Disk.csv` encodes 7.5 degrees**, not the 2.5 in its name (the `copy`
  beside it has 2.5). Unresolved upstream; the sim reports what the file says.
- **Platform has no contact mechanics.** Its "contact patch" deletes colliding waypoints, its tilted
  frame check only logs, and `generate_contact_analysis` is a coverage percentage with a flat 25 mm
  `backing_pad_height` compliance allowance. Force is an operator constant per pass. There is no
  pressure ground truth to validate against — only coverage geometry.
- **Hatch pitch** is `tool_contact_width * (1 - overlap/100)` with `tool_contact_width` the disc
  DIAMETER (`calculateInferredParameters`); point spacing along a line is `pathgap_sanding_line`.

## Gotchas found on this machine (6.1.0-rc.26, RTX 5080)

- **Standalone-script hangs/stops:** pass `--/exts/isaacsim.core.throttling/enable_async=false`,
  `--/app/asyncRendering=false`, `--/app/asyncRenderingLowLatency=false` (else the main thread hangs one
  update after Play) and `--/isaac/startup/ros_bridge_extension=` (else the ROS 2 bridge's late load failure
  stops the timeline ~3 updates after Play). See `extra_args` in `scripts/nut_bolt_hydroelastic.py`.
- **Reading poses:** Newton writes simulated poses to Fabric, so `XformPrim`/USD reads return the authored
  pose. Use `omni.physics.tensors.create_simulation_view("warp", backend="newton", stage_id=-1)
  .create_rigid_body_view(path).get_transforms()` (xyz + quat xyzw).
- **NaN on mesh-mesh thread contact:** MuJoCo-Warp with the default pyramidal cone goes NaN on the first
  nut-bolt contact. Set `cone = "elliptic"` (or `solver = "cg"`). `MuJoCoSolverConfig` has no such field, but
  any attribute set on the config instance is forwarded as a kwarg to `newton.solvers.SolverMuJoCo`
  (see its `__init__` for `iterations`, `integrator`, `impratio`, …).
- **Nut slipping through threads:** at the walkthrough's `hydroelasticStiffness=1e10` the nut falls through the
  threads; `1e12` threads correctly (≈1.96 mm/turn on the 2 mm-pitch M16).
- **Viewport overlays land in captures.** The grid, origin axes and selection outline are drawn
  over the render, and the viewport extension sets them itself at startup, so `--/...` overrides on
  the command line do not survive. Turn them off through `carb.settings` once the app is up:
  `/app/viewport/grid/enabled`, `/app/viewport/outline/enabled` and
  `/persistent/app/viewport/Viewport/Viewport0/guide/{grid,axis,selection}/visible`.
- **No ffmpeg on this host** and OpenCV's wheel only writes MJPG/AVI. `imageio-ffmpeg` (pip, ships
  its own static binary) is what `render_views.py` uses for H.264.
- **Black viewport:** the stage has no lights of its own. Run the walkthrough's lighting action
  (`omni.kit.viewport.menubar.lighting` / `set_lighting_mode_camera`) or pick a mode from the viewport's
  "Stage Lights" menu.
- **A part run's recording only holds the last SDF chunk's slab.** `press_part.py` recreates the
  stage per chunk, so replaying its `press.usda` shows the disc over empty space for most of the
  pass. Run `scripts/grind/rebake_part_usd.py <run>` first — it rebuilds the file from
  `geometry.npz` + `patches/` with one slab over the whole band, on the CPU, in under a second.
- **Recordings** bake only the nut's world transform per update; the parts are referenced from NVIDIA's S3
  asset root, so replay needs the same asset access (plain `pxr`/usdview cannot resolve those https
  references — open recordings through Isaac Sim).
- **SDF memory, not bake time, is the wall:** measured on the 7" pad, 1 mm voxels cost 1.1 GB and
  500 um cost 7.4 GB (1/voxel^3); 250 um does not fit a 16 GB card. `grind_sim.analytic.sdf_memory_mib`
  is the fitted curve; the band-volume count in `sdf_voxel_budget` is ~1000x optimistic.
- **Penetration must be >= ~2 voxels or the patch lies:** under a voxel the peak pressure inflates
  (2.2x at 0.26 voxels, 213x at 0.001) while force and area still look right; between ~0.5 and 1
  voxel the force itself goes 30% low. See `docs/gate_findings.md`.
- **Crash reporter:** pass `--/crashreporter/enabled=false`, else every run spends ~30 s after the
  shutdown segfault trying to upload a dump.
- Debugging without a window is much faster: run `--headless` and inspect `isaacsim.physics.newton.acquire_stage()`
  (`.model`, `.state_0`, `.contacts`, `.collision_pipeline`). Isaac Sim segfaults on close (virtual_gantry
  extension) — harmless.
