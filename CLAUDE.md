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

```bash
source env.sh
python scripts/nut_bolt_hydroelastic.py --headless --steps 2400   # prints nut z / quat every sim-second
isaacsim                                                          # full GUI editor
```
The first launch compiles shaders (5–10 min). Assets stream from NVIDIA's S3 asset root.

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
