# hydroelastic-simulation

Contact mechanics for GrayMatter's Scan&Grind cell, simulated in **Isaac Sim 6.1** with the
**Newton** physics backend's hydroelastic contact model.

The cell's planner decides *where* the tool goes and holds a force setpoint the operator dials in.
It models no contact mechanics at all — its "contact patch" is a collision filter, and its contact
analysis is a coverage percentage. This repo supplies the missing half: given the cell's own TCP
frames and toolpath spacing, what the contact patch actually is, how pressure is distributed
across it, and what loads that puts back through the spindle.

Two results so far:

- **A tilted TCP covers 38.6% of a flat plate where the cell's own metric reports 100%**, because
  hatch pitch is computed from the disc *diameter* while a tilted disc contacts about a quarter of
  it. Spinning the same TCP 90° — lead tilt instead of side tilt — recovers it to **90.2%** at
  identical force, tilt and path. Spindle bending is unchanged either way, so the machine's own
  telemetry cannot tell the two apart.
- **Hydroelastic contact is only trustworthy above ~2 SDF voxels of penetration.** Below that it
  fails quietly: force and area stay plausible while peak pressure inflates (2.2x at a quarter
  voxel, 213x at a thousandth). A hard 7" plate pressed flat penetrates 26–650 nm — three to four
  orders short of anything a GPU can resolve.

Details: [`docs/flat_part_workflow.md`](docs/flat_part_workflow.md) and
[`docs/gate_findings.md`](docs/gate_findings.md).

## Getting started

Isaac Sim 6.1 runs on the host, not in the ROS 2 containers. Python 3.12 from `uv`; the system
3.10 cannot run Isaac Sim 6.x.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
source env.sh          # activates .venv, accepts the Omniverse EULA
pytest                 # 93 tests, CPU only, no Isaac Sim needed
```

The venv is ~25 GB and gitignored. Never move it with `mv` — its shebangs hold absolute paths;
rebuild it instead.

## The workflow

**Simulate headless, then replay.** A windowed physics run costs minutes of startup and can
diverge; replay is baked USD animation that opens in seconds and looks identical every time.

```bash
# plan a planner-shaped path over a flat plate and press at every waypoint
python scripts/grind/press_path.py --plate 0.3 0.3 --every 4 --force 25 --record \
    --out runs/path_nominal

# the cell's tilted TCP, and the same TCP spun a quarter turn (lead tilt)
python scripts/grind/press_path.py --tcp-csv <settings>/tools/tcp/TCP_7in_2.5deg_0000_Disk.csv \
    --plate 0.3 0.3 --every 4 --force 25 --record --out runs/path_rim_7p5
python scripts/grind/press_path.py --tcp-csv <settings>/tools/tcp/TCP_7in_2.5deg_0000_Disk.csv \
    --tcp-spin-deg 90 --plate 0.3 0.3 --every 4 --force 25 --record --out runs/path_rim_7p5_spun90

# coverage + overlap maps, both metrics, and the grids behind them
python scripts/grind/path_report.py runs/path_* --out runs/path_report

# watch any pass
python scripts/replay_recording.py runs/path_rim_7p5_spun90/press.usda --loop \
    --eye 0.5 0.5 0.4 --target 0 0 0
```

Before trusting a new configuration, check it is resolvable at all — seconds, no GPU:

```bash
python scripts/grind/feasibility.py --memory-budget-gb 8 --out runs/feasibility
```

## Layout

```
src/grind_sim/
  tcp.py         the cell's TCP frames + the planner's waypoint triad
  toolpath.py    planner-shaped hatch path over a flat plate
  loads.py       patch -> axial force, moment arm, spindle bending
  coverage.py    simulated coverage + a reimplementation of the cell's metric
  analytic.py    Winkler reference and the measured SDF cost curves
  geometry.py    watertight pad/sheet meshes
  pressure.py    VENDORED from sanding-wm: per-face pressure reconstruction
scripts/grind/   feasibility, voxel ladder, presses, reports
docs/            findings, figures, the originating prompt
tests/           93 CPU-only tests; no GPU, no Isaac Sim
```

`runs/`, `recordings/` and `.venv/` are gitignored, so figures the docs link to live in
`docs/figures/`.

## Conventions worth knowing before you change anything

Each of these was got wrong once, and each failure produced plausible-looking results.
`CLAUDE.md` has the full list with sources; the short version:

- **Tool Z points into the part** (`vz = -surface_normal`). The waypoint triad is
  `vx = direction x normal` (the raw outward normal) and travel is `-vy`, not `vx`.
- **TCP CSVs are `X, Y, Z, Qx, Qy, Qz, Qw`** — vector first. Scalar-first parsing yields a valid
  rotation and the wrong frame.
- **A TCP is the contact point.** A tilted TCP sits ~76 mm out toward the rim and the disc leans
  onto it; if a tilted disc touches far from its own axis, the frame is wrong. Tests assert 20 mm.
- **Every shape needs an SDF that resolves its own geometry.** A sheet thinner than one voxel makes
  contact detection flicker on and off with voxel size.

## Limits

- **Pad stiffness is unmeasured.** It is a calibration parameter (`TODO_MEASURE` in `configs/`),
  currently run foam-soft because a hard pad pressed flat is unresolvable. Absolute pressures are
  provisional; ratios between configurations are more reliable.
- **There is no pressure ground truth.** The planner models no contact mechanics, so only coverage
  geometry can be cross-checked. Nothing here has been validated against hardware.
- **Rib geometry, friction vs abrasive, hardness and Preston coefficients are all unknown** and
  left null rather than invented; the config warns at startup.
- Two planner conventions (row alternation, and a tilt-sign flip that depends on traversal
  direction) could not be settled from the source and are recorded per run in `summary.json`.
