# Fidelity gate: does Newton hydroelastic contact work for a grind pad on a flat sheet?

Measured 2026-09-21 on Isaac Sim 6.1.0-rc.26 / newton 1.5.0 / RTX 5080 (16 GB), against the
analytic Winkler reference in `src/grind_sim/analytic.py`. Raw runs are under `runs/`
(gitignored); regenerate with the commands in each section.

**Answer in one line:** hydroelastic contact reproduces the analytic patch to within about 10-20%
*when the penetration is at least ~2 SDF voxels deep*, and that condition is unreachable for a
hard 7" plate pressed on a flat sheet — it needs voxels about 1000x finer than the GPU can hold.

## 1. The master variable is penetration divided by voxel size

Every case, sorted by how well the penetration is resolved (`runs/gate_report/cases.csv`):

| case | pad E | tilt | pad voxel | depth/voxel | force | area | peak |
|---|---|---|---|---|---|---|---|
| hard, flat | 20 MPa | 0° | 500 µm | 0.001 | 1.39x | 0.99x | **213x** |
| foam, flat | 100 kPa | 0° | 500 µm | 0.26 | 1.00x | 1.00x | **2.2x** |
| hard, tilted | 20 MPa | 5° | 500 µm | 0.63 | **0.69x** | 0.84x | 0.99x |
| foam, tilted | 100 kPa | 5° | 3 mm | 0.90 | 0.86x | 0.96x | 0.81x |
| foam, tilted | 100 kPa | 5° | 2 mm | 1.35 | 0.88x | 0.98x | 0.85x |
| foam, tilted | 100 kPa | 5° | 1.46 mm | 1.86 | 0.89x | 0.98x | 0.88x |
| foam, tilted | 100 kPa | 5° | 1 mm | 2.71 | 0.90x | 0.98x | 0.88x |
| foam, tilted | 100 kPa | 5° | 700 µm | 3.86 | 0.90x | 1.02x | 0.90x |

Ratios are sim / analytic Winkler at the same prescribed pad pose. Two distinct failure modes:

- **Under 1 voxel:** the integrated force and the patch area can still look right while the **peak
  pressure inflates** — 2.2x at a quarter-voxel, 213x at a thousandth. A pressure map from an
  under-resolved patch is worthless even when `sum(P*A)` is correct.
- **Between about 0.5 and 1 voxel:** the force itself goes badly wrong (0.68-0.69x) while the peak
  looks innocent. There is no single number that tells you a run is healthy; check the depth.

This reproduces, and explains, sanding-wm's flat-workpiece finding (peak 3.5x analytic,
`sanding-wm/docs/dev/HANDOFF.md:317`). Their flat foam case sits at a fraction of a voxel, exactly
where we measure 2.2x. It is a resolution artefact, not a property of hydroelastic contact.

The contact-pair ordering flips with the relative SDF resolution of the two shapes: before the
sheet-voxel fix every face came back rigid-shape-first, afterwards the resolved cases come back
compliant-shape-first (`soft_shape_first_faces` = all of them). Upstream warns that ordering
resolves the peak poorly (1.6-5.5x inflation); here it did not inflate anything (peak 0.88x), so
the warning is worth heeding but is not what drives the numbers in this geometry.

### The sheet's own SDF has to resolve the sheet

Found the hard way, and it invalidated the first pass of this table. The sheet voxel was derived
from the pad voxel (4x), which left a 3 mm sheet **under one voxel thick** (0.47-0.63). Whether
contact was detected at all then depended on how the grid happened to land: at a 1.2 mm pad voxel
4614 faces, at 1.35 mm **none**, at 1.457 mm 3155, at 1.6 mm none again. After capping the sheet
voxel at a quarter of the sheet thickness, every voxel size finds contact, the patch resolves with
~3x more faces, and the numbers stop moving (force 0.89x +/- 0.01 across a 1.2-1.6 mm pad voxel).

The lesson generalises: **every shape in the pair needs its own SDF to resolve its own geometry**,
not just the contact. A thin workpiece is the easy one to get wrong.

## 2. A systematic ~11% force deficit remains, even when resolved

Resolved cases cluster at 0.86-0.90x on force and 0.81-0.90x on peak pressure — a consistent bias,
not scatter, and it shrinks only slowly with finer voxels (0.86x at 3 mm, 0.90x at 700 µm). Unexplained
so far. Candidates, cheapest first: marching-cubes edge handling at the crescent boundary
(`mc_edge_clamp_min`), depth sampled at face centroids rather than integrated over the face, and the
analytic model treating the pad as perfectly rigid while Newton splits the deflection between both
shapes. Worth resolving before anyone calibrates a pad modulus against these numbers, because a
20% force bias maps straight onto a 20% error in fitted stiffness.

## 3. GPU memory sets the floor, and it is not subtle

Measured with `scripts/grind/voxel_ladder.py` (7" pad, 2 mm band, float32):

| voxel | bake | GPU |
|---|---|---|
| 2 mm | 0.8 s | 397 MiB |
| 1 mm | 1.7 s | 1109 MiB |
| 500 µm | 2.5 s | 7357 MiB |
| 156 µm | — | stalled past 10 min at ~12 GB |

Cost goes as 1/voxel³, so 250 µm would want ~57 GB. **Practical floor: 500 µm, comfortable: 1 mm.**
Baking is cheap; memory is the wall. The band-volume estimate in `sdf_voxel_budget` is about 1000x
optimistic (Newton allocates in 8³ tiles with per-block textures) — use `sdf_memory_mib`, which is
fitted to the table above.

## 4. Consequence for the 3M 80514

Winkler penetration at the pad's full 244 cm² face, from `scripts/grind/feasibility.py`:

| pad | flat, 25 N | 5° tilt, 25 N |
|---|---|---|
| 500 MPa (extra-hard, as specified) | 26 nm | 80 µm |
| 20 MPa (soft end of the prompt's sweep) | 650 nm | 310 µm |
| 100 kPa (foam interface pad, for scale) | 130 µm | 2.7 mm |

Against a 500 µm floor and the 2-voxel rule (so ≥1 mm of penetration), **no hard-plate case is
reachable**: flat is short by three to four orders of magnitude, and even 5° of tilt leaves the
20 MPa pad at 0.63 voxels, where we measured the force 31% low. Only foam-like compliance with
tilt lands in the usable regime.

This is a statement about *contact compliance*, not about the tool. A hard plate pressed flat
against a hard sheet has no compliant layer, so there is no pressure field to resolve — the contact
is rigid-on-rigid and the pressure distribution is set by geometry (flatness, tilt, ribs), which is
what hydroelastic contact is least able to tell you here.

## 5. Recommendation

1. **Do not use hydroelastic contact for the hard plate on flat sheet.** Use the analytic Winkler
   model (`grind_sim.analytic`, or sanding-wm's `DiscEngine`), which is exact for this geometry,
   costs nothing, and is the engine sanding-wm already chose on the same evidence.
2. **Model the compliance that actually exists.** If the real stack has a fiber disc and especially
   an interface foam pad, that layer is the contact compliance, it is foam-soft, and it moves the
   case into the regime where hydroelastic works (2.7 mm penetration at 5°). Measure the stack's
   stiffness before building on either engine.
3. **Keep hydroelastic for what it is good at:** geometry-driven patches — ribs, tilt crescents,
   curved or scanned parts — where the patch shape is not analytically available. Budget 1 mm
   voxels, verify depth ≥ 2 voxels per run, and treat peak pressure as untrustworthy below that.
4. **Before trusting absolute pressures**, close out the 10-20% deficit in §2.

## Reproduce

```bash
source env.sh
python scripts/grind/feasibility.py --memory-budget-gb 8 --out runs/feasibility
python scripts/grind/voxel_ladder.py --voxels 4e-3 2e-3 1e-3 5e-4 --out runs/voxel_ladder
python scripts/grind/press_flat.py --pad-modulus 100e3 --tilt-deg 5 --forces 10 25 30 \
    --out runs/gate_foam_tilt5
python scripts/grind/press_flat.py --pad-modulus 20e6 --tilt-deg 0 --forces 25 --voxel 5e-4 \
    --out runs/gate_hard_flat
python scripts/grind/gate_report.py runs/gate_* runs/conv_* --out runs/gate_report
```

## Gaps this gate did not touch

Ribs (deferred from v1), the force controller and touchdown transients, spin and traverse,
spring-mounted sheet bending, multi-material comparison, and Preston removal. The unmeasured
inputs — rib geometry, pad modulus, friction vs abrasive, hardness, Preston k, pad mass — are
null in `configs/` and warn at startup rather than carrying invented defaults.
