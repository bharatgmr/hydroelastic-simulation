# Claude Code prompt: Hydroelastic grinding-contact simulation in Isaac Sim 6.1 (Newton)
### Case: 3M 80514 7" ribbed extra-hard face plate on a flat metal sheet (Al / carbon steel / stainless)

> Paste everything below the line into Claude Code. Start in **Plan Mode**, and don't write code until the plan is approved.

---

## Role and goal

You are helping me build a standalone Isaac Sim 6.1 simulation of a grinding pad coming down onto a flat metal sheet and applying a controlled normal force. It uses the **Newton physics backend's hydroelastic contact** model, so I get a contact *patch* with a pressure distribution instead of a few point contacts.

This is phase 1: **contact mechanics only**. The sequence is approach, touchdown, force ramp, hold, and optionally spin plus a short traverse. The per-contact pressure data we log must be ready to feed into a Preston-style removal model later (MRR ∝ p·v). The same scene must run for different sheet metals by changing one config key.

## Step 0: Research before you code (mandatory)

Do not guess APIs. Isaac Sim 6.x changed many APIs, and Newton is new. Read these first and cite what you rely on in your plan:

1. Hydroelastic contact: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/hydroelastic_contact.html
2. Nut-and-bolt walkthrough (closest working reference): https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/hydroelastic_contact_walkthrough.html
3. Newton backend overview: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/newton_physics.html
4. `isaacsim.physics.newton` Python API: the `NewtonConfig`, `CollisionConfig`, `HydroelasticConfig`, and `configure_newton` fields, plus how to select the solver
5. Newton collisions concepts: https://newton-physics.github.io/newton/stable/concepts/collisions.html#hydroelastic-contacts
6. Newton USD schemas: https://github.com/newton-physics/newton-usd-schemas
7. Robot setup tips for Newton, Newton actuators (Python), and the contact sensor docs, to learn how to **read per-contact data** (points, normals, depth, stiffness, forces) from the Newton contact buffer
8. Isaac Sim 6.1 requirements page, to confirm support for my OS, driver, and GPU (see Environment)
9. The 3M product page for the pad: https://www.3m.com/3M/en_US/p/dc/v000074340/

After reading, grep the **installed** Isaac Sim / Newton sources and verify every class, attribute, and field name. Where the docs and the installed code disagree, the installed code wins; tell me about the mismatch. Also do a quick search (GitHub, arXiv) for existing grinding, sanding, or polishing sims using Newton, Isaac, or Drake hydroelastic contact that we could reuse. Report what you find in 5 lines or less before planning.

## The tool: 3M 80514 Disc Pad Face Plate, ribbed, 7", extra hard (red)

What 3M and distributors state about it:
- It is a **backing face plate**. It supports a same-size 7" **fiber disc**, which does the actual abrasion, and mounts on a separate **4-1/2" disc pad hub**.
- Its underside has **ribs**. The ribs deliberately *reduce contact area* so the pressure on the abrasive goes up. Rib pressure concentration is therefore a real design feature, and our sim must be able to show it.
- Red means extra hard density, intended for heavy grinding with coarse grades.
- Max speed is **8,600 RPM**. Clamp the spindle command to this.

Geometry to build (parametric, in meters internally; I think in inches and mm):
- Pad OD = 7 in = **177.8 mm**. Thickness = 0.5 in = **12.7 mm**.
- Hub-backed zone: Ø 4.5 in = **114.3 mm**. Center arbor hole: 7/8 in = 22.2 mm. The hole doesn't touch the sheet, but keep it for mass and inertia.
- **Ribs:** the product pages don't give the rib pattern. Implement a parametric rib generator with these parameters: `n_ribs`, `rib_width`, `rib_height`, `rib_pattern` (radial or concentric), `rib_inner_radius`, and `rib_outer_radius`. Mark every rib default as `TODO_MEASURE` in the config. Also provide a `flat` variant with no ribs, for A/B comparison.
- The **fiber disc** (≈0.8 mm, configurable) is folded into the pad in v1: one watertight solid, with its compliance included in the pad stiffness. Leave a flag to model it as a separate thin layer later.
- Generate the pad as a **watertight solid** with trimesh or CadQuery, and verify it is watertight. Export it to USD.

**Hub-backed vs. unsupported flex.** Hydroelastic bodies don't bend. The real plate is stiff over the 4.5" hub and more compliant toward the rim. Approximate this by building the pad as **two coaxial hydroelastic collision shapes on the same rigid body**: an inner disc (Ø114.3) with stiffness `k_pad_inner`, and an outer annulus with a softer `k_pad_outer`. Treat this as v2; v1 uses a single `k_pad`. Before building it, check that Newton allows multiple SDF shapes per body with different stiffness.

## The workpiece: flat metal sheet with selectable material

- Parametric box: default 300 × 300 mm, thickness `t_sheet` (default 3 mm; also test 1.5 mm and 6.35 mm). It is watertight by construction.
- The **narrow band inner** must be smaller than `t_sheet / 2`. Check this in code.
- Material comes from `materials.yaml`, selected with `--material aluminum_6061|carbon_steel_a36|stainless_304|...`:

```yaml
aluminum_6061:    {E: 68.9e9, nu: 0.33, rho: 2700, mu_vs_abrasive: TODO, hardness_HB: TODO, preston_k: TODO}
carbon_steel_a36: {E: 200e9,  nu: 0.26, rho: 7850, mu_vs_abrasive: TODO, hardness_HB: TODO, preston_k: TODO}
stainless_304:    {E: 193e9,  nu: 0.29, rho: 8000, mu_vs_abrasive: TODO, hardness_HB: TODO, preston_k: TODO}
```

Keep the file extensible, and don't invent values for TODO fields. Leave them null and warn at startup.

### Important physics note: how metal choice actually enters the sim

Hydroelastic stiffness combines in series: `k_eff = k_pad·k_work/(k_pad+k_work)`. The pad is thousands of times softer than any of these metals, so **k_eff ≈ k_pad, and the normal force and pressure patch will barely change between Al, carbon steel, and stainless** if metal only enters through contact stiffness. That result is physically right, not a bug. Make the metal matter through the channels where it really does:

1. **Contact stiffness (minor effect).** Set `k_work = E/(h_layer·(1−ν²))`, and cap it at `k_cap = 100·k_pad` for numerical conditioning. Report the resulting error in k_eff, which should be under 1%. Log the uncapped value too.
2. **Sheet bending compliance (major effect for thin sheet).** Mount the sheet on a prismatic joint normal to its surface, with a spring of stiffness `k_sheet` taken from plate theory. Use the flexural rigidity `D = E·t³/(12(1−ν²))`, the support condition from config (`clamped_edges | simply_supported | rigid_backing`), and the support span. Print the formula and source in the README. With `rigid_backing`, the spring is infinite and the joint is fixed.
3. **Friction.** Use `mu_vs_abrasive` per material. It drives tangential force and spindle reaction torque.
4. **Density and mass.** These only matter if the sheet is free or spring-mounted.
5. **Phase 2 hooks.** Carry `hardness_HB` and `preston_k` through to the logged metadata. Don't use them yet.

## Stiffness calibration for the pad (don't pick blindly)

3M doesn't publish a modulus for the red face plate, so treat `k_pad` (N/m³) as a **calibrated parameter**:

- Starting guess: `k_pad ≈ E_pad / h_pad`, with `h_pad` = 12.7 mm and `E_pad` swept over 20–500 MPa. Log the implied penetration at the operating force.
- Write `calibrate_pad.py`. It takes a CSV of real force-vs-displacement data (the pad pressed onto a rigid plate at 0° and at the working tilt, recorded with the robot F/T sensor) and fits `k_pad`, and `k_pad_inner`/`k_pad_outer` in v2, by running the sim and minimizing the error. Also provide a synthetic-data mode so the script is testable now.
- Sanity check: at 0° tilt with a flat pad, the full 7" face is ≈ 0.0248 m². So 50 N is only ≈ 2 kPa mean pressure, and penetration will be tiny. Real grinding uses **tilt**, which gives a small crescent contact near the rim, and the **ribs** cut the area further. Make sure the voxel size resolves the penetration at the working force. If penetration is under 2 voxels, warn.

## SDF and contact settings (Newton defaults are wrong for this scale)

- `newton:sdfTargetVoxelSize`: start at 0.25 mm on the pad and 0.5 mm on the sheet. It must be at most `rib_width / 4` when ribs are on. Set it explicitly; it overrides `sdfMaxResolution`.
- Narrow band: about ±2 mm on the pad. On the sheet, use min(±2 mm, 0.4·t_sheet).
- `newton:sdfTextureFormat`: default `float32`. Compare it with `uint16`.
- `newton:contactMargin = 0`. Tune `contactGap` only if we see tunneling at approach speed.
- Contact reduction is a flag. Turn it **off** for pressure maps and **on** for sweeps.
- Expose `mc_edge_clamp_min`, and leave it at the default unless patches look degenerate.
- **Use the MuJoCo solver.** XPBD ignores the per-contact stiffness `c`. If MuJoCo can't be used, stop and tell me.
- Call `configure_newton(...)` with `HydroelasticConfig(enabled=True, ...)` after the stage loads, before play, and again after any reload.
- Both shapes need `NewtonSDFCollisionAPI` and `newton:hydroelasticEnabled = true`.

## Tool actuation

No robot arm in v1. Mount the pad on a small articulation:
- A **prismatic joint along the pad axis**. The pad axis is tilted from the sheet normal by `tilt_deg` (default 10°; sweep 0–20°), with the tilt direction configurable. It is driven by a force/admittance controller: velocity-controlled approach, contact detection, a ramp to the setpoint, then hold.
- A **revolute spindle** (velocity-driven, clamped to 8,600 RPM).
- An optional tangential prismatic joint for the traverse (mm/s).

The controller lives in a pure-Python module with no Isaac imports. It gets the measured force and returns a command. Keep the algorithm logic and the Isaac glue cleanly separated.

## Config (single YAML plus CLI overrides)

`material`, `t_sheet`, `sheet_support`, `support_span`, `pad_variant (flat|ribbed)`, rib params, `k_pad` (or inner/outer), `tilt_deg`, `tilt_azimuth_deg`, `approach_speed`, `contact_threshold_N`, `force_setpoints_N` (list; I'll set the range from our real GrindV7 values), `ramp_s`, `hold_s`, `spindle_rpm`, `traverse_speed`, `traverse_dist`, `dt`, `substeps`, SDF params, `reduce_contacts`, `headless`.

## Outputs

Per run, write to `runs/<timestamp>_<material>_<pad_variant>_<tilt>_<force>/`:
1. `timeseries.csv` with columns t, commanded and measured normal force, tangential force, spindle reaction torque, max and mean penetration, patch area, number of contacts, sheet deflection (if spring-mounted), and tool pose.
2. `patches/*.npz` in the **sheet frame**, with per-contact points, normals, depth, area or weight, stiffness, and estimated pressure. Include metadata: material, pad variant, tilt, and force.
3. Plots: force vs. time (setpoint, **touchdown overshoot**, settle time, steady-state error), patch area vs. force, a pressure heatmap at steady state with the rib outlines overlaid, and a **per-material comparison plot** (same force and tilt; Al vs. CS vs. SS).
4. `summary.json` with overshoot, settle time, steady-state error, mean and peak pressure, patch dimensions, and k_eff (capped and uncapped).
5. A `sweep.py` that runs material × pad_variant × tilt × force and writes one combined CSV.

## Validation (required)

1. **Flat pad, 0° tilt, rigid backing.** Patch area should be close to the full 7" face (0.0248 m², minus the arbor hole), mean pressure close to F/A, and force balance within 2%.
2. **Ribbed vs. flat, same force.** The ribbed patch area should be close to the rib face area, and the mean pressure higher by roughly the area ratio.
3. **Tilted pad.** The patch should be a crescent near the rim, with the pressure centroid on the low side.
4. **Material check, rigid backing.** Al, CS, and SS should give nearly identical normal-force results; report the % difference. With a spring-mounted 1.5 mm sheet they should differ, and Al should deflect about 2.9× more than steel, following the E ratio.
5. **Convergence.** Halve the voxel size and dt. Force and patch area should change by less than 5%.
6. **Point-contact baseline** (hydroelastic off), to show why we use hydroelastic contact.

## Environment

- Ubuntu 22.04, HWE kernel 6.8, RTX 5080 (Blackwell), `nvidia-driver-595-open`, CUDA 13.2. Confirm Isaac Sim 6.1 supports this setup before installing.
- Install Isaac Sim **isolated** (pip venv or the official container). Do not touch my existing ROS 2 grind dev containers.
- Standalone `SimulationApp` scripts with `--headless`.

## Workflow

1. A plan: file layout, the verified API calls with their doc and source references, risks, and open questions (rib dimensions, pad modulus, and friction values are known unknowns). Wait for my approval.
2. Milestone 1: a flat 7" pad at 0° lands on a rigid-backed 3 mm A36 sheet and holds 50 N, with CSV output and a force plot.
3. Milestone 2: tilt, the material switch, and the spring-mounted sheet.
4. Milestone 3: ribbed pad, pressure maps, and spin/traverse.
5. Milestone 4: calibration script, sweeps, and validation report. Then a README.

Keep each milestone runnable. If anything isn't supported (for example, reading per-contact hydroelastic data from Python, or multiple SDF shapes per body), stop and tell me. Don't hack around it silently.
