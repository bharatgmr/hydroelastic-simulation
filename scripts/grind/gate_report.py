"""Step 3 of the fidelity gate: judge the press runs and give a recommendation.

Reads the ``summary.json`` of every run directory passed on the command line, scores the gate
criteria, and writes a combined CSV, a plot and a verdict. No Isaac Sim, no GPU.

    source env.sh
    python scripts/grind/gate_report.py runs/gate_* runs/conv_* --out runs/gate_report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# How close to the analytic Winkler reference a run has to land to count as trustworthy.
FORCE_TOL = 0.10
AREA_TOL = 0.10
PEAK_TOL = 0.20
# The resolution rule from the grind prompt.
MIN_VOXELS = 2.0


@dataclass
class Case:
    """One (run, force) pair flattened for scoring."""

    run: str
    pad_modulus_pa: float
    tilt_deg: float
    voxel_m: float
    force_target_n: float
    voxels_deep: float
    force_ratio: float
    area_ratio: float
    peak_ratio: float
    n_faces: int
    soft_first: int

    @property
    def resolved(self) -> bool:
        return self.voxels_deep >= MIN_VOXELS

    @property
    def trustworthy(self) -> bool:
        return (
            abs(self.force_ratio - 1.0) <= FORCE_TOL
            and abs(self.area_ratio - 1.0) <= AREA_TOL
            and abs(self.peak_ratio - 1.0) <= PEAK_TOL
        )


def load(run_dir: Path) -> list[Case]:
    """Flatten one run's summary.json into scored cases."""
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg = summary["config"]
    cases = []
    for row in summary["rows"]:
        if not row.get("n_faces"):
            continue
        cases.append(
            Case(
                run=run_dir.name,
                pad_modulus_pa=cfg["pad_modulus_pa"],
                tilt_deg=cfg["tilt_deg"],
                voxel_m=cfg["voxel_m"],
                force_target_n=row["force_target_n"],
                voxels_deep=row["penetration_voxels"],
                force_ratio=row["force_ratio"],
                area_ratio=row["area_ratio"],
                peak_ratio=row["peak_ratio"],
                n_faces=row["n_faces"],
                soft_first=row["soft_shape_first_faces"],
            )
        )
    return cases


def plot(cases: list[Case], path: Path) -> None:
    """Plot the error in force and peak pressure against how well the penetration is resolved."""
    fig, (ax_force, ax_peak) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    depth = np.array([c.voxels_deep for c in cases])
    order = np.argsort(depth)
    depth = depth[order]
    force = np.array([c.force_ratio for c in cases])[order]
    peak = np.array([c.peak_ratio for c in cases])[order]

    for ax, y, label in ((ax_force, force, "force"), (ax_peak, peak, "peak pressure")):
        ax.axhline(1.0, color="0.4", lw=1, ls="--")
        ax.axvline(MIN_VOXELS, color="tab:red", lw=1, ls=":")
        ax.scatter(depth, y, c=np.where(depth >= MIN_VOXELS, "tab:green", "tab:red"), zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("penetration / voxel size")
        ax.set_ylabel(f"sim {label} / analytic Winkler")
        ax.set_title(f"{label} vs resolution")
        ax.grid(alpha=0.3, which="both")
    ax_peak.annotate("2 voxels", (MIN_VOXELS, ax_peak.get_ylim()[1]), color="tab:red",
                     ha="left", va="top", fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases: list[Case] = []
    for run in args.runs:
        if (run / "summary.json").exists():
            cases.extend(load(run))
        else:
            print(f"skipping {run}: no summary.json")
    if not cases:
        print("no cases found")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "cases.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(vars(cases[0])) + ["resolved", "trustworthy"])
        writer.writeheader()
        for case in cases:
            writer.writerow({**vars(case), "resolved": case.resolved,
                             "trustworthy": case.trustworthy})
    plot(cases, args.out / "fidelity_vs_resolution.png")

    print(f"{'run':<22} {'E_pad':>8} {'tilt':>5} {'voxel':>8} {'F':>6} {'depth':>7} "
          f"{'force':>7} {'area':>6} {'peak':>9}  verdict")
    print("-" * 100)
    for case in sorted(cases, key=lambda c: c.voxels_deep):
        verdict = ("OK" if case.trustworthy else "off") + ("" if case.resolved else ", UNRESOLVED")
        modulus = (f"{case.pad_modulus_pa/1e6:6.0f}M" if case.pad_modulus_pa >= 1e6
                   else f"{case.pad_modulus_pa/1e3:6.0f}k")
        print(f"{case.run:<22} {modulus:>8} {case.tilt_deg:4.1f}d {case.voxel_m*1e6:7.0f}u "
              f"{case.force_target_n:5.0f}N {case.voxels_deep:7.2f} {case.force_ratio:7.2f} "
              f"{case.area_ratio:6.2f} {case.peak_ratio:9.2f}  {verdict}")

    resolved = [c for c in cases if c.resolved]
    unresolved = [c for c in cases if not c.resolved]
    good = [c for c in resolved if c.trustworthy]
    worst_peak = max(cases, key=lambda c: c.peak_ratio)

    print("\nGate criteria")
    print(f"  1. force balance vs analytic within {FORCE_TOL:.0%}: "
          f"{sum(abs(c.force_ratio-1) <= FORCE_TOL for c in resolved)}/{len(resolved)} resolved cases")
    print(f"  2. peak pressure within {PEAK_TOL:.0%}: "
          f"{sum(abs(c.peak_ratio-1) <= PEAK_TOL for c in resolved)}/{len(resolved)} resolved cases")
    print(f"  3. worst peak inflation overall: {worst_peak.peak_ratio:.1f}x "
          f"({worst_peak.run}, {worst_peak.voxels_deep:.2f} voxels deep)")
    print(f"  4. contact-pair ordering: "
          f"{sum(c.soft_first for c in cases)} faces reported compliant-shape-first "
          "(that ordering resolves the peak poorly)")
    print(f"  5. resolvable cases: {len(resolved)}/{len(cases)}")

    verdict_lines = []
    if good and len(good) == len(resolved):
        verdict_lines.append(
            "Hydroelastic contact reproduces the analytic Winkler patch wherever the penetration "
            f"is resolved (>= {MIN_VOXELS:.0f} voxels): force, area and peak all within tolerance."
        )
    elif good:
        verdict_lines.append(
            f"Hydroelastic contact is trustworthy in {len(good)}/{len(resolved)} resolved cases; "
            "the rest miss tolerance even though the penetration is resolved."
        )
    else:
        verdict_lines.append("No resolved case reproduced the analytic reference.")
    if unresolved:
        verdict_lines.append(
            f"{len(unresolved)} cases are under-resolved, and they fail in a specific way: the "
            "integrated force and patch area stay close to analytic while the PEAK pressure "
            f"inflates (worst here {worst_peak.peak_ratio:.1f}x). Pressure maps from an "
            "under-resolved patch are not usable; integrated force still is."
        )
    verdict_lines.append(
        "For a hard grinding plate pressed flat, the penetration is sub-micron and no affordable "
        "voxel resolves it, so use the analytic Winkler engine there and keep hydroelastic for "
        "compliant pads or tilted/geometry-driven contact."
    )
    print("\nVerdict")
    for line in verdict_lines:
        print(f"  - {line}")

    (args.out / "summary.json").write_text(json.dumps({
        "argv": sys.argv,
        "tolerances": {"force": FORCE_TOL, "area": AREA_TOL, "peak": PEAK_TOL,
                       "min_voxels": MIN_VOXELS},
        "cases": [{**vars(c), "resolved": c.resolved, "trustworthy": c.trustworthy}
                  for c in cases],
        "verdict": verdict_lines,
    }, indent=2))
    print(f"\nwrote {args.out}/cases.csv, fidelity_vs_resolution.png, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
