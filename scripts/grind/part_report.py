"""Draw what a pass did to each scanned part: contact patches over the marked band.

Reads what ``press_part.py`` wrote and maps every patch back into the band's own frame, so the
contact can be read against the green-painted region it was supposed to treat. No GPU, no Isaac.

    source env.sh
    python scripts/grind/part_report.py runs/part_* --out runs/part_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grind_sim.coverage import CoverageGrid, overlap_stats  # noqa: E402


def load(run: Path) -> dict:
    """Read a part run: its geometry, its rows, and every contact patch."""
    summary = json.loads((run / "summary.json").read_text())
    geometry = np.load(run / "geometry.npz")
    patches = []
    for path in sorted((run / "patches").glob("wp_*.npz")):
        data = np.load(path)
        if len(data["centroid"]):
            patches.append((data["centroid"], data["pressure"], data["area"]))
    return {"name": run.name.replace("part_", ""), "summary": summary,
            "geometry": geometry, "patches": patches}


def grid_for(run: dict, cell: float) -> CoverageGrid:
    """Rasterise the patches in the band's local frame, spanning the marked region plus a margin."""
    marked = run["geometry"]["marked_local"]
    span_x = float(marked[:, 0].max() - marked[:, 0].min()) + 0.08
    span_y = float(marked[:, 1].max() - marked[:, 1].min()) + 0.12
    grid = CoverageGrid.empty(span_x, span_y, cell)
    # CoverageGrid centres itself on the origin; the band's local frame is already centred on the
    # marked region's centroid, so the two line up without a shift.
    for centroid, pressure, area in run["patches"]:
        grid.add_patch(centroid, pressure, area)
    return grid


def marked_coverage(run: dict, grid: CoverageGrid) -> dict:
    """How much of the *marked* region the pass actually touched.

    Coverage over the whole raster is not the question a weld asks — the band is what was painted
    for treatment, so the number that matters is the fraction of marked points that ended up under
    real contact.
    """
    marked = run["geometry"]["marked_local"]
    nx = grid.shape[1]
    ix = np.floor((marked[:, 0] - grid.origin[0]) / grid.cell_size).astype(int)
    iy = np.floor((marked[:, 1] - grid.origin[1]) / grid.cell_size).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < grid.shape[0])
    touched = np.zeros(len(marked), dtype=bool)
    flat_touch = (grid.contact_area > 0).ravel()
    touched[inside] = flat_touch[iy[inside] * nx + ix[inside]]
    return {"marked_points": int(len(marked)),
            "marked_touched": int(touched.sum()),
            "marked_coverage_percent": 100.0 * float(touched.mean())}


def contact_offsets(run: dict) -> dict:
    """How far the contact sat from the line it was told to follow.

    The tool is 178 mm across but the band is 13-45 mm wide, so on a part with any relief the rim
    can find neighbouring material before the marked region — the contact then sits off to one
    side and the painted band is not what gets ground. Across-band offset is the number that
    matters; along-band offset is mostly the tool's own lead, since a tilted TCP contacts ahead of
    its own axis.
    """
    rows = run["summary"]["rows"]
    across = np.array([r["cop_y_m"] - r["y_m"] for r in rows]) * 1e3
    along = np.array([r["cop_x_m"] - r["x_m"] for r in rows]) * 1e3
    half_width = run["summary"]["band"]["width_m"] * 1e3 / 2.0
    missed = np.abs(across) > half_width
    return {"across_mm": across, "along_mm": along, "half_width_mm": half_width,
            "off_band_fraction": float(missed.mean()),
            "across_mean_mm": float(across.mean()),
            "across_abs_p95_mm": float(np.percentile(np.abs(across), 95))}


def plot(runs: list[dict], grids: list[CoverageGrid], out: Path) -> None:
    """One column per part: the patch field over the band, the loads, and where contact landed."""
    fig, axes = plt.subplots(3, len(runs), figsize=(5.6 * len(runs), 12.2), squeeze=False)
    for col, (run, grid) in enumerate(zip(runs, grids)):
        marked = run["geometry"]["marked_local"]
        line = run["geometry"]["centreline"]
        extent = [grid.origin[0] * 1e3, (grid.origin[0] + grid.shape[1] * grid.cell_size) * 1e3,
                  grid.origin[1] * 1e3, (grid.origin[1] + grid.shape[0] * grid.cell_size) * 1e3]

        ax = axes[0][col]
        ax.set_facecolor("#1b1f26")
        ax.scatter(marked[:, 0] * 1e3, marked[:, 1] * 1e3, s=1, c="#2f7d4f", alpha=0.35,
                   label="marked region", linewidths=0)
        applied = np.where(grid.applied_pressure > 0, grid.applied_pressure / 1e3, np.nan)
        image = ax.imshow(applied, origin="lower", extent=extent, cmap="magma",
                          interpolation="nearest", alpha=0.95)
        fig.colorbar(image, ax=ax, fraction=0.04, label="applied pressure [kPa]")
        ax.plot(line[:, 0] * 1e3, line[:, 1] * 1e3, color="#7fd4ff", lw=1.0, label="centreline")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.3)
        ax.set_aspect("equal")
        ax.set_title(f"{run['name']}\n{len(run['patches'])} presses along "
                     f"{np.linalg.norm(np.diff(line, axis=0), axis=1).sum()*1e3:.0f} mm",
                     fontsize=10)

        ax = axes[2][col]
        offsets = contact_offsets(run)
        x_along = np.array([r["x_m"] for r in run["summary"]["rows"]]) * 1e3
        ax.axhspan(-offsets["half_width_mm"], offsets["half_width_mm"], color="#2f7d4f",
                   alpha=0.25, label="marked band")
        ax.axhline(0.0, color="#7fd4ff", lw=1.0, label="commanded line")
        ax.plot(x_along, offsets["across_mm"], color="#c2402c", lw=1.2,
                label="contact centre, across")
        ax.plot(x_along, offsets["along_mm"], color="#6b7280", lw=1.0, ls="--",
                label="contact centre, along")
        ax.set_xlabel("along the band [mm]")
        ax.set_ylabel("offset from the line [mm]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_title(f"contact sits off the line on {offsets['off_band_fraction']*100:.0f}% of "
                     f"waypoints", fontsize=10)

        ax = axes[1][col]
        rows = run["summary"]["rows"]
        x = np.array([r["x_m"] for r in rows]) * 1e3
        ax.plot(x, [r["patch_area_m2"] * 1e4 for r in rows], color="#c2600f", label="patch area [cm2]")
        ax.plot(x, [r["pressure_peak_pa"] / 1e3 for r in rows], color="#3d7ea6",
                label="peak pressure [kPa]")
        ax.plot(x, [r["force_axial_n"] for r in rows], color="#6b7280", ls="--",
                label="axial force [N]")
        ax.set_xlabel("along the band [mm]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title("what varies along the pass", fontsize=10)
        axes[0][col].set_xlabel("along the band [mm]")
        axes[0][col].set_ylabel("across [mm]")

    fig.suptitle("A single pass down the marked band, on the scanned surface", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_part(run: dict, grid: CoverageGrid, out: Path) -> Path:
    """One part, one file: the patch field, the loads along the pass, and where contact landed.

    The comparison figure is for reading five parts against each other; this is the one to open
    when the question is what happened on a single scan.
    """
    marked = run["geometry"]["marked_local"]
    line = run["geometry"]["centreline"]
    rows = run["summary"]["rows"]
    offsets = contact_offsets(run)
    hit = marked_coverage(run, grid)
    tcp = run["summary"]["tcp"]
    band = run["summary"]["band"]
    extent = [grid.origin[0] * 1e3, (grid.origin[0] + grid.shape[1] * grid.cell_size) * 1e3,
              grid.origin[1] * 1e3, (grid.origin[1] + grid.shape[0] * grid.cell_size) * 1e3]
    x_along = np.array([r["x_m"] for r in rows]) * 1e3

    fig = plt.figure(figsize=(11.0, 10.6))
    spec = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.0, 1.0], hspace=0.32)

    ax = fig.add_subplot(spec[0])
    ax.set_facecolor("#1b1f26")
    ax.scatter(marked[:, 0] * 1e3, marked[:, 1] * 1e3, s=1.5, c="#3f9c66", alpha=0.35,
               linewidths=0, label="marked region")
    applied = np.where(grid.applied_pressure > 0, grid.applied_pressure / 1e3, np.nan)
    image = ax.imshow(applied, origin="lower", extent=extent, cmap="magma",
                      interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01, label="applied pressure [kPa]")
    ax.plot(line[:, 0] * 1e3, line[:, 1] * 1e3, color="#7fd4ff", lw=1.2, label="commanded line")
    cop_x = np.array([r["cop_x_m"] for r in rows]) * 1e3
    cop_y = np.array([r["cop_y_m"] for r in rows]) * 1e3
    ax.plot(cop_x, cop_y, color="#ffd166", lw=0.9, ls="--", label="contact centre")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.35)
    ax.set_aspect("equal")
    ax.set_xlabel("along the band [mm]")
    ax.set_ylabel("across [mm]")
    ax.set_title(f"{run['name']} — band {band['length_m']*1e3:.0f} x {band['width_m']*1e3:.0f} mm, "
                 f"{len(rows)} presses at {run['summary']['config']['force_setpoint_n']:.0f} N "
                 f"through {tcp['name']}", fontsize=11)

    ax = fig.add_subplot(spec[1])
    ax.plot(x_along, [r["patch_area_m2"] * 1e4 for r in rows], color="#c2600f",
            label="patch area [cm2]")
    ax.plot(x_along, [r["pressure_peak_pa"] / 1e3 for r in rows], color="#3d7ea6",
            label="peak pressure [kPa]")
    ax.plot(x_along, [r["force_axial_n"] for r in rows], color="#6b7280", ls="--",
            label="axial force [N]")
    ax.set_xlabel("along the band [mm]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("what varies along the pass", fontsize=10)

    ax = fig.add_subplot(spec[2])
    ax.axhspan(-offsets["half_width_mm"], offsets["half_width_mm"], color="#3f9c66", alpha=0.22,
               label="marked band")
    ax.axhline(0.0, color="#7fd4ff", lw=1.0, label="commanded line")
    ax.plot(x_along, offsets["across_mm"], color="#c2402c", lw=1.3, label="contact centre, across")
    ax.plot(x_along, offsets["along_mm"], color="#6b7280", lw=1.0, ls="--",
            label="contact centre, along")
    ax.set_xlabel("along the band [mm]")
    ax.set_ylabel("offset from the line [mm]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"contact centre off the band on {offsets['off_band_fraction']*100:.0f}% of "
                 f"waypoints · {hit['marked_coverage_percent']:.0f}% of the marked region touched",
                 fontsize=10)

    destination = out / f"part_{run['name']}.png"
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--cell-size", type=float, default=0.0015)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    runs, grids = [], []
    for path in args.runs:
        if not (path / "summary.json").exists():
            print(f"skipping {path}: no summary.json")
            continue
        run = load(path)
        runs.append(run)
        grids.append(grid_for(run, args.cell_size))
    if not runs:
        print("nothing to report")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    targets = [args.out]
    if args.figures_dir:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        targets.append(args.figures_dir)
    written = []
    for target in targets:
        plot(runs, grids, target / "parts.png")
        written.extend(plot_part(run, grid, target) for run, grid in zip(runs, grids))

    print(f"{'part':<18} {'band mm':>10} {'presses':>8} {'marked hit':>11} {'patch cm2':>10} "
          f"{'peak kPa':>9} {'force N':>9} {'off-band':>9} {'across mm':>10}")
    print("-" * 108)
    rows_out = []
    for run, grid in zip(runs, grids):
        rows = run["summary"]["rows"]
        stats = overlap_stats(grid)
        hit = marked_coverage(run, grid)
        band = run["summary"]["band"]
        area = np.array([r["patch_area_m2"] for r in rows]) * 1e4
        peak = np.array([r["pressure_peak_pa"] for r in rows]) / 1e3
        force = np.array([r["force_axial_n"] for r in rows])
        arm = np.array([r["moment_arm_mm"] for r in rows])
        bend = np.array([r["bending_moment_tool_nm"] for r in rows])
        offsets = contact_offsets(run)
        print(f"{run['name']:<18} {band['length_m']*1e3:6.0f}x{band['width_m']*1e3:<3.0f} "
              f"{len(rows):8d} {hit['marked_coverage_percent']:10.1f}% "
              f"{area.mean():6.2f}+-{area.std():<3.2f} {peak.mean():8.1f} {force.mean():8.2f} "
              f"{offsets['off_band_fraction']*100:8.0f}% {offsets['across_mean_mm']:+9.1f}")
        rows_out.append({"part": run["name"], **hit,
                         "contact_off_band_fraction": offsets["off_band_fraction"],
                         "contact_across_mean_mm": offsets["across_mean_mm"],
                         "contact_across_abs_p95_mm": offsets["across_abs_p95_mm"],
                         "moment_arm_mean_mm": float(arm.mean()),
                         "bending_mean_nm": float(bend.mean()),
                         "band_length_m": band["length_m"], "band_width_m": band["width_m"],
                         "presses": len(rows),
                         "patch_area_mean_cm2": float(area.mean()),
                         "patch_area_std_cm2": float(area.std()),
                         "pressure_peak_mean_kpa": float(peak.mean()),
                         "force_axial_mean_n": float(force.mean()),
                         **{f"overlap_{k}": v for k, v in stats.items()}})

    (args.out / "summary.json").write_text(json.dumps({"argv": sys.argv, "rows": rows_out},
                                                      indent=2))
    print("\nThe disc is 178 mm across and these bands are 13-45 mm wide, so wherever the part has\n"
          "relief the rim can reach neighbouring material before the marked region. Where that\n"
          "happens the contact sits off the line and the painted band is not what gets ground.")
    print(f"\nwrote {args.out}/parts.png, summary.json, and one figure per part: "
          + ", ".join(sorted({path.name for path in written}))
          + (f"\n      figures also in {args.figures_dir}" if args.figures_dir else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
