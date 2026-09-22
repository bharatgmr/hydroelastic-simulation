"""Turn a path walk into coverage maps and the two coverage numbers, side by side.

    source env.sh
    python scripts/grind/path_report.py runs/path_nominal runs/path_rim_7p5 --out runs/path_report

No GPU and no Isaac Sim: it reads what ``press_path.py`` wrote.
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

from grind_sim.coverage import (  # noqa: E402
    coverage_gap,
    overlap_stats,
    plate_point_cloud,
    platform_coverage,
    sim_coverage,
)


def load_run(run: Path) -> dict:
    """Read one run's summary plus every patch it saved."""
    summary = json.loads((run / "summary.json").read_text())
    patches, positions = [], []
    for path in sorted((run / "patches").glob("wp_*.npz")):
        data = np.load(path)
        patches.append((data["centroid"], data["pressure"], data["area"]))
        positions.append(data["position"])
    return {"summary": summary, "patches": patches,
            "positions": np.array(positions) if positions else np.zeros((0, 3)),
            "name": run.name}


def analyse(run: dict, cell_size: float) -> dict:
    """Compute both coverage measures for one run."""
    path = run["summary"]["path"]
    size_x, size_y = path["plate_x_m"], path["plate_y_m"]
    grid = sim_coverage(run["patches"], size_x, size_y, cell_size)
    cloud = plate_point_cloud(size_x, size_y, spacing=cell_size)
    platform = platform_coverage(run["positions"], cloud,
                                 tool_contact_width=path["tool_contact_width_m"])
    return {"grid": grid, "platform": platform, "gap": coverage_gap(grid, platform),
            "overlap": overlap_stats(grid),
            "path": path, "summary": run["summary"], "name": run["name"]}


def plot(analyses: list[dict], out: Path, cell_size: float) -> None:
    """One column per run: contact coverage, mean pressure, and the two percentages."""
    fig, axes = plt.subplots(2, len(analyses), figsize=(6.0 * len(analyses), 9.5), squeeze=False)
    for col, analysis in enumerate(analyses):
        grid = analysis["grid"]
        path = analysis["path"]
        extent = [-path["plate_x_m"] / 2 * 1e3, path["plate_x_m"] / 2 * 1e3,
                  -path["plate_y_m"] / 2 * 1e3, path["plate_y_m"] / 2 * 1e3]

        ax = axes[0][col]
        visits = np.where(grid.visits > 0, grid.visits, np.nan)
        image = ax.imshow(visits, origin="lower", extent=extent, cmap="viridis")
        fig.colorbar(image, ax=ax, fraction=0.046, label="passes over the cell")
        tcp = analysis["summary"]["tcp"]
        ax.set_title(f"{analysis['name']}\n{tcp['name']} — tilt {tcp['tilt_deg']:.1f}°, "
                     f"TCP {tcp['radial_offset_m']*1e3:.0f} mm out", fontsize=10)

        ax = axes[1][col]
        pressure = np.where(grid.contact_area > 0, grid.mean_pressure / 1e3, np.nan)
        image = ax.imshow(pressure, origin="lower", extent=extent, cmap="inferno")
        fig.colorbar(image, ax=ax, fraction=0.046, label="mean pressure [kPa]")
        gap = analysis["gap"]
        ax.set_title(f"sim coverage {gap['sim_coverage_percentage']:.1f}% · "
                     f"cell metric {gap['platform_coverage_percentage']:.1f}% "
                     f"({gap['platform_over_sim']:.2f}x)", fontsize=10)

        for row in (0, 1):
            axes[row][col].set_xlabel("x [mm]")
            axes[row][col].set_ylabel("y [mm]")

    fig.suptitle(f"Flat-part pass: simulated contact vs the cell's coverage metric "
                 f"({cell_size*1e3:.1f} mm cells)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_grid(analysis: dict, out: Path) -> Path:
    """Save the accumulated heat map itself, not just a picture of it.

    The PNG is a view; this is the data behind it, so the map can be re-plotted, differenced
    against another pass, or fed to a removal model without re-running the simulation.
    """
    grid, path = analysis["grid"], analysis["path"]
    destination = out / f"grid_{analysis['name']}.npz"
    np.savez_compressed(
        destination,
        visits=grid.visits,
        applied_pressure_pa=grid.applied_pressure,
        mean_pressure_pa=grid.mean_pressure,
        peak_pressure_pa=grid.pressure_peak,
        contact_area_m2=grid.contact_area,
        dose_n=grid.dose,
        cell_size_m=grid.cell_size,
        origin_m=grid.origin,
        plate_size_m=np.array([path["plate_x_m"], path["plate_y_m"]]),
        tcp_name=analysis["summary"]["tcp"]["name"],
        tilt_deg=analysis["summary"]["tcp"]["tilt_deg"],
        spin_deg=analysis["summary"]["tcp"].get("spin_deg", 0.0),
    )
    return destination


def plot_combined(analysis: dict, out: Path) -> Path:
    """One sheet, one heat map: applied pressure with the untouched plate left dark.

    The comparison figure is for reading three passes against each other; this is the single map
    of what one pass did to the plate, which is the thing worth pinning on a wall.
    """
    grid, stats, path = analysis["grid"], analysis["overlap"], analysis["path"]
    extent = [-path["plate_x_m"] / 2 * 1e3, path["plate_x_m"] / 2 * 1e3,
              -path["plate_y_m"] / 2 * 1e3, path["plate_y_m"] / 2 * 1e3]

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.set_facecolor("#1b1f26")  # untouched plate reads as bare metal, not as zero pressure
    applied = np.where(grid.applied_pressure > 0, grid.applied_pressure / 1e3, np.nan)
    image = ax.imshow(applied, origin="lower", extent=extent, cmap="magma",
                      interpolation="nearest")
    bar = fig.colorbar(image, ax=ax, fraction=0.046)
    bar.set_label("applied pressure, summed over passes [kPa]")

    # Outline where passes stacked, so overlap is legible on top of the pressure field.
    ax.contour(np.clip(grid.visits, 0, 3), levels=[1.5, 2.5], extent=extent, origin="lower",
               colors=["#7fd4ff", "#ff9d5c"], linewidths=0.8)

    tcp = analysis["summary"]["tcp"]
    spin = tcp.get("spin_deg", 0.0)
    ax.set_title(f"{analysis['name']} — {tcp['name']}\n"
                 f"tilt {tcp['tilt_deg']:.1f}°" + (f", spun {spin:.0f}°" if spin else "")
                 + f" · {stats['remaining_fraction']*100:.1f}% untouched · "
                 f"{stats['overlapped_fraction']*100:.1f}% overlapped · "
                 f"peak {stats['applied_pressure_peak_pa']/1e3:.1f} kPa", fontsize=10)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.tight_layout()
    destination = out / f"heatmap_{analysis['name']}.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_overlap(analyses: list[dict], out: Path) -> None:
    """What the pass left behind: untouched plate, single hits, and where passes stacked.

    Left column classifies every cell; right column shows the pressure actually applied there,
    summed over passes, because that is what an overlap costs (or a gap withholds).
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    classes = ListedColormap(["#2b2f36", "#3d7ea6", "#e0902b", "#c2402c"])
    bounds = BoundaryNorm([0, 1, 2, 3, 99], classes.N)

    fig, axes = plt.subplots(2, len(analyses), figsize=(6.2 * len(analyses), 9.8), squeeze=False)
    for col, analysis in enumerate(analyses):
        grid, stats, path = analysis["grid"], analysis["overlap"], analysis["path"]
        extent = [-path["plate_x_m"] / 2 * 1e3, path["plate_x_m"] / 2 * 1e3,
                  -path["plate_y_m"] / 2 * 1e3, path["plate_y_m"] / 2 * 1e3]

        ax = axes[0][col]
        ax.imshow(np.clip(grid.visits, 0, 3), origin="lower", extent=extent, cmap=classes,
                  norm=bounds, interpolation="nearest")
        ax.legend(handles=[
            Patch(facecolor="#2b2f36", label=f"remaining {stats['remaining_fraction']*100:.1f}%"),
            Patch(facecolor="#3d7ea6", label=f"one pass {stats['single_pass_fraction']*100:.1f}%"),
            Patch(facecolor="#e0902b", label="two passes"),
            Patch(facecolor="#c2402c", label=f"3+ (max {stats['max_visits']})"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
        tcp = analysis["summary"]["tcp"]
        spin = tcp.get("spin_deg", 0.0)
        ax.set_title(f"{analysis['name']}\n{tcp['name']} — tilt {tcp['tilt_deg']:.1f}°"
                     + (f", spun {spin:.0f}°" if spin else ""), fontsize=10)

        ax = axes[1][col]
        applied = np.where(grid.applied_pressure > 0, grid.applied_pressure / 1e3, np.nan)
        image = ax.imshow(applied, origin="lower", extent=extent, cmap="magma")
        fig.colorbar(image, ax=ax, fraction=0.046, label="applied pressure, summed [kPa]")
        ax.set_title(f"overlapped {stats['overlapped_fraction']*100:.1f}% of plate · "
                     f"peak {stats['applied_pressure_peak_pa']/1e3:.1f} kPa", fontsize=10)

        for row in (0, 1):
            axes[row][col].set_xlabel("x [mm]")
            axes[row][col].set_ylabel("y [mm]")

    fig.suptitle("What the pass leaves: untouched plate, single hits, and stacked passes",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--cell-size", type=float, default=0.0025)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"),
                        help="also write the figures here, where they are tracked; runs/ is "
                             "gitignored, so anything written only there is lost on a fresh clone")
    args = parser.parse_args()

    analyses = []
    for run in args.runs:
        if not (run / "summary.json").exists():
            print(f"skipping {run}: no summary.json")
            continue
        analyses.append(analyse(load_run(run), args.cell_size))
    if not analyses:
        print("nothing to report")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    targets = [args.out]
    if args.figures_dir:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        targets.append(args.figures_dir)
    written = []
    for target in targets:
        plot(analyses, target / "coverage.png", args.cell_size)
        plot_overlap(analyses, target / "overlap.png")
        for analysis in analyses:
            written.append(plot_combined(analysis, target))
    # The grid data goes with the run, where the patches it came from live.
    written.extend(save_grid(analysis, args.out) for analysis in analyses)

    print(f"{'run':<22} {'tilt':>6} {'waypoints':>10} {'sim cov':>9} {'cell cov':>9} {'ratio':>7} "
          f"{'mean p':>9} {'peak p':>9} {'bend':>8}")
    print("-" * 100)
    rows = []
    for analysis in analyses:
        grid, gap = analysis["grid"], analysis["gap"]
        rows_csv = analysis["summary"]["rows"]
        touched = grid.contact_area > 0
        mean_pressure = float(grid.mean_pressure[touched].mean()) if touched.any() else 0.0
        peak_pressure = float(grid.pressure_peak.max())
        bending = float(np.mean([r["bending_moment_tool_nm"] for r in rows_csv]))
        tcp = analysis["summary"]["tcp"]
        print(f"{analysis['name']:<22} {tcp['tilt_deg']:5.1f}° "
              f"{len(rows_csv):10d} {gap['sim_coverage_percentage']:8.1f}% "
              f"{gap['platform_coverage_percentage']:8.1f}% {gap['platform_over_sim']:7.2f} "
              f"{mean_pressure/1e3:8.2f}k {peak_pressure/1e3:8.2f}k {bending:7.3f}")
        rows.append({
            "run": analysis["name"],
            **{f"overlap_{k}": v for k, v in analysis["overlap"].items()},
            "tcp": tcp["name"],
            "tilt_deg": tcp["tilt_deg"],
            "waypoints_pressed": len(rows_csv),
            **gap,
            "mean_pressure_pa": mean_pressure,
            "peak_pressure_pa": peak_pressure,
            "mean_bending_moment_nm": bending,
            "swept_area_m2": analysis["path"]["swept_area_m2"],
            "platform_bucket_coverage": analysis["platform"]["bucket_coverage"],
        })

    print(f"\n{'run':<22} {'remaining':>10} {'one pass':>10} {'overlapped':>11} "
          f"{'applied 1x':>11} {'applied 2x+':>12} {'peak applied':>13}")
    print("-" * 95)
    for analysis in analyses:
        stats = analysis["overlap"]
        print(f"{analysis['name']:<22} {stats['remaining_fraction']*100:9.1f}% "
              f"{stats['single_pass_fraction']*100:9.1f}% {stats['overlapped_fraction']*100:10.1f}% "
              f"{stats['applied_pressure_single_pa']/1e3:10.2f}k "
              f"{stats['applied_pressure_overlapped_pa']/1e3:11.2f}k "
              f"{stats['applied_pressure_peak_pa']/1e3:12.2f}k")

    print("\nThe cell's metric counts a point as touched when it lies under the disc cylinder and\n"
          "within backing_pad_height (25 mm) of the contact plane, and its cylinder never tilts.\n"
          "The simulated coverage counts cells where contact pressure actually landed. The ratio\n"
          "is how much the geometric allowance flatters the reported number.")

    (args.out / "summary.json").write_text(json.dumps({"argv": sys.argv, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}/coverage.png, overlap.png, summary.json")
    print(f"      per-pass heat maps + grids: {', '.join(p.name for p in written)}")
    if args.figures_dir:
        print(f"      figures also in {args.figures_dir} (tracked; runs/ is gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
