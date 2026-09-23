#!/usr/bin/env python3
"""Rebuild a part run's replay recording so it carries the whole part, not one chunk.

`press_part.py` rebuilds the stage per SDF chunk, so the `press.usda` it writes holds whichever
slab happened to be resident at the end — the disc then flies over empty space for most of the
pass. Everything needed to author the recording properly is already on disk (`geometry.npz`, the
per-waypoint patches, and the TCP recorded in `summary.json`), so this rebuilds it from the saved
run: one slab over the entire band, the disc placed at every pressed waypoint, the contact patch
coloured by pressure, and the marked region drawn in green so it is obvious when contact leaves it.

No Isaac Sim, no GPU, a couple of seconds per run:

    python scripts/grind/rebake_part_usd.py runs/part_*
    python scripts/replay_recording.py runs/part_BHARAT_CLOUD_3/press.usda --loop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from grind_sim.analytic import PadSpec  # noqa: E402
from grind_sim.geometry import make_pad  # noqa: E402
from grind_sim.partmesh import slab_mesh  # noqa: E402
from grind_sim.tcp import TcpFrame, Transform, waypoint_frame  # noqa: E402
from grind_sim.viz import PressFrame, write_press_usd  # noqa: E402


def rebake(run: Path, *, cell: float, thickness: float, margin: float, fps: float,
           lift: float, touch_radius: float, marker_cap: int) -> Path:
    summary = json.loads((run / "summary.json").read_text())
    geometry = np.load(run / "geometry.npz")

    tcp_record = summary["tcp"]
    if tcp_record.get("source_csv"):
        nominal = Path(tcp_record["source_csv"]).with_name("TCP_7in_Nominal.csv")
        tcp = TcpFrame.from_csv_pair(nominal, Path(tcp_record["source_csv"]))
    else:
        tcp = TcpFrame.from_parameters(name="nominal")
    if tcp_record.get("spin_deg"):
        tcp = tcp.spun(tcp_record["spin_deg"])

    # One slab over the whole neighbourhood the pass ever saw, rather than a 150 mm chunk of it.
    # This mesh is for looking at, so it can be coarser than the one the SDF was baked from.
    part_mesh = slab_mesh(geometry["neighbourhood_local"], cell=cell, thickness=thickness,
                          margin=margin)
    pad_mesh = make_pad(PadSpec(thickness=summary["config"].get("pad_thickness_m", 0.0127)))

    marked = geometry["marked_local"]
    marked_tree = cKDTree(marked[:, :2])
    touched = np.zeros(len(marked), dtype=bool)

    frames, touched_by_frame = [], []
    for patch_file in sorted((run / "patches").glob("wp_*.npz")):
        patch = np.load(patch_file)
        triad = waypoint_frame(normal=patch["normal"], direction=patch["tangent"])
        tcp_in_world = Transform(triad.rotation,
                                 patch["position"] + triad.rotation[:, 2] * float(patch["depth"]))
        translate, quat = tcp.pad_pose_at(tcp_in_world)

        # Raise the patch clear of the marked overlay. Both are point clouds sitting on the same
        # surface, and without this the band is drawn over the contact exactly where they overlap —
        # hiding the one thing worth looking at.
        centroid = np.asarray(patch["centroid"], dtype=np.float64)
        if len(centroid):
            centroid = centroid + np.asarray(patch["normal"], dtype=np.float64) * lift
            for index in marked_tree.query_ball_point(patch["centroid"][:, :2], touch_radius):
                touched[index] = True
        frames.append(PressFrame(tuple(translate), quat, centroid, patch["pressure"],
                                 label=patch_file.stem))
        touched_by_frame.append(touched.copy())
    if not frames:
        raise SystemExit(f"{run}: no patches to rebake")

    usd_path = run / "press.usda"
    if usd_path.exists():
        usd_path.unlink()
    write_press_usd(str(usd_path), pad_mesh, part_mesh, frames, fps=fps,
                    marker_points=marked, marker_touched=touched_by_frame,
                    max_points=marker_cap)

    lo, hi = part_mesh.bounds
    centre = (lo + hi) / 2.0
    span = float(np.linalg.norm(hi - lo))
    eye = centre + np.array([0.0, -0.45 * span, 0.55 * span])
    ground = float(touched.mean()) * 100.0
    print(f"{run.name}: {len(frames)} frames, part {(hi[0]-lo[0])*1e3:.0f} x "
          f"{(hi[1]-lo[1])*1e3:.0f} mm, {ground:.0f}% of the marked region ground -> {usd_path}")
    print(f"    python scripts/replay_recording.py {usd_path} --loop \\\n"
          f"        --eye {eye[0]:.3f} {eye[1]:.3f} {eye[2]:.3f} "
          f"--target {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f}")
    return usd_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="run directories from press_part.py")
    parser.add_argument("--cell", type=float, default=0.003,
                        help="display resample spacing [m]; coarser than the sim's 2 mm")
    parser.add_argument("--thickness", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.03,
                        help="flat skirt beyond the scanned neighbourhood [m]")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--lift", type=float, default=0.0015,
                        help="raise the contact patch this far off the surface for display [m]")
    parser.add_argument("--touch-radius", type=float, default=0.004,
                        help="a marked point within this of a contact point counts as ground [m]")
    parser.add_argument("--marker-points", type=int, default=12000,
                        help="cap on marked points drawn; they carry a colour per frame")
    args = parser.parse_args()

    for run in args.runs:
        if not (run / "summary.json").exists():
            print(f"{run}: not a part run, skipped")
            continue
        rebake(run, cell=args.cell, thickness=args.thickness, margin=args.margin, fps=args.fps,
               lift=args.lift, touch_radius=args.touch_radius, marker_cap=args.marker_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
