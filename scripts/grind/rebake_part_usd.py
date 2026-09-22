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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from grind_sim.analytic import PadSpec  # noqa: E402
from grind_sim.geometry import make_pad  # noqa: E402
from grind_sim.partmesh import slab_mesh  # noqa: E402
from grind_sim.tcp import TcpFrame, Transform, waypoint_frame  # noqa: E402
from grind_sim.viz import PressFrame, write_press_usd  # noqa: E402


def rebake(run: Path, *, cell: float, thickness: float, margin: float, fps: float) -> Path:
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

    frames = []
    for patch_file in sorted((run / "patches").glob("wp_*.npz")):
        patch = np.load(patch_file)
        triad = waypoint_frame(normal=patch["normal"], direction=patch["tangent"])
        tcp_in_world = Transform(triad.rotation,
                                 patch["position"] + triad.rotation[:, 2] * float(patch["depth"]))
        translate, quat = tcp.pad_pose_at(tcp_in_world)
        frames.append(PressFrame(tuple(translate), quat, patch["centroid"], patch["pressure"],
                                 label=patch_file.stem))
    if not frames:
        raise SystemExit(f"{run}: no patches to rebake")

    usd_path = run / "press.usda"
    if usd_path.exists():
        usd_path.unlink()
    write_press_usd(str(usd_path), pad_mesh, part_mesh, frames, fps=fps,
                    marker_points=geometry["marked_local"])

    lo, hi = part_mesh.bounds
    centre = (lo + hi) / 2.0
    span = float(np.linalg.norm(hi - lo))
    eye = centre + np.array([0.0, -0.45 * span, 0.55 * span])
    print(f"{run.name}: {len(frames)} frames, part {(hi[0]-lo[0])*1e3:.0f} x "
          f"{(hi[1]-lo[1])*1e3:.0f} mm -> {usd_path}")
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
    args = parser.parse_args()

    for run in args.runs:
        if not (run / "summary.json").exists():
            print(f"{run}: not a part run, skipped")
            continue
        rebake(run, cell=args.cell, thickness=args.thickness, margin=args.margin, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
