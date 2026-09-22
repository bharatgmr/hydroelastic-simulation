"""Collect run outputs into one JSON for the config explorer page.

Every run directory contributes its summary plus a thinned copy of each contact patch, so the
page can draw the patch without shipping 200k faces.

    source env.sh
    python scripts/grind/export_ui_data.py runs/tcp_* runs/gate_* --out runs/ui/data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAX_POINTS = 2500


def thin(points: np.ndarray, pressure: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a patch for display, keeping the peak so the colour scale stays honest."""
    if len(points) <= MAX_POINTS:
        return points, pressure
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), MAX_POINTS - 1, replace=False)
    keep = np.append(keep, int(np.argmax(pressure)))
    return points[keep], pressure[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = []
    for run in sorted(args.runs):
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        config = summary.get("config", {})
        tcp = summary.get("tcp")
        for row in summary.get("rows", []):
            if not row.get("n_faces"):
                continue
            force = row.get("force_target_n")
            patch_path = run / "patches" / f"force_{force:.0f}N.npz"
            if not patch_path.exists():
                continue
            data = np.load(patch_path, allow_pickle=True)
            points, pressure = thin(data["centroid"], data["pressure"])
            depth = row.get("tcp_depth_m")
            cases.append({
                "run": run.name,
                "kind": "tcp" if tcp else "gate",
                "tcp": tcp,
                "force_target_n": force,
                "pad_modulus_pa": config.get("pad_modulus_pa"),
                "tilt_deg": (tcp or {}).get("tilt_deg", config.get("tilt_deg", 0.0)),
                "radial_mm": (tcp or {}).get("radial_offset_m", 0.0) * 1e3,
                "voxel_m": config.get("voxel_m"),
                "sheet_voxel_m": config.get("sheet_voxel_m"),
                "metrics": {k: v for k, v in row.items() if isinstance(v, (int, float))},
                "tcp_depth_m": depth,
                "argv": summary.get("argv"),
                "points_mm": np.round(points[:, :2] * 1e3, 2).tolist(),
                "pressure_kpa": np.round(pressure / 1e3, 3).tolist(),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}, separators=(",", ":")))
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} — {len(cases)} cases, {size_kb:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
