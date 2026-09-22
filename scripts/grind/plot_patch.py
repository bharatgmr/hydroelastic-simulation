"""Draw the contact patches a press run saved, as pressure maps in the sheet frame.

This is the picture behind the gate table: where the pad actually touches, and how hard. No GPU
and no Isaac Sim — it reads the ``patches/*.npz`` written by ``press_flat.py``.

    source env.sh
    python scripts/grind/plot_patch.py runs/gate_* --out runs/patch_maps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grind_sim.analytic import PadSpec  # noqa: E402


def draw(ax, npz_path: Path, pad: PadSpec) -> str:
    """Draw one patch onto an axis and return a one-line caption."""
    data = np.load(npz_path)
    centroid = data["centroid"]
    pressure = data["pressure"]
    area = data["area"]
    force = float(np.sum(pressure * area))

    # Pad outline at its commanded pose: the patch lives under the pad, in the sheet's frame.
    for radius in (pad.outer_radius, pad.inner_radius):
        ax.add_patch(plt.Circle((0, 0), radius * 1e3, fill=False, color="0.6", lw=1, ls="--"))

    if len(pressure):
        # Size the dots to the face area so a sparse patch doesn't read as a dense one.
        sizes = np.clip(area / area.max() * 18.0, 1.0, 18.0)
        dots = ax.scatter(centroid[:, 0] * 1e3, centroid[:, 1] * 1e3, c=pressure / 1e3,
                          s=sizes, cmap="inferno", linewidths=0)
        bar = ax.figure.colorbar(dots, ax=ax, fraction=0.046, pad=0.02)
        bar.set_label("pressure [kPa]", fontsize=8)
        bar.ax.tick_params(labelsize=7)

    limit = pad.outer_radius * 1e3 * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.tick_params(labelsize=7)

    modulus = float(data["pad_modulus_pa"])
    label = f"{modulus/1e6:.0f} MPa" if modulus >= 1e6 else f"{modulus/1e3:.0f} kPa"
    ax.set_title(
        f"{npz_path.parent.parent.name}\n"
        f"E={label}, tilt {float(data['tilt_deg']):.0f}°, "
        f"{float(data['force_target_n']):.0f} N target",
        fontsize=9,
    )
    return (f"{npz_path.parent.parent.name}: {len(pressure)} faces, {force:.1f} N, "
            f"peak {pressure.max()/1e3:.1f} kPa, area {area.sum()*1e4:.1f} cm^2"
            if len(pressure) else f"{npz_path.parent.parent.name}: no contact")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--force", type=float, default=25.0, help="which force step to draw")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    found = []
    for run in args.runs:
        path = run / "patches" / f"force_{args.force:.0f}N.npz"
        if path.exists():
            found.append(path)
        else:
            print(f"skipping {run}: no patch at {args.force:.0f} N")
    if not found:
        print("nothing to draw")
        return 1

    pad = PadSpec()
    cols = min(len(found), 2)
    rows = (len(found) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.2 * rows), squeeze=False)
    for ax, path in zip(axes.ravel(), found):
        print(draw(ax, path, pad))
    for ax in axes.ravel()[len(found):]:
        ax.axis("off")
    fig.suptitle(f"Contact pressure at {args.force:.0f} N (sheet frame, pad outline dashed)",
                 fontsize=11)
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"patches_{args.force:.0f}N.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
