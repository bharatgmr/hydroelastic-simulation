"""Step 0 of the fidelity gate: can hydroelastic contact resolve this case at all?

No simulator is involved. For each (pad modulus, force, tilt) it computes the Winkler penetration
and asks what voxel size would be needed to resolve it, then what that voxel size would cost. The
rule from the grind prompt is that penetration under two voxels is not resolved.

    source env.sh
    python scripts/grind/feasibility.py [--memory-budget-gb 2.0] [--out runs/feasibility]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grind_sim.analytic import (  # noqa: E402
    PadSpec,
    foundation_stiffness,
    press_flat,
    press_tilted,
    sdf_memory_mib,
)

# Prompt's calibration sweep for the unknown pad modulus, plus the foam interface pad that
# sanding-wm actually runs (E ~ 100 kPa over 19 mm) as a soft-end reference point.
MODULI_PA = (20.0e6, 100.0e6, 500.0e6)
FOAM_REFERENCE_PA = 100.0e3
# sanding-wm's sanding envelope, used as the default until grind numbers arrive.
FORCES_N = (10.0, 25.0, 30.0)
TILTS_DEG = (0.0, 2.0, 5.0)
VOXELS_PER_PENETRATION = 2.0


def smallest_affordable_voxel(budget_gb: float) -> float:
    """Smallest voxel whose SDF still fits in ``budget_gb`` gigabytes [m].

    Uses the measured cost curve (see :func:`sdf_memory_mib`), not the band-volume count, which
    is about 1000x optimistic. Cost grows as 1/voxel^3, so walk up until it fits.
    """
    voxel = 1.0e-5
    while sdf_memory_mib(voxel) > budget_gb * 1024.0:
        voxel *= 1.02
        if voxel > 0.05:
            raise RuntimeError("no affordable voxel below 50 mm; budget is too small")
    return voxel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-budget-gb", type=float, default=2.0)
    parser.add_argument("--pad-thickness", type=float, default=0.0127)
    parser.add_argument("--out", type=Path, default=None, help="directory for summary.json")
    args = parser.parse_args()

    pad = PadSpec(thickness=args.pad_thickness)
    floor_voxel = smallest_affordable_voxel(args.memory_budget_gb)

    print(f"pad: OD {2*pad.outer_radius*1000:.1f} mm, bore {2*pad.inner_radius*1000:.1f} mm, "
          f"thickness {pad.thickness*1000:.1f} mm, face area {pad.face_area*1e4:.1f} cm^2")
    print(f"smallest voxel within {args.memory_budget_gb:.1f} GB: {floor_voxel*1e6:.0f} um "
          f"(measured curve: {sdf_memory_mib(floor_voxel):.0f} MiB)")
    print(f"resolvable means penetration >= {VOXELS_PER_PENETRATION:.0f} voxels\n")

    header = (f"{'E_pad':>9} {'k [N/m^3]':>10} {'F [N]':>6} {'tilt':>5} {'area %':>7} "
              f"{'p_peak [kPa]':>12} {'delta':>10} {'voxel need':>11} {'SDF at need':>12}  verdict")
    print(header)
    print("-" * len(header))

    rows = []
    for modulus in (*MODULI_PA, FOAM_REFERENCE_PA):
        k = foundation_stiffness(modulus, pad.thickness)
        for force in FORCES_N:
            for tilt in TILTS_DEG:
                res = (press_flat(pad, k, force) if tilt == 0.0
                       else press_tilted(pad, k, force, tilt))
                voxel_needed = res.penetration_max / VOXELS_PER_PENETRATION
                gib = sdf_memory_mib(voxel_needed) / 1024.0
                ok = voxel_needed >= floor_voxel
                verdict = "resolvable" if ok else "UNRESOLVABLE"
                rows.append({
                    "modulus_pa": modulus,
                    "k_n_per_m3": k,
                    "force_n": force,
                    "tilt_deg": tilt,
                    "contact_fraction": res.contact_fraction,
                    "pressure_peak_pa": res.pressure_peak,
                    "pressure_mean_pa": res.pressure_mean,
                    "penetration_m": res.penetration_max,
                    "voxel_needed_m": voxel_needed,
                    "sdf_gib_at_needed_voxel": gib,
                    "resolvable": ok,
                })
                label = (f"{modulus/1e6:7.3f}M" if modulus >= 1e6 else f"{modulus/1e3:7.0f}k")
                print(f"{label:>9} {k:10.2e} {force:6.1f} {tilt:4.1f}d "
                      f"{res.contact_fraction*100:6.1f}% {res.pressure_peak/1e3:12.2f} "
                      f"{_fmt_len(res.penetration_max):>10} {_fmt_len(voxel_needed):>11} "
                      f"{_fmt_mem(gib):>12}  {verdict}")

    # The decisive number: the stiffest pad that stays resolvable at the nominal operating point.
    nominal_force, nominal_tilt = 25.0, 0.0
    res_ref = press_flat(pad, 1.0, nominal_force)  # unit k gives pressure directly
    k_max = res_ref.pressure_peak / (VOXELS_PER_PENETRATION * floor_voxel)
    e_max = k_max * pad.thickness
    print(f"\nAt {nominal_force:.0f} N and {nominal_tilt:.0f} deg tilt, staying resolvable within "
          f"{args.memory_budget_gb:.0f} GB needs k <= {k_max:.2e} N/m^3,")
    print(f"i.e. a pad modulus E <= {e_max/1e3:.0f} kPa over {pad.thickness*1000:.1f} mm — "
          "foam, not a hard grinding plate.")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        summary = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": sys.argv,
            "pad": {
                "outer_radius_m": pad.outer_radius,
                "inner_radius_m": pad.inner_radius,
                "thickness_m": pad.thickness,
                "face_area_m2": pad.face_area,
            },
            "memory_budget_gb": args.memory_budget_gb,
            "floor_voxel_m": floor_voxel,
            "voxels_per_penetration": VOXELS_PER_PENETRATION,
            "max_resolvable_k_n_per_m3": k_max,
            "max_resolvable_modulus_pa": e_max,
            "rows": rows,
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out / 'summary.json'}")
    return 0


def _fmt_len(metres: float) -> str:
    """Format a length in whichever of mm/um/nm reads naturally."""
    if metres >= 1e-3:
        return f"{metres*1e3:.3f} mm"
    if metres >= 1e-6:
        return f"{metres*1e6:.2f} um"
    return f"{metres*1e9:.1f} nm"


def _fmt_mem(gib: float) -> str:
    """Format a memory estimate, giving up gracefully on absurd ones."""
    if gib >= 1e6:
        return f"{gib/1e9:.0e} EB"
    if gib >= 1e3:
        return f"{gib/1e3:.1f} TB"
    if gib >= 1.0:
        return f"{gib:.1f} GB"
    return f"{gib*1024:.0f} MB"


if __name__ == "__main__":
    raise SystemExit(main())
