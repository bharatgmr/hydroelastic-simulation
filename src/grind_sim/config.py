"""Typed configuration for the grind-contact runs.

Frozen pydantic models over a YAML file, in the style of sanding-wm's ``config/engine.py``. The
point of the types here is not ceremony: several fields are genuinely unmeasured (rib geometry,
pad modulus, friction, Preston k), and this module makes a run say so at startup instead of
silently standing on an invented number.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

__all__ = [
    "GrindConfig",
    "MaterialSpec",
    "load_config",
    "load_materials",
    "missing_values",
]


class RibSpec(BaseModel, frozen=True):
    """Rib pattern on the pad underside. Every field is unmeasured; ribs are not built in v1."""

    n_ribs: int | None = None
    rib_width_m: float | None = None
    rib_height_m: float | None = None
    rib_pattern: Literal["radial", "concentric"] | None = None
    rib_inner_radius_m: float | None = None
    rib_outer_radius_m: float | None = None


class FiberDiscSpec(BaseModel, frozen=True):
    thickness_m: float = 0.0008
    separate_layer: bool = False


class PadConfig(BaseModel, frozen=True):
    outer_diameter_m: float = 0.1778
    bore_diameter_m: float = 0.0222
    thickness_m: float = 0.0127
    mass_kg: float | None = None
    modulus_pa: float | None = None
    variant: Literal["flat", "ribbed"] = "flat"
    ribs: RibSpec = Field(default_factory=RibSpec)
    fiber_disc: FiberDiscSpec = Field(default_factory=FiberDiscSpec)


class SheetConfig(BaseModel, frozen=True):
    side_m: float = 0.300
    thickness_m: float = 0.003
    material: str = "carbon_steel_a36"
    support: Literal["rigid_backing", "clamped_edges", "simply_supported"] = "rigid_backing"
    support_span_m: float | None = None


class ContactConfig(BaseModel, frozen=True):
    pad_voxel_m: float | None = None
    sheet_voxel_m: float | None = None
    narrow_band_m: float = 0.002
    texture_format: Literal["uint8", "uint16", "float32"] = "float32"
    contact_margin_m: float = 0.0
    reduce_contacts: bool = False
    mc_edge_clamp_min: float = 0.0
    memory_budget_gb: float = 8.0


class MotionConfig(BaseModel, frozen=True):
    tilt_deg: float = 5.0
    tilt_azimuth_deg: float = 0.0
    force_setpoints_n: list[float] = Field(default_factory=lambda: [10.0, 25.0, 30.0])
    approach_speed_m_s: float = 0.002
    contact_threshold_n: float = 1.0
    ramp_s: float = 0.5
    hold_s: float = 1.0
    spindle_rpm: float = 5000.0
    traverse_speed_m_s: float = 0.05
    traverse_dist_m: float = 0.05


class SolverConfig(BaseModel, frozen=True):
    dt: float = 1.0 / 240.0
    substeps: int = 2
    cone: Literal["pyramidal", "elliptic"] = "elliptic"
    max_contacts: int = 200_000


class GrindConfig(BaseModel, frozen=True):
    """Whole-run configuration."""

    pad: PadConfig = Field(default_factory=PadConfig)
    sheet: SheetConfig = Field(default_factory=SheetConfig)
    contact: ContactConfig = Field(default_factory=ContactConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)

    # 3M's rating for the 80514 plate; commanding past it is a modelling error, not a choice.
    MAX_SPINDLE_RPM: float = 8600.0

    def spindle_rpm_clamped(self) -> float:
        """Spindle speed, clamped to the plate's rated maximum."""
        return min(self.motion.spindle_rpm, self.MAX_SPINDLE_RPM)


class MaterialSpec(BaseModel, frozen=True):
    """One workpiece material. The nulls are unmeasured, not defaults."""

    E: float
    nu: float
    rho: float
    mu_vs_abrasive: float | None = None
    hardness_HB: float | None = None
    preston_k: float | None = None

    def foundation_stiffness(self, thickness: float, *, cap: float | None = None) -> float:
        """Winkler stiffness ``E / (h * (1 - nu^2))`` [N/m^3], optionally capped.

        Args:
            thickness: Layer thickness [m].
            cap: Upper bound for numerical conditioning, e.g. 100x the pad stiffness.
        """
        k = self.E / (thickness * (1.0 - self.nu**2))
        return min(k, cap) if cap is not None else k


def missing_values(model: BaseModel, prefix: str = "") -> list[str]:
    """List dotted paths of every field that is ``None``, recursing into nested models."""
    missing: list[str] = []
    for name, value in model:
        path = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            missing.extend(missing_values(value, f"{path}."))
        elif value is None:
            missing.append(path)
    return missing


def load_config(path: str | Path, *, warn: bool = True) -> GrindConfig:
    """Load and validate a run config, warning about unmeasured fields.

    Args:
        path: YAML file, e.g. ``configs/grind_v1.yaml``.
        warn: Emit a ``RuntimeWarning`` naming every null field.

    Returns:
        The validated config.
    """
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    config = GrindConfig.model_validate(data)
    if warn:
        gaps = missing_values(config)
        if gaps:
            warnings.warn(
                "unmeasured config values (TODO_MEASURE), not defaults: " + ", ".join(gaps),
                RuntimeWarning,
                stacklevel=2,
            )
    return config


def load_materials(path: str | Path, *, warn: bool = True) -> dict[str, MaterialSpec]:
    """Load ``materials.yaml`` into validated specs, warning about unmeasured properties."""
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    materials = {name: MaterialSpec.model_validate(spec) for name, spec in data.items()}
    if warn:
        gaps = [f"{name}.{field}" for name, spec in materials.items()
                for field in missing_values(spec)]
        if gaps:
            warnings.warn(
                "unmeasured material properties, left null on purpose: " + ", ".join(gaps),
                RuntimeWarning,
                stacklevel=2,
            )
    return materials
