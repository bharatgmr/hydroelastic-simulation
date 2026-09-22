"""The config's job is to refuse to invent numbers, so that is what gets tested."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from grind_sim.config import GrindConfig, load_config, load_materials, missing_values

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_config_loads():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        config = load_config(CONFIGS / "grind_v1.yaml")
    assert config.pad.outer_diameter_m == pytest.approx(0.1778)
    assert config.sheet.material == "carbon_steel_a36"
    assert config.contact.reduce_contacts is False  # pressure maps need the full surface


def test_shipped_config_warns_about_every_unmeasured_field():
    with pytest.warns(RuntimeWarning, match="TODO_MEASURE") as record:
        load_config(CONFIGS / "grind_v1.yaml")
    message = str(record[0].message)
    for expected in ("pad.modulus_pa", "pad.ribs.n_ribs", "sheet.support_span_m"):
        assert expected in message


def test_missing_values_walks_nested_models():
    config = GrindConfig()
    gaps = missing_values(config)
    assert "pad.ribs.rib_width_m" in gaps
    assert "pad.modulus_pa" in gaps
    assert "contact.narrow_band_m" not in gaps  # has a real default


def test_materials_load_and_flag_their_gaps():
    with pytest.warns(RuntimeWarning, match="unmeasured material properties"):
        materials = load_materials(CONFIGS / "materials.yaml")
    assert set(materials) == {"aluminum_6061", "carbon_steel_a36", "stainless_304"}
    steel = materials["carbon_steel_a36"]
    assert steel.E == pytest.approx(200.0e9)
    assert steel.mu_vs_abrasive is None
    assert steel.preston_k is None


def test_material_foundation_stiffness_and_cap():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        steel = load_materials(CONFIGS / "materials.yaml")["carbon_steel_a36"]
    uncapped = steel.foundation_stiffness(0.003)
    assert uncapped == pytest.approx(200.0e9 / (0.003 * (1 - 0.26**2)))
    assert steel.foundation_stiffness(0.003, cap=1.0e9) == 1.0e9


def test_spindle_speed_is_clamped_to_the_plate_rating():
    config = GrindConfig.model_validate({"motion": {"spindle_rpm": 12000.0}})
    assert config.spindle_rpm_clamped() == 8600.0


def test_config_is_frozen():
    config = GrindConfig()
    with pytest.raises(Exception):
        config.pad = None
