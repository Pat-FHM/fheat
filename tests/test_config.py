"""Tests for fheat_core.config.FHeatConfig.

Covers:
- default values are sane
- supply > return temperature invariant is enforced
- output_format whitelist is enforced
- valid configurations construct successfully
"""
from __future__ import annotations

import pytest

from fheat_core.config import FHeatConfig


class TestDefaults:
    def test_default_construction_succeeds(self):
        cfg = FHeatConfig()
        assert cfg.supply_temperature == 80.0
        assert cfg.return_temperature == 50.0
        assert cfg.wld_threshold == 500.0
        assert cfg.buffer_distance == 50.0
        assert cfg.building_class == 3
        assert cfg.wind_class == 1
        assert cfg.year == 2022
        assert cfg.output_dir == "./output"
        assert cfg.output_format == "gpkg"

    def test_supply_strictly_greater_than_return(self):
        cfg = FHeatConfig()
        assert cfg.supply_temperature > cfg.return_temperature


class TestTemperatureInvariant:
    def test_supply_equal_return_raises(self):
        with pytest.raises(ValueError, match="supply_temperature"):
            FHeatConfig(supply_temperature=70.0, return_temperature=70.0)

    def test_supply_below_return_raises(self):
        with pytest.raises(ValueError, match="supply_temperature"):
            FHeatConfig(supply_temperature=40.0, return_temperature=60.0)

    def test_supply_just_above_return_succeeds(self):
        cfg = FHeatConfig(supply_temperature=60.001, return_temperature=60.0)
        assert cfg.supply_temperature > cfg.return_temperature


class TestOutputFormat:
    @pytest.mark.parametrize("fmt", ["gpkg", "fgb", "geojson", "gml"])
    def test_allowed_formats_succeed(self, fmt):
        cfg = FHeatConfig(output_format=fmt)
        assert cfg.output_format == fmt

    @pytest.mark.parametrize("fmt", ["shp", "csv", "GPKG", "", "json"])
    def test_disallowed_formats_raise(self, fmt):
        with pytest.raises(ValueError, match="output_format"):
            FHeatConfig(output_format=fmt)


class TestCustomConfig:
    def test_full_custom_config(self, tmp_path):
        cfg = FHeatConfig(
            supply_temperature=95.0,
            return_temperature=65.0,
            wld_threshold=750.0,
            buffer_distance=25.0,
            building_class=2,
            wind_class=0,
            year=2030,
            output_dir=str(tmp_path),
            output_format="geojson",
        )
        assert cfg.supply_temperature == 95.0
        assert cfg.year == 2030
        assert cfg.output_dir == str(tmp_path)


class TestOptimizationConfig:
    def test_defaults(self):
        from fheat_core.config import OptimizationConfig

        cfg = OptimizationConfig()
        assert (cfg.interest_rate, cfg.lifetime_pipes) == (0.08, 20)
        assert (cfg.source_capex_eur_per_kw, cfg.lifetime_source) == (598.0, 20)
        assert cfg.heat_cost_eur_per_kwh == 0.08
        assert (cfg.mip_rel_gap, cfg.mip_abs_gap, cfg.time_limit_s) == (0.01, "auto", 300.0)

    @pytest.mark.parametrize("kwargs, match", [
        ({"interest_rate": 0.0}, "interest_rate"),
        ({"lifetime_pipes": 0}, "lifetime_pipes"),
        ({"lifetime_source": -1}, "lifetime_source"),
        ({"time_limit_s": 0}, "time_limit_s"),
        ({"source_capex_eur_per_kw": -1.0}, "source_capex_eur_per_kw"),
        ({"heat_cost_eur_per_kwh": -0.1}, "heat_cost_eur_per_kwh"),
        ({"mip_rel_gap": -0.01}, "mip_rel_gap"),
        ({"mip_abs_gap": "fast"}, "mip_abs_gap"),
        ({"mip_abs_gap": -5.0}, "mip_abs_gap"),
    ])
    def test_invalid_values_raise(self, kwargs, match):
        from fheat_core.config import OptimizationConfig

        with pytest.raises(ValueError, match=match):
            OptimizationConfig(**kwargs)

    def test_numeric_abs_gap(self):
        from fheat_core.config import OptimizationConfig

        assert OptimizationConfig(mip_abs_gap=250.0).mip_abs_gap == 250.0
