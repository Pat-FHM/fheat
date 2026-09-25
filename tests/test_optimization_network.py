"""Tests for build_network and network_method = "milp" in the pipeline.

Covers:
- build_network returns NetworkResult: NetSchema-conformant net_gdf, the
  candidate buildings with connect / connection_status and the report; the
  conftest source lies on a street vertex
- network step: connect written back to the full building table, status
  "nicht erreichbar", on_unreachable = "error", one heat source only
- orchestrator: all steps, result summary with milp_* key figures,
  save_outputs() with German labels
- adapter pipe costs (provide_pipe_costs) replace the bundled ones
- T9 economic mode: a far, small building gets connect = 0 and the status
  "wirtschaftlich nicht angeschlossen"
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

pytest.importorskip("oemof.solph")

from fheat_core import columns as cols  # noqa: E402
from fheat_core.config import FHeatConfig, OptimizationConfig  # noqa: E402
from fheat_core.optimization import STATUS_CONNECTED, STATUS_NOT_ECONOMIC, STATUS_UNREACHABLE  # noqa: E402
from fheat_core.optimization.network import NetworkResult, OptimizationReport, build_network  # noqa: E402
from fheat_core.orchestrator import FHeatOrchestrator  # noqa: E402
from fheat_core.resources import load_pipe_costs  # noqa: E402
from fheat_core.schemas import NetSchema  # noqa: E402
from fheat_core.state import Phase, PipelineState  # noqa: E402
from fheat_core.steps import network  # noqa: E402

from tests.conftest import CRS, StubAdapter  # noqa: E402


@pytest.fixture
def milp_cfg(tmp_path):
    return FHeatConfig(network_method="milp", output_dir=str(tmp_path), buffer_distance=15.0)


@pytest.fixture
def area(buildings_gdf, streets_gdf, parcels_gdf, source_gdf, temperature_series):
    """Conftest area plus building 3 (connect = 0) and building 4 next to a
    street without a route to the source (unreachable). The conftest source
    lies exactly on the first street vertex."""
    extra = gpd.GeoDataFrame(
        {
            cols.BUILDING_ID: [3, 4],
            cols.CONNECT: [0, 1],
            cols.HEAT_DEMAND: [20000.0, 30000.0],
            cols.THERMAL_POWER: [12.0, 18.0],
            cols.FULL_LOAD_HOURS: [1666.0, 1666.0],
            cols.LOAD_PROFILE: ["EFH", "EFH"],
            "geometry": [
                Polygon([(150, 0), (160, 0), (160, 10), (150, 10)]),
                Polygon([(545, 505), (555, 505), (555, 515), (545, 515)]),
            ],
        },
        crs=CRS,
    )
    buildings = gpd.GeoDataFrame(pd.concat([buildings_gdf, extra], ignore_index=True), crs=CRS)
    streets = gpd.GeoDataFrame(
        {cols.ROUTABLE: [1, 1], "geometry": [streets_gdf.geometry.iloc[0], LineString([(500, 500), (600, 500)])]},
        crs=CRS,
    )
    adapter = StubAdapter(buildings, streets, parcels_gdf, source_gdf,
                          temperature=temperature_series, holidays={})
    return buildings, streets, source_gdf, adapter


def _state(buildings, streets, source):
    return PipelineState(buildings_gdf=buildings, streets_gdf=streets, source_gdf=source, phase=Phase.STATUS)


class TestBuildNetwork:
    def test_returns_network_result(self, area, milp_cfg):
        buildings, streets, source, adapter = area
        result = build_network(buildings, streets, source, milp_cfg, adapter)
        assert isinstance(result, NetworkResult)
        net_gdf, candidates, report = result
        NetSchema.validate(net_gdf)
        assert isinstance(report, OptimizationReport)
        assert not net_gdf.attrs
        assert isinstance(candidates, gpd.GeoDataFrame)
        assert sorted(candidates[cols.BUILDING_ID]) == [0, 1, 2, 4]
        assert set(candidates.columns) == set(buildings.columns) | {cols.CONNECTION_STATUS}

    def test_report(self, area, milp_cfg):
        buildings, streets, source, adapter = area
        report = build_network(buildings, streets, source, milp_cfg, adapter).report
        assert report.unreachable == [4]
        assert report.milp.termination == "optimal"
        assert {"quantity", "edge_type", "DN", "deviation", "r_squared"} <= set(report.fit_quality.columns)
        assert report.simplification.unreachable_buildings == 1
        assert report.post.producer_capacity == pytest.approx(report.milp.source_capacity, rel=0.2)
        summary = report.summary()
        parts = [summary[f"milp_cost_{k}_eur_a"] for k in ("source", "heat", "losses", "pipes")]
        assert sum(parts) == pytest.approx(summary["milp_objective_eur_a"], abs=1.0)
        assert summary["milp_fit_warnings"] == len(report.fit_warnings)


class TestNetworkStep:
    def test_connect_and_status_written_back(self, area, milp_cfg):
        buildings, streets, source, adapter = area
        state = network.run(_state(buildings, streets, source), milp_cfg, adapter)
        b = state.buildings_gdf.set_index(cols.BUILDING_ID)
        assert list(b.loc[[0, 1, 2], cols.CONNECTION_STATUS]) == [STATUS_CONNECTED] * 3
        assert list(b.loc[[0, 1, 2], cols.CONNECT]) == [1, 1, 1]
        assert (b.loc[4, cols.CONNECT], b.loc[4, cols.CONNECTION_STATUS]) == (0, STATUS_UNREACHABLE)
        assert b.loc[3, cols.CONNECT] == 0 and pd.isna(b.loc[3, cols.CONNECTION_STATUS])
        assert len(state.buildings_gdf) == len(buildings)

    def test_report_moved_to_state(self, area, milp_cfg):
        buildings, streets, source, adapter = area
        state = network.run(_state(buildings, streets, source), milp_cfg, adapter)
        assert isinstance(state.optimization_report, OptimizationReport)
        assert state.phase == Phase.NETWORK

    def test_on_unreachable_error(self, area, tmp_path):
        buildings, streets, source, adapter = area
        cfg = FHeatConfig(network_method="milp", output_dir=str(tmp_path),
                          optimization=OptimizationConfig(on_unreachable="error"))
        with pytest.raises(ValueError, match="cannot be reached"):
            network.run(_state(buildings, streets, source), cfg, adapter)

    def test_more_than_one_source_raises(self, area, milp_cfg):
        buildings, streets, _, adapter = area
        sources = gpd.GeoDataFrame({"geometry": [Point(-10, -5), Point(200, -5)]}, crs=CRS)
        with pytest.raises(ValueError, match="exactly one heat source"):
            network.run(_state(buildings, streets, sources), milp_cfg, adapter)

    def test_adapter_pipe_costs_are_used(self, area, milp_cfg):
        buildings, streets, source, adapter = area

        class CostlyAdapter(StubAdapter):
            def provide_pipe_costs(self):
                costs = load_pipe_costs()
                costs["cost_eur_per_m"] *= 2
                return costs

        costly = CostlyAdapter(buildings, streets, adapter.fetch_parcels(), source)
        base = network.run(_state(buildings, streets, source), milp_cfg, adapter).optimization_report
        double = network.run(_state(buildings, streets, source), milp_cfg, costly).optimization_report
        assert double.post.invest_cost == pytest.approx(2 * base.post.invest_cost)


class TestOrchestrator:
    def test_full_run_summary_and_outputs(self, area, milp_cfg, tmp_path):
        *_, adapter = area
        orch = FHeatOrchestrator(config=milp_cfg, adapter=adapter)
        orch.run_all()
        summary = orch.state.result_summary
        assert summary["total_buildings"] == 3
        assert summary["milp_unreachable_buildings"] == 1
        assert summary["milp_termination"] == "optimal"
        for key in ("milp_producer_capacity_kw", "milp_pipe_invest_eur", "milp_cost_line_deviation",
                    "milp_pipe_annuity_eur_a", "milp_glf_max_deviation", "milp_gap_eur_a",
                    "milp_capacity_max_deviation", "milp_sections_above_largest_dn"):
            assert summary[key] is not None, key

        saved = orch.save_outputs()
        netz = gpd.read_file(saved["netz"])
        for label in ("GLF_Modell", "Kapazitaet_Modell [kW]", "Investition [EUR]", "Annuitaet [EUR/a]"):
            assert label in netz.columns, label
        gebaeude = gpd.read_file(saved["buildings"])
        assert "Anschlussstatus" in gebaeude.columns
        assert (gebaeude["Anschlussstatus"] == STATUS_UNREACHABLE).sum() == 1

    def test_shortest_path_stays_default(self, area, tmp_path):
        *_, adapter = area
        orch = FHeatOrchestrator(config=FHeatConfig(output_dir=str(tmp_path)), adapter=adapter)
        orch.run_all()
        assert orch.state.optimization_report is None
        assert cols.CONNECTION_STATUS not in orch.state.buildings_gdf.columns
        assert not any(k.startswith("milp_") for k in orch.state.result_summary)


@pytest.fixture
def t9_area(parcels_gdf, source_gdf):
    """Three large buildings (300 MWh/a) at the start of the street, one small
    building (10 MWh/a) at the end of a 2.8 km extension."""
    buildings = gpd.GeoDataFrame(
        {
            cols.BUILDING_ID: [0, 1, 2, 3],
            cols.CONNECT: [1, 1, 1, 1],
            cols.HEAT_DEMAND: [300000.0, 300000.0, 300000.0, 10000.0],
            cols.THERMAL_POWER: [150.0, 150.0, 150.0, 5.0],
            cols.FULL_LOAD_HOURS: [2000.0] * 4,
            cols.LOAD_PROFILE: ["MFH"] * 4,
            "geometry": [Polygon([(x, 0), (x + 10, 0), (x + 10, 10), (x, 10)]) for x in (0, 50, 100, 2995)],
        },
        crs=CRS,
    )
    streets = gpd.GeoDataFrame(
        {cols.ROUTABLE: [1, 1], "geometry": [LineString([(-10, -5), (200, -5)]), LineString([(200, -5), (3000, -5)])]},
        crs=CRS,
    )
    return buildings, streets, source_gdf, StubAdapter(buildings, streets, parcels_gdf, source_gdf)


class TestT9EconomicStep:
    @staticmethod
    def _cfg(tmp_path):
        return FHeatConfig(
            network_method="milp", output_dir=str(tmp_path),
            optimization=OptimizationConfig(mode="wirtschaftlich", heat_price_eur_per_kwh=0.15),
        )

    def test_connect_zero_written_back(self, t9_area, tmp_path):
        buildings, streets, source, adapter = t9_area
        state = network.run(_state(buildings, streets, source), self._cfg(tmp_path), adapter)
        b = state.buildings_gdf.set_index(cols.BUILDING_ID)
        assert list(b[cols.CONNECT]) == [1, 1, 1, 0]
        assert list(b[cols.CONNECTION_STATUS]) == [STATUS_CONNECTED] * 3 + [STATUS_NOT_ECONOMIC]
        assert state.optimization_report.not_connected_economic == [3]

    def test_summary(self, t9_area, tmp_path):
        buildings, streets, source, adapter = t9_area
        state = network.run(_state(buildings, streets, source), self._cfg(tmp_path), adapter)
        summary = state.optimization_report.summary()
        assert summary["milp_mode"] == "wirtschaftlich"
        assert (summary["milp_connected_buildings"], summary["milp_not_connected_economic"]) == (3, 1)
        assert summary["milp_revenue_eur_a"] == pytest.approx(0.15 * 900000.0, rel=1e-6)
        assert summary["milp_glf_estimated_sections"] >= 1

    def test_forced_mode_connects_all(self, t9_area, milp_cfg):
        buildings, streets, source, adapter = t9_area
        state = network.run(_state(buildings, streets, source), milp_cfg, adapter)
        assert list(state.buildings_gdf[cols.CONNECT]) == [1, 1, 1, 1]
        assert state.optimization_report.summary()["milp_revenue_eur_a"] == 0.0

    def test_nothing_pays_gives_empty_network(self, t9_area, tmp_path, caplog):
        buildings, streets, source, adapter = t9_area
        cfg = FHeatConfig(
            network_method="milp", output_dir=str(tmp_path),
            optimization=OptimizationConfig(mode="wirtschaftlich", heat_price_eur_per_kwh=0.0),
        )
        orch = FHeatOrchestrator(config=cfg, adapter=adapter)
        with caplog.at_level("WARNING", logger="fheat_core.optimization.network"):
            orch.run_all()
        assert orch.state.net_gdf.empty
        NetSchema.validate(orch.state.net_gdf)
        assert (orch.state.buildings_gdf[cols.CONNECTION_STATUS] == STATUS_NOT_ECONOMIC).all()
        assert (orch.state.buildings_gdf[cols.CONNECT] == 0).all()
        summary = orch.state.result_summary
        assert summary["milp_connected_buildings"] == 0
        assert summary["milp_cost_line_deviation"] is None and summary["milp_glf_max_deviation"] is None
        assert any("no building pays" in r.getMessage() for r in caplog.records)
        assert "netz" not in orch.save_outputs()
