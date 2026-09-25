"""Tests for build_network and network_method = "milp" in the pipeline.

Covers:
- build_network returns NetSchema-conformant net_gdf, the report and the
  candidate buildings with connect / connection_status
- network step: connect written back to the full building table, status
  "nicht erreichbar", on_unreachable = "error", one heat source only
- orchestrator: all steps, result summary with milp_* key figures,
  save_outputs() with German labels
- adapter pipe costs (provide_pipe_costs) replace the bundled ones
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

pytest.importorskip("oemof.solph")

from fheat_core import columns as cols  # noqa: E402
from fheat_core.config import FHeatConfig, OptimizationConfig  # noqa: E402
from fheat_core.optimization import STATUS_CONNECTED, STATUS_UNREACHABLE  # noqa: E402
from fheat_core.optimization.network import REPORT_KEY, OptimizationReport, build_network  # noqa: E402
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
def area(buildings_gdf, streets_gdf, parcels_gdf, temperature_series):
    """Conftest area plus building 3 (connect = 0) and building 4 next to a
    street without a route to the source (unreachable). The source lies 10 m
    beside the street: a source exactly on a street vertex is rejected by the
    graph simplification (no single source connection edge)."""
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
    source = gpd.GeoDataFrame({"geometry": [Point(-10, -15)]}, crs=CRS)
    adapter = StubAdapter(buildings, streets, parcels_gdf, source,
                          temperature=temperature_series, holidays={})
    return buildings, streets, source, adapter


def _state(buildings, streets, source):
    return PipelineState(buildings_gdf=buildings, streets_gdf=streets, source_gdf=source, phase=Phase.STATUS)


class TestBuildNetwork:
    def test_returns_net_and_candidates(self, area, milp_cfg):
        buildings, streets, source, adapter = area
        net_gdf, candidates = build_network(buildings, streets, source, milp_cfg, adapter)
        NetSchema.validate(net_gdf)
        assert isinstance(net_gdf.attrs[REPORT_KEY], OptimizationReport)
        assert isinstance(candidates, gpd.GeoDataFrame)
        assert sorted(candidates[cols.BUILDING_ID]) == [0, 1, 2, 4]
        assert set(candidates.columns) == set(buildings.columns) | {cols.CONNECTION_STATUS}

    def test_report(self, area, milp_cfg):
        buildings, streets, source, adapter = area
        net_gdf, _ = build_network(buildings, streets, source, milp_cfg, adapter)
        report = net_gdf.attrs[REPORT_KEY]
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
        assert REPORT_KEY not in state.net_gdf.attrs
        assert state.phase == Phase.NETWORK

    def test_on_unreachable_error(self, area, tmp_path):
        buildings, streets, source, adapter = area
        cfg = FHeatConfig(network_method="milp", output_dir=str(tmp_path),
                          optimization=OptimizationConfig(on_unreachable="error"))
        with pytest.raises(ValueError, match="cannot be reached"):
            network.run(_state(buildings, streets, source), cfg, adapter)

    def test_more_than_one_source_raises(self, area, milp_cfg):
        buildings, streets, _, adapter = area
        sources = gpd.GeoDataFrame({"geometry": [Point(-10, -15), Point(200, -15)]}, crs=CRS)
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
