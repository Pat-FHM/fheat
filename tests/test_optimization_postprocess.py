"""Tests for fheat_core.optimization.postprocess (T8).

Covers:
- exact GLF(n) per built section, DN from calculate_diameter_velocity_loss
  that carries GLF(n) · S, both heat loss columns
- real investment from the pipe costs of the DN, annuity
- model values next to the post-calculation and their deviations
- producer capacity GLF(N) · ΣQ + losses, geometry in flow direction
"""
from __future__ import annotations

import pytest

pytest.importorskip("oemof.solph")

from fheat_core import columns as cols  # noqa: E402
from fheat_core.algorithms.network import (  # noqa: E402
    calculate_diameter_velocity_loss,
    calculate_glf,
    calculate_volumeflow,
)
from fheat_core.config import OptimizationConfig  # noqa: E402
from fheat_core.optimization import HOURS_PER_YEAR, SOURCE_CONNECTION  # noqa: E402
from fheat_core.optimization.energysystem import pipe_annuity, solve_network  # noqa: E402
from fheat_core.optimization.glf_terms import design_loads  # noqa: E402
from fheat_core.optimization.linearize import linearize_pipes, merge_pipe_costs  # noqa: E402
from fheat_core.optimization.postprocess import postprocess  # noqa: E402
from fheat_core.optimization.preprocess import simplify_network  # noqa: E402
from fheat_core.resources import load_pipe_costs, load_pipe_info  # noqa: E402
from fheat_core.schemas import NetSchema  # noqa: E402

from tests.optimization_graphs import CRS, random_case  # noqa: E402

HTEMP, LTEMP = 80.0, 50.0


def _post(seed, soil_temperature=10.0):
    net = simplify_network(*random_case(seed))
    cfg = OptimizationConfig(soil_temperature=soil_temperature)
    pipe_info, pipe_costs = load_pipe_info(), load_pipe_costs()
    lin = linearize_pipes(pipe_info, pipe_costs, design_loads(net), HTEMP, LTEMP, soil_temperature=soil_temperature)
    demand = {k: net.graph.nodes[n]["power"] * 2000 for k, n in net.building_nodes.items()}
    res = solve_network(net, lin, demand, cfg)
    net_gdf, post = postprocess(net, res.edges, lin, pipe_info, pipe_costs, pipe_annuity(cfg), CRS)
    return net, res, net_gdf, post


@pytest.mark.parametrize("seed", range(6))
class TestPostCalculation:
    def test_net_schema_and_built_sections_only(self, seed):
        _, res, net_gdf, _ = _post(seed)
        NetSchema.validate(net_gdf)
        assert len(net_gdf) == int(res.edges["built"].sum())
        for col in NetSchema.optional_columns:
            assert col in net_gdf.columns, col

    def test_t8_exact_glf_and_dn(self, seed):
        _, _, net_gdf, _ = _post(seed)
        pipe_info = load_pipe_info().set_index("DN")
        for _, row in net_gdf.iterrows():
            assert row[cols.GLF] == pytest.approx(calculate_glf(row[cols.N_BUILDINGS]))
            assert row[cols.THERMAL_POWER_GLF] == pytest.approx(row[cols.GLF] * row[cols.THERMAL_POWER])
            vf = calculate_volumeflow(row[cols.THERMAL_POWER_GLF], HTEMP, LTEMP)
            assert row[cols.VOLUME_FLOW] == pytest.approx(vf)
            assert vf <= pipe_info.loc[row[cols.NOMINAL_DIAMETER], "max_volumeFlow"]

    def test_losses_equal_fheat_at_10_degrees(self, seed):
        _, _, net_gdf, _ = _post(seed)
        pipe_info = load_pipe_info()
        for _, row in net_gdf.iterrows():
            dn, vel, loss, loss_extra = calculate_diameter_velocity_loss(
                row[cols.VOLUME_FLOW], HTEMP, LTEMP, row[cols.LENGTH], pipe_info, row[cols.TYPE]
            )
            assert dn == row[cols.NOMINAL_DIAMETER]
            assert row[cols.VELOCITY] == pytest.approx(vel)
            assert row[cols.HEAT_LOSS] == pytest.approx(loss)
            assert row[cols.HEAT_LOSS_EXTRA_INSULATION] == pytest.approx(loss_extra)

    def test_real_costs_from_dn(self, seed):
        _, _, net_gdf, post = _post(seed)
        costs = merge_pipe_costs(load_pipe_info(), load_pipe_costs()).set_index("DN")["cost_eur_per_m"]
        annuity = pipe_annuity(OptimizationConfig())
        for _, row in net_gdf.iterrows():
            assert row[cols.INVEST_COST] == pytest.approx(costs[row[cols.NOMINAL_DIAMETER]] * row[cols.LENGTH])
            assert row[cols.ANNUAL_COST] == pytest.approx(annuity * row[cols.INVEST_COST])
        assert post.invest_cost == pytest.approx(net_gdf[cols.INVEST_COST].sum())
        assert post.annual_cost == pytest.approx(net_gdf[cols.ANNUAL_COST].sum())

    def test_model_values_and_deviation(self, seed):
        _, res, net_gdf, post = _post(seed)
        built = res.edges[res.edges["built"]]
        assert net_gdf[cols.CAPACITY_MODEL].sum() == pytest.approx(built[cols.CAPACITY_MODEL].sum())
        assert net_gdf[cols.GLF_MODEL].sum() == pytest.approx(built[cols.GLF_MODEL].sum())
        assert post.invest_cost_model == pytest.approx(built[cols.INVEST_COST_MODEL].sum())
        assert post.cost_line_deviation == pytest.approx(post.invest_cost_model / post.invest_cost - 1)
        assert post.glf_max_deviation == pytest.approx(
            (net_gdf[cols.GLF_MODEL] / net_gdf[cols.GLF] - 1).abs().max()
        )

    def test_producer_capacity(self, seed):
        net, _, net_gdf, post = _post(seed)
        powers = [net.graph.nodes[n]["power"] for n in net.building_nodes.values()]
        loss_kw = net_gdf[cols.HEAT_LOSS].sum() / HOURS_PER_YEAR
        assert post.heat_loss == pytest.approx(loss_kw)
        assert post.producer_capacity == pytest.approx(calculate_glf(len(powers)) * sum(powers) + loss_kw)

    def test_geometry_in_flow_direction(self, seed):
        net, res, net_gdf, _ = _post(seed)
        built = res.edges[res.edges["built"]].reset_index(drop=True)
        for geom, flow_from, flow_to in zip(net_gdf.geometry, built["flow_from"], built["flow_to"], strict=True):
            assert geom.coords[0] == net.node_coords[flow_from]
            assert geom.coords[-1] == net.node_coords[flow_to]


class TestSoilTemperature:
    def test_warmer_soil_lowers_losses(self):
        _, _, cold, _ = _post(1, soil_temperature=10.0)
        _, _, warm, _ = _post(1, soil_temperature=15.0)
        assert warm[cols.HEAT_LOSS].sum() < cold[cols.HEAT_LOSS].sum()
        src = warm[warm[cols.TYPE] == SOURCE_CONNECTION]
        assert len(src) == 1
