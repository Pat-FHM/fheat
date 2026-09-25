"""Tests for fheat_core.optimization.block and .energysystem (forced mode, no GLF).

Covers:
- T3 no double use: one flow direction per built section, one capacity and
  hence one DN per built section, no parallel sections
- T4 radial network: the built sections form an arborescence from the source
- T5 forced mode: every reachable building is connected, unreachable ones
  are reported
- T7 energy balance: producer = demand + losses; producer capacity covers
  the source connection plus losses
- objective terms, fixed bridges, mip_abs_gap "auto", time limit, errors
"""
from __future__ import annotations

import math

import networkx as nx
import pytest

pytest.importorskip("oemof.solph")

from oemof.tools import economics  # noqa: E402

from fheat_core import columns as cols  # noqa: E402
from fheat_core.algorithms.network import (  # noqa: E402
    calculate_diameter_velocity_loss,
    calculate_volumeflow,
)
from fheat_core.config import OptimizationConfig  # noqa: E402
from fheat_core.optimization import HOUSE_CONNECTION  # noqa: E402
from fheat_core.optimization.energysystem import (  # noqa: E402
    HOURS_PER_YEAR,
    MIP_ABS_GAP_AUTO_SHARE,
    solve_network,
)
from fheat_core.optimization.glf_terms import reference_tree  # noqa: E402
from fheat_core.optimization.linearize import linearize_pipes  # noqa: E402
from fheat_core.optimization.preprocess import POWER, simplify_network  # noqa: E402
from fheat_core.resources import load_pipe_costs, load_pipe_info  # noqa: E402

from tests.optimization_graphs import grid_case, random_case, street_graph  # noqa: E402

FULL_LOAD_HOURS = 2000.0
HTEMP, LTEMP = 80.0, 50.0


def _solve(G, b, s, **config):
    net = simplify_network(G, b, s)
    powers = [net.graph.nodes[n][POWER] for n in net.building_nodes.values()]
    lin = linearize_pipes(load_pipe_info(), load_pipe_costs(), powers, HTEMP, LTEMP)
    demand = {k: b.at[k, cols.THERMAL_POWER] * FULL_LOAD_HOURS for k in b.index}
    return net, lin, solve_network(net, lin, demand, OptimizationConfig(**config))


def _built_tree(res) -> nx.DiGraph:
    built = res.edges[res.edges["built"]]
    return nx.DiGraph(list(zip(built["flow_from"], built["flow_to"], strict=True)))


@pytest.fixture(scope="module")
def mesh():
    """Mesh A(0,0)-B(10,0)-C(10,10)-D(0,10) with a spur at D.

    A-B, B-C 10 m; A-D, D-C 50 m. Source at A, building 0 at C (30 kW),
    building 1 at the end of the spur D-(−10,20) (20 kW). Optimal: A-B-C and A-D.
    """
    dx = math.sqrt(25 ** 2 - 5 ** 2)
    G, b, s = street_graph(
        streets=[
            [(0, 0), (10, 0)], [(10, 0), (10, 10)],
            [(0, 0), (-dx, 5), (0, 10)], [(0, 10), (5, 10 + dx), (10, 10)],
            [(0, 10), (-10, 20)],
        ],
        buildings=[((15, 10), (10, 10), 30.0), ((-10, 25), (-10, 20), 20.0)],
        sources=[((-5, 0), (0, 0))],
    )
    net, lin, res = _solve(G, b, s)
    return net, lin, res


def _random_solved(seed):
    return _solve(*random_case(seed))


# ---------------------------------------------------------------------------
# optimum of a small mesh
# ---------------------------------------------------------------------------


class TestMeshOptimum:
    def test_optimal(self, mesh):
        _, _, res = mesh
        assert res.termination == "optimal"
        assert res.best_bound == pytest.approx(res.objective, rel=OptimizationConfig().mip_rel_gap)

    def test_runtimes_reported(self, mesh):
        _, _, res = mesh
        assert 0 < res.build_time_s < 60
        assert 0 < res.solve_time_s < 60

    def test_route(self, mesh):
        net, _, res = mesh
        built = res.edges[res.edges["built"] & (res.edges[cols.TYPE] == "Straßenleitung")]
        coords = {frozenset((net.node_coords[u], net.node_coords[v])) for u, v in zip(built["u"], built["v"], strict=True)}
        assert coords == {
            frozenset(((0, 0), (10, 10))),     # A-B-C merged, 20 m
            frozenset(((0, 0), (0, 10))),      # A-D, 50 m
            frozenset(((0, 10), (-10, 20))),   # spur
        }

    def test_capacity_equals_power_without_glf(self, mesh):
        _, _, res = mesh
        built = res.edges[res.edges["built"]]
        assert built["capacity"].to_numpy() == pytest.approx(built[cols.THERMAL_POWER].to_numpy())
        src = res.edges[res.edges[cols.TYPE] == "Quellenanschluss"].iloc[0]
        assert (src["capacity"], src[cols.N_BUILDINGS]) == (pytest.approx(50.0), pytest.approx(2))

    def test_objective_parts(self, mesh):
        _, lin, res = mesh
        cfg = OptimizationConfig()
        parts = res.objective_parts
        assert parts.total == pytest.approx(res.objective)
        pipe_annuity = economics.annuity(1.0, cfg.lifetime_pipes, cfg.interest_rate)
        assert parts.pipes == pytest.approx(pipe_annuity * res.edges["invest_cost"].sum())
        ep_costs = economics.annuity(cfg.source_capex_eur_per_kw, cfg.lifetime_source, cfg.interest_rate)
        assert parts.source_invest == pytest.approx(ep_costs * res.source_capacity)
        heat_cost = cfg.heat_cost_eur_per_kwh * HOURS_PER_YEAR
        assert parts.heat_demand == pytest.approx(heat_cost * res.demand_flow)
        assert parts.heat_losses == pytest.approx(heat_cost * res.loss_flow)

    def test_invest_cost_of_built_sections(self, mesh):
        _, lin, res = mesh
        for _, row in res.edges.iterrows():
            expected = lin.invest_cost(row[cols.TYPE], row[cols.LENGTH], row["capacity"], int(row["built"]))
            assert row["invest_cost"] == pytest.approx(expected, abs=1e-6)

    def test_mip_abs_gap_auto(self, mesh):
        net, lin, res = mesh
        cfg = OptimizationConfig()
        tree = reference_tree(net)
        invest = sum(
            lin.invest_cost(net.graph.edges[e][cols.TYPE], net.graph.edges[e][cols.LENGTH], s0, 1)
            for e, s0 in tree.s0.items()
        )
        pipe_annuity = economics.annuity(1.0, cfg.lifetime_pipes, cfg.interest_rate)
        assert res.mip_abs_gap == pytest.approx(MIP_ABS_GAP_AUTO_SHARE * pipe_annuity * invest)

    def test_numeric_mip_abs_gap_passed_through(self):
        G, b, s = random_case(3)
        _, _, res = _solve(G, b, s, mip_abs_gap=123.0)
        assert res.mip_abs_gap == 123.0


# ---------------------------------------------------------------------------
# T3, T4, T5, T7 on random graphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
class TestRandomNetworks:
    def test_t3_one_direction_one_dn(self, seed):
        net, lin, res = _random_solved(seed)
        e = res.edges
        assert not e.duplicated(["u", "v"]).any()
        assert not net.graph.is_multigraph()
        assert (e["built"] == e["flow_from"].notna()).all()
        assert (e.loc[~e["built"], "capacity"].abs() < 1e-6).all()
        pipe_info = load_pipe_info()
        for _, row in e[e["built"]].iterrows():
            assert {row["flow_from"], row["flow_to"]} == {row["u"], row["v"]}
            vf = calculate_volumeflow(row["capacity"], HTEMP, LTEMP)
            dn, *_ = calculate_diameter_velocity_loss(vf, HTEMP, LTEMP, row[cols.LENGTH], pipe_info, row[cols.TYPE])
            assert dn in set(pipe_info["DN"])

    def test_t4_arborescence(self, seed):
        net, _, res = _random_solved(seed)
        tree = _built_tree(res)
        assert nx.is_arborescence(tree)
        assert [n for n, d in tree.in_degree() if d == 0] == [net.source_node()]

    def test_t5_all_reachable_buildings_connected(self, seed):
        net, _, res = _random_solved(seed)
        tree = _built_tree(res)
        assert set(net.building_nodes.values()) <= set(tree.nodes)
        houses = res.edges[res.edges[cols.TYPE] == HOUSE_CONNECTION]
        assert houses["built"].all() and houses["fixed"].all()
        assert set(net.unreachable_buildings).isdisjoint(net.building_nodes)

    def test_t7_energy_balance(self, seed):
        net, lin, res = _random_solved(seed)
        assert res.producer_flow == pytest.approx(res.demand_flow + res.loss_flow)
        powers = sum(net.graph.nodes[n][POWER] for n in net.building_nodes.values())
        assert res.demand_flow == pytest.approx(powers * FULL_LOAD_HOURS / HOURS_PER_YEAR)
        assert res.loss_flow == pytest.approx(res.edges["heat_loss"].sum())
        src = res.edges[res.edges[cols.TYPE] == "Quellenanschluss"].iloc[0]
        assert res.source_capacity == pytest.approx(src["capacity"] + res.loss_flow)
        assert src["capacity"] == pytest.approx(powers)

    def test_fixed_bridges(self, seed):
        net, _, res = _random_solved(seed)
        bridges = {tuple(sorted(e)) for e in net.bridges()}
        fixed = {(u, v) for u, v in res.edges.loc[res.edges["fixed"], ["u", "v"]].itertuples(index=False)}
        assert fixed == bridges
        assert res.edges.loc[res.edges["fixed"], "built"].all()


# ---------------------------------------------------------------------------
# solver settings and errors
# ---------------------------------------------------------------------------


class TestSolver:
    @pytest.mark.slow
    def test_time_limit_is_kept(self, caplog):
        """A grid without gap tolerance cannot be solved in 2 s; HiGHS stops at 2 s."""
        G, b, s = grid_case(5, 100, seed=1)
        with caplog.at_level("WARNING", logger="fheat_core.optimization.energysystem"):
            _, _, res = _solve(G, b, s, mip_rel_gap=0.0, mip_abs_gap=0.0, time_limit_s=2.0)
        assert res.termination == "maxTimeLimit"
        assert res.solve_time_s < 2.0 + 1.5
        assert res.best_bound < res.objective
        assert nx.is_arborescence(_built_tree(res))
        assert any("maxTimeLimit" in r.getMessage() for r in caplog.records)

    def test_two_sources_raise(self):
        G, b, s = street_graph(
            streets=[[(0, 0), (50, 0), (100, 0)]],
            buildings=[((50, 10), (50, 0), 10.0)],
            sources=[((-10, 0), (0, 0)), ((110, 0), (100, 0))],
        )
        with pytest.raises(ValueError, match="exactly one heat source"):
            _solve(G, b, s)

    def test_section_above_largest_dn_raises(self):
        G, b, s = street_graph(
            streets=[[(0, 0), (50, 0)]],
            buildings=[((50, 10), (50, 0), 1e6)],
            sources=[((-10, 0), (0, 0))],
        )
        with pytest.raises(ValueError, match="more than the largest DN"):
            _solve(G, b, s)
