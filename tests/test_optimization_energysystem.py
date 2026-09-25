"""Tests for fheat_core.optimization.block and .energysystem (forced mode).

Covers:
- T3 no double use: one flow direction per built section, one capacity and
  hence one DN per built section, no parallel sections
- T4 radial network: the built sections form an arborescence from the source
- T5 forced mode: every reachable building is connected, unreachable ones
  are reported
- T7 energy balance: producer = demand + losses; producer capacity covers
  the source connection plus losses
- simultaneity: C = g · S per section, producer sized with GLF(N) · ΣQ
- T6 the GLF changes the route: two clusters get separate feeders without
  GLF and a shared trunk with GLF (variant B)
- economic mode: the tree supplies exactly the connected buildings, revenue
  in the objective, T9 a far, small building stays unconnected
- objective terms, fixed bridges, mip_abs_gap "auto" (only absolute gap),
  time limit, errors
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
from fheat_core.algorithms.network import calculate_glf  # noqa: E402
from fheat_core.optimization import (  # noqa: E402
    GLF_OFF,
    GLF_REFERENCE,
    HOURS_PER_YEAR,
    HOUSE_CONNECTION,
    MODE_ECONOMIC,
)
from fheat_core.optimization.energysystem import (  # noqa: E402
    MIP_ABS_GAP_AUTO_SHARE,
    solve_network,
)
from fheat_core.optimization.glf_terms import design_loads, glf_factors, reference_tree  # noqa: E402
from fheat_core.optimization.linearize import linearize_pipes  # noqa: E402
from fheat_core.optimization.preprocess import POWER, simplify_network  # noqa: E402
from fheat_core.resources import load_pipe_costs, load_pipe_info  # noqa: E402

from tests.optimization_graphs import grid_case, random_case, street_graph, two_cluster_case  # noqa: E402

FULL_LOAD_HOURS = 2000.0
HTEMP, LTEMP = 80.0, 50.0


def _solve(G, b, s, htemp=HTEMP, ltemp=LTEMP, **config):
    cfg = OptimizationConfig(**config)
    net = simplify_network(G, b, s)
    lin = linearize_pipes(load_pipe_info(), load_pipe_costs(), design_loads(net), htemp, ltemp)
    demand = {k: b.at[k, cols.THERMAL_POWER] * FULL_LOAD_HOURS for k in b.index}
    return net, lin, solve_network(net, lin, demand, cfg)


def _built_tree(res) -> nx.DiGraph:
    built = res.edges[res.edges["built"]]
    return nx.DiGraph(list(zip(built["flow_from"], built["flow_to"], strict=True)))


def _mesh_graph():
    """Mesh A(0,0)-B(10,0)-C(10,10)-D(0,10) with a spur at D.

    A-B, B-C 10 m; A-D, D-C 50 m. Source at A, building 0 at C (30 kW),
    building 1 at the end of the spur D-(−10,20) (20 kW). Optimal: A-B-C and A-D.
    """
    dx = math.sqrt(25 ** 2 - 5 ** 2)
    return street_graph(
        streets=[
            [(0, 0), (10, 0)], [(10, 0), (10, 10)],
            [(0, 0), (-dx, 5), (0, 10)], [(0, 10), (5, 10 + dx), (10, 10)],
            [(0, 10), (-10, 20)],
        ],
        buildings=[((15, 10), (10, 10), 30.0), ((-10, 25), (-10, 20), 20.0)],
        sources=[((-5, 0), (0, 0))],
    )


@pytest.fixture(scope="module")
def mesh():
    return _solve(*_mesh_graph())


@pytest.fixture(scope="module")
def mesh_without_glf():
    return _solve(*_mesh_graph(), glf_mode=GLF_OFF)


def _random_solved(seed):
    return _solve(*random_case(seed))


# ---------------------------------------------------------------------------
# optimum of a small mesh
# ---------------------------------------------------------------------------


class TestMeshOptimum:
    def test_optimal(self, mesh):
        _, _, res = mesh
        assert res.termination == "optimal"
        assert res.glf_mode == OptimizationConfig().glf_mode
        assert res.achieved_gap <= res.mip_abs_gap + 1e-6

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

    def test_capacity_equals_power_without_glf(self, mesh_without_glf):
        _, _, res = mesh_without_glf
        built = res.edges[res.edges["built"]]
        assert (built[cols.GLF_MODEL] == 1.0).all()
        assert built[cols.CAPACITY_MODEL].to_numpy() == pytest.approx(built[cols.THERMAL_POWER].to_numpy())
        src = res.edges[res.edges[cols.TYPE] == "Quellenanschluss"].iloc[0]
        assert (src[cols.CAPACITY_MODEL], src[cols.N_BUILDINGS]) == (pytest.approx(50.0), pytest.approx(2))

    def test_capacity_with_glf(self, mesh):
        _, _, res = mesh
        built = res.edges[res.edges["built"]]
        assert built[cols.CAPACITY_MODEL].to_numpy() == pytest.approx(
            (built[cols.GLF_MODEL] * built[cols.THERMAL_POWER]).to_numpy()
        )
        src = res.edges[res.edges[cols.TYPE] == "Quellenanschluss"].iloc[0]
        assert src[cols.GLF_MODEL] == pytest.approx(calculate_glf(2))
        assert res.source_capacity == pytest.approx(calculate_glf(2) * 50.0 + res.loss_flow)

    def test_glf_lowers_producer_capacity(self, mesh, mesh_without_glf):
        assert mesh[2].source_capacity < mesh_without_glf[2].source_capacity

    def test_objective_parts(self, mesh):
        _, lin, res = mesh
        cfg = OptimizationConfig()
        parts = res.objective_parts
        assert parts.total == pytest.approx(res.objective)
        pipe_annuity = economics.annuity(1.0, cfg.lifetime_pipes, cfg.interest_rate)
        assert parts.pipes == pytest.approx(pipe_annuity * res.edges[cols.INVEST_COST_MODEL].sum())
        ep_costs = economics.annuity(cfg.source_capex_eur_per_kw, cfg.lifetime_source, cfg.interest_rate)
        assert parts.source_invest == pytest.approx(ep_costs * res.source_capacity)
        heat_cost = cfg.heat_cost_eur_per_kwh * HOURS_PER_YEAR
        assert parts.heat_demand == pytest.approx(heat_cost * res.demand_flow)
        assert parts.heat_losses == pytest.approx(heat_cost * res.loss_flow)

    def test_invest_cost_of_built_sections(self, mesh):
        _, lin, res = mesh
        for _, row in res.edges.iterrows():
            expected = lin.invest_cost(row[cols.TYPE], row[cols.LENGTH], row[cols.CAPACITY_MODEL], int(row["built"]))
            assert row[cols.INVEST_COST_MODEL] == pytest.approx(expected, abs=1e-6)

    def test_mip_abs_gap_auto(self, mesh):
        net, lin, res = mesh
        cfg = OptimizationConfig()
        tree = reference_tree(net)
        g = glf_factors(net, tree, cfg.glf_mode)
        invest = sum(
            lin.invest_cost(net.graph.edges[e][cols.TYPE], net.graph.edges[e][cols.LENGTH], g[e] * s0, 1)
            for e, s0 in tree.s0.items()
        )
        pipe_annuity = economics.annuity(1.0, cfg.lifetime_pipes, cfg.interest_rate)
        assert res.mip_abs_gap == pytest.approx(MIP_ABS_GAP_AUTO_SHARE * pipe_annuity * invest)

    def test_numeric_mip_abs_gap_passed_through(self):
        G, b, s = random_case(3)
        _, _, res = _solve(G, b, s, mip_abs_gap=123.0)
        assert res.mip_abs_gap == 123.0


# ---------------------------------------------------------------------------
# T6: the GLF changes the route
# ---------------------------------------------------------------------------


class TestT6GlfChangesRoute:
    """Two clusters of 50 buildings à 100 kW; feeders 90 m, trunk 60 m, branches 50 m.

    At 70/50 °C. The outcome depends on the cost line: at 80/50 °C the line is
    flatter (larger Q_max per DN) and the trunk is cheaper even without GLF.
    """

    @staticmethod
    def _built_streets(glf_mode):
        (G, b, s), nodes = two_cluster_case()
        net, _, res = _solve(G, b, s, htemp=70.0, ltemp=50.0, glf_mode=glf_mode)
        name = {frozenset((nodes[p], nodes[q])): p + q for p in nodes for q in nodes if p < q}
        built = res.edges[res.edges["built"] & (res.edges[cols.TYPE] == "Straßenleitung")]
        return {name[frozenset((net.node_coords[u], net.node_coords[v]))]
                for u, v in zip(built["u"], built["v"], strict=True)}

    def test_separate_feeders_without_glf(self):
        assert self._built_streets(GLF_OFF) == {"AS", "BS"}

    def test_shared_trunk_with_glf(self):
        assert self._built_streets(GLF_REFERENCE) == {"JS", "AJ", "BJ"}


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
        assert (e.loc[~e["built"], cols.CAPACITY_MODEL].abs() < 1e-6).all()
        pipe_info = load_pipe_info()
        for _, row in e[e["built"]].iterrows():
            assert {row["flow_from"], row["flow_to"]} == {row["u"], row["v"]}
            vf = calculate_volumeflow(row[cols.CAPACITY_MODEL], HTEMP, LTEMP)
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
        assert res.source_capacity == pytest.approx(src[cols.CAPACITY_MODEL] + res.loss_flow)
        n = len(net.building_nodes)
        assert src[cols.CAPACITY_MODEL] == pytest.approx(calculate_glf(n) * powers)

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
            _, _, res = _solve(G, b, s, mip_abs_gap=0.0, time_limit_s=2.0)
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


# ---------------------------------------------------------------------------
# economic mode
# ---------------------------------------------------------------------------

PRICE = 0.15   # [€/kWh]


def _economic(G, b, s):
    return _solve(G, b, s, mode=MODE_ECONOMIC, heat_price_eur_per_kwh=PRICE)


def _t9_graph():
    """Three large buildings (150 kW) within 100 m of the source, one small
    building (5 kW) at the end of a 2.9 km street."""
    return street_graph(
        streets=[[(0, 0), (30, 0), (60, 0), (100, 0)], [(100, 0), (3000, 0)]],
        buildings=[((30, 10), (30, 0), 150.0), ((60, 10), (60, 0), 150.0),
                   ((100, 10), (100, 0), 150.0), ((3000, 10), (3000, 0), 5.0)],
        sources=[((-5, 0), (0, 0))],
    )


class TestT9Economic:
    def test_far_small_building_not_connected(self):
        net, _, res = _economic(*_t9_graph())
        assert res.mode == MODE_ECONOMIC
        assert sorted(res.connected) == [0, 1, 2]
        far = net.building_nodes[3]
        assert far not in _built_tree(res).nodes

    def test_forced_mode_connects_it(self):
        _, _, res = _solve(*_t9_graph())
        assert sorted(res.connected) == [0, 1, 2, 3]

    def test_revenue_in_objective(self):
        _, _, res = _economic(*_t9_graph())
        parts = res.objective_parts
        assert parts.revenue == pytest.approx(PRICE * HOURS_PER_YEAR * res.demand_flow)
        assert res.demand_flow == pytest.approx(3 * 150.0 * FULL_LOAD_HOURS / HOURS_PER_YEAR)
        assert parts.total == pytest.approx(res.objective)
        assert res.objective < 0

    def test_glf_on_bridges_is_an_estimate(self):
        net, _, res = _economic(*_t9_graph())
        e = res.edges
        houses = e[e[cols.TYPE] == HOUSE_CONNECTION]
        assert not houses[cols.GLF_MODEL_ESTIMATED].any()
        src = e[e[cols.TYPE] == "Quellenanschluss"].iloc[0]
        assert bool(src[cols.GLF_MODEL_ESTIMATED])
        assert src[cols.GLF_MODEL] == pytest.approx(calculate_glf(4))
        assert src[cols.N_BUILDINGS] == pytest.approx(3)

    def test_forced_mode_glf_on_bridges_is_exact(self):
        _, _, res = _solve(*_t9_graph())
        e = res.edges
        assert not e.loc[e["fixed"], cols.GLF_MODEL_ESTIMATED].any()


@pytest.mark.parametrize("seed", range(10))
class TestEconomicRandomNetworks:
    def test_tree_supplies_exactly_the_connected_buildings(self, seed):
        net, _, res = _economic(*random_case(seed))
        tree = _built_tree(res)
        connected = {net.building_nodes[k] for k in res.connected}
        if not connected:
            assert tree.number_of_nodes() == 0
            return
        assert nx.is_arborescence(tree)
        assert [n for n, d in tree.in_degree() if d == 0] == [net.source_node()]
        assert set(net.building_nodes.values()) & set(tree.nodes) == connected

    def test_energy_balance(self, seed):
        net, _, res = _economic(*random_case(seed))
        powers = sum(net.graph.nodes[net.building_nodes[k]][POWER] for k in res.connected)
        assert res.demand_flow == pytest.approx(powers * FULL_LOAD_HOURS / HOURS_PER_YEAR)
        assert res.producer_flow == pytest.approx(res.demand_flow + res.loss_flow)

    def test_not_worse_than_forced(self, seed):
        """Forced connection is one feasible economic solution."""
        G, b, s = random_case(seed)
        _, _, forced = _solve(G, b, s)
        _, _, economic = _economic(G, b, s)
        forced_with_revenue = forced.objective - PRICE * HOURS_PER_YEAR * forced.demand_flow
        assert economic.objective <= forced_with_revenue + economic.mip_abs_gap + 1e-6
