"""Tests for fheat_core.optimization.block on its own (Pyomo model without oemof).

Covers:
- NetworkSets: fixed bridges with one direction, free sections with two
- T3 on the variables: λ_ij + λ_ji = y_e ≤ 1 for every section
- radial constraint, every built direction supplies a building
- capacity with simultaneity: C_e = g_e · S_e on every built section
"""
from __future__ import annotations

import pytest

po = pytest.importorskip("pyomo.environ")

from pyomo.contrib.appsi.solvers import Highs  # noqa: E402

from fheat_core import columns as cols  # noqa: E402
from fheat_core.optimization import GLF_REFERENCE  # noqa: E402
from fheat_core.optimization.block import NetworkSets, build_network_block  # noqa: E402
from fheat_core.optimization.glf_terms import design_loads, glf_factors, reference_tree  # noqa: E402
from fheat_core.optimization.linearize import linearize_pipes  # noqa: E402
from fheat_core.optimization.preprocess import KIND, KIND_JUNCTION, simplify_network  # noqa: E402
from fheat_core.resources import load_pipe_costs, load_pipe_info  # noqa: E402

from tests.optimization_graphs import random_case  # noqa: E402


def _network(seed):
    net = simplify_network(*random_case(seed))
    lin = linearize_pipes(load_pipe_info(), load_pipe_costs(), design_loads(net), 80, 50)
    return net, lin


def _solved_block(seed):
    net, lin = _network(seed)
    m = po.ConcreteModel()
    m.netz = build_network_block(net, lin, glf_factors(net, reference_tree(net), GLF_REFERENCE))
    m.objective = po.Objective(expr=m.netz.invest_cost)
    Highs().solve(m)
    return net, m.netz


@pytest.mark.parametrize("seed", range(8))
class TestBlock:
    def test_sets(self, seed):
        net, lin = _network(seed)
        sets = NetworkSets.of(net, float(lin.capacities["capacity"].max()))
        assert sets.fixed_edges == {tuple(sorted(e)) for e in net.bridges()}
        for e, arcs in sets.arcs_of_edge.items():
            assert len(arcs) == (1 if e in sets.fixed_edges else 2)
        assert net.source_node() in sets.source_edge
        assert sets.total_power == pytest.approx(sum(sets.building_power.values()))

    def test_t3_one_direction_per_section(self, seed):
        _, b = _solved_block(seed)
        for u, v in b.EDGES:
            directions = [a for a in b.ARCS if {a[0], a[1]} == {u, v}]
            total = sum(po.value(b.direction[a]) for a in directions)
            assert total == pytest.approx(po.value(b.built[u, v]))
            assert total <= 1 + 1e-6

    def test_radial_and_supplying(self, seed):
        net, b = _solved_block(seed)
        for j, kind in net.graph.nodes(data=KIND):
            if kind == KIND_JUNCTION:
                incoming = sum(po.value(b.direction[a]) for a in b.ARCS if a[1] == j)
                assert incoming <= 1 + 1e-6
        for a in b.ARCS:
            if po.value(b.direction[a]) > 0.5:
                assert po.value(b.count_flow[a]) >= 1 - 1e-6

    def test_capacity_with_glf(self, seed):
        net, b = _solved_block(seed)
        g = glf_factors(net, reference_tree(net), GLF_REFERENCE)
        for e in b.EDGES:
            flow = sum(po.value(b.power_flow[a]) for a in b.ARCS if {a[0], a[1]} == set(e))
            assert po.value(b.capacity[e]) == pytest.approx(g[e] * flow, abs=1e-6)

    def test_only_free_sections_are_decisions(self, seed):
        net, b = _solved_block(seed)
        free = [e for e in b.EDGES if not b.built[e].fixed]
        assert len(free) == sum(1 for *_, d in net.graph.edges(data=True) if not d["is_bridge"])
        assert all(net.graph.edges[e][cols.TYPE] == "Straßenleitung" for e in free)
