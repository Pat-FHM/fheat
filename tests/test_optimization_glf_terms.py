"""Tests for fheat_core.optimization.glf_terms.

Covers:
- reference_tree: n0 and S0 per section equal the shortest-path network of
  compute_network (same Dijkstra logic), segment by segment; node counts
- design_loads: end of the regression range (with simultaneity, both modes)
- glf_factors: variant A on bridges, variant B on tree sections and outside
  the tree (smaller count among ends with a count above 0), g = 1 without
  simultaneity
"""
from __future__ import annotations

import math

import pytest

from fheat_core import columns as cols
from fheat_core.algorithms.network import calculate_glf, compute_network
from fheat_core.optimization import GLF_OFF, GLF_REFERENCE
from fheat_core.optimization.glf_terms import ReferenceTree, design_loads, glf_factors, reference_tree
from fheat_core.optimization.preprocess import GEOMETRY, edge_key, simplify_network
from fheat_core.resources import load_pipe_info

from tests.optimization_graphs import CRS, random_case, street_graph


def _compute_network_segments(G, b, s):
    net_gdf = compute_network(G, b, s, load_pipe_info(), cols.THERMAL_POWER, 80, 50, CRS)
    return {
        frozenset(row.geometry.coords): (row[cols.N_BUILDINGS], row[cols.THERMAL_POWER])
        for _, row in net_gdf.iterrows()
    }


class TestReferenceTree:
    def test_line(self):
        G, b, s = street_graph(
            streets=[[(0, 0), (10, 0), (20, 0), (30, 0)]],
            buildings=[((10, 10), (10, 0), 10.0), ((30, 10), (30, 0), 20.0)],
            sources=[((-10, 0), (0, 0))],
        )
        net = simplify_network(G, b, s)
        tree = reference_tree(net)
        ids = net.node_ids
        src_edge = tuple(sorted((ids[(-10, 0)], ids[(0, 0)])))
        assert (tree.n0[src_edge], tree.s0[src_edge]) == (2, pytest.approx(30.0))
        last = tuple(sorted((ids[(10, 0)], ids[(30, 0)])))
        assert (tree.n0[last], tree.s0[last]) == (1, pytest.approx(20.0))

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_compute_network(self, seed):
        G, b, s = random_case(seed)
        net = simplify_network(G, b, s)
        tree = reference_tree(net)
        expected = _compute_network_segments(G, b, s)
        segments = 0
        for e, n0 in tree.n0.items():
            coords = list(net.graph.edges[e][GEOMETRY].coords)
            for seg in zip(coords[:-1], coords[1:], strict=True):
                segments += 1
                n, p = expected[frozenset(seg)]
                assert n == n0
                assert p == pytest.approx(tree.s0[e])
        assert segments == len(expected)


@pytest.fixture
def mesh():
    """Mesh A(0,0)-B(10,0)-C(10,10)-D(0,10) with a spur D-E(-10,20).

    A-B, B-C 10 m; A-D, D-C 50 m. Source at A, three buildings at C (10 kW
    each), two at E (20 kW each). Reference tree: A-C (via B, 20 m) and
    A-D-E; C-D lies outside the tree.
    """
    dx = math.sqrt(25 ** 2 - 5 ** 2)
    G, b, s = street_graph(
        streets=[
            [(0, 0), (10, 0)], [(10, 0), (10, 10)],
            [(0, 0), (-dx, 5), (0, 10)], [(0, 10), (5, 10 + dx), (10, 10)],
            [(0, 10), (-10, 20)],
        ],
        buildings=[((15, 10 + i), (10, 10), 10.0) for i in range(3)]
        + [((-10, 25 + i), (-10, 20), 20.0) for i in range(2)],
        sources=[((-5, 0), (0, 0))],
    )
    net = simplify_network(G, b, s)
    ids = net.node_ids
    sections = {
        name: edge_key(ids[p], ids[q])
        for name, p, q in [
            ("source", (-5, 0), (0, 0)), ("A-C", (0, 0), (10, 10)), ("A-D", (0, 0), (0, 10)),
            ("C-D", (10, 10), (0, 10)), ("D-E", (0, 10), (-10, 20)),
        ]
    }
    return net, sections


class TestNodeCount:
    def test_counts(self, mesh):
        net, _ = mesh
        tree = reference_tree(net)
        ids = net.node_ids
        assert tree.node_count[ids[(-5, 0)]] == 5
        assert tree.node_count[ids[(10, 10)]] == 3
        assert tree.node_count[ids[(0, 10)]] == 2
        for n in net.building_nodes.values():
            assert tree.node_count[n] == 1


class TestDesignLoads:
    def test_with_simultaneity(self, mesh):
        net, _ = mesh
        loads = design_loads(net)
        assert loads.house == pytest.approx(calculate_glf(1) * 20.0)
        assert loads.street == pytest.approx(calculate_glf(5) * 70.0)


class TestGlfFactors:
    def test_variant_a_on_bridges(self, mesh):
        net, sec = mesh
        g = glf_factors(net, reference_tree(net), GLF_REFERENCE)
        assert g[sec["source"]] == pytest.approx(calculate_glf(5))
        assert g[sec["D-E"]] == pytest.approx(calculate_glf(2))
        for n in net.building_nodes.values():
            (street,) = net.graph.neighbors(n)
            assert g[edge_key(n, street)] == pytest.approx(calculate_glf(1))

    def test_variant_b_on_tree_sections(self, mesh):
        net, sec = mesh
        g = glf_factors(net, reference_tree(net), GLF_REFERENCE)
        assert g[sec["A-C"]] == pytest.approx(calculate_glf(3))
        assert g[sec["A-D"]] == pytest.approx(calculate_glf(2))

    def test_variant_b_outside_tree_uses_smaller_node_count(self, mesh):
        net, sec = mesh
        tree = reference_tree(net)
        assert sec["C-D"] not in tree.n0
        g = glf_factors(net, tree, GLF_REFERENCE)
        assert g[sec["C-D"]] == pytest.approx(calculate_glf(2))

    def test_outside_tree_ignores_ends_outside_the_tree(self, mesh):
        """An end with count 0 is not in the tree; the other end decides."""
        net, sec = mesh
        tree = reference_tree(net)
        d = net.node_ids[(0, 10)]
        counts = {n: c for n, c in tree.node_count.items() if n != d}
        g = glf_factors(net, ReferenceTree(tree.n0, tree.s0, counts), GLF_REFERENCE)
        assert g[sec["C-D"]] == pytest.approx(calculate_glf(3))

    def test_outside_tree_both_ends_outside(self, mesh):
        net, sec = mesh
        tree = reference_tree(net)
        c, d = net.node_ids[(10, 10)], net.node_ids[(0, 10)]
        counts = {n: k for n, k in tree.node_count.items() if n not in (c, d)}
        g = glf_factors(net, ReferenceTree(tree.n0, tree.s0, counts), GLF_REFERENCE)
        assert g[sec["C-D"]] == pytest.approx(calculate_glf(1))

    def test_every_section_has_a_factor(self, mesh):
        net, _ = mesh
        g = glf_factors(net, reference_tree(net), GLF_REFERENCE)
        assert set(g) == {edge_key(u, v) for u, v in net.graph.edges}
        assert all(calculate_glf(5) <= f <= calculate_glf(1) for f in g.values())

    def test_off_is_one(self, mesh):
        net, _ = mesh
        g = glf_factors(net, reference_tree(net), GLF_OFF)
        assert set(g.values()) == {1.0}
