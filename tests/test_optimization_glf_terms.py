"""Tests for fheat_core.optimization.glf_terms.

Covers:
- reference_tree: n0 and S0 per section equal the shortest-path network of
  compute_network (same Dijkstra logic), segment by segment
"""
from __future__ import annotations

import pytest

from fheat_core import columns as cols
from fheat_core.algorithms.network import compute_network
from fheat_core.optimization.glf_terms import reference_tree
from fheat_core.optimization.preprocess import GEOMETRY, simplify_network
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
