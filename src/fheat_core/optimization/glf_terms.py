"""Simultaneity (GLF) in the route choice: factor g_e per section for block.py, (7).

The design capacity of a pipe is GLF(n) · S with the simultaneity factor
GLF(n) = a + b / (1 + (n/c)^d) (Winter et al. 2001, ``calculate_glf``). Both
n and S depend on the route; to keep the model linear without additional
binary variables, n is fixed per section before the solve:

``glf_mode = "referenz"``
    (A) Bridges with a fixed direction: n = buildings behind the bridge. In
        the forced mode n and S are fixed there, so the GLF is exact.
    (B) Other sections: n0 from the shortest-path reference tree, the network
        ``compute_network`` builds. For a section outside the tree n0 is the
        smaller node count among its ends with a count above 0 (an end with 0
        is not in the tree and says nothing about the buildings behind the
        pipe); with both ends at 0, n0 = 1.
``glf_mode = "aus"``
    g_e = 1 (C = S), for comparison.

Both modes use the same cost line: its regression range ends at the design
loads with simultaneity (:func:`design_loads`), so a comparison shows the
effect of the GLF only.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from fheat_core import columns as cols
from fheat_core.algorithms.network import calculate_glf
from fheat_core.optimization import GLF_OFF, MODE_FORCED
from fheat_core.optimization.linearize import DesignLoads
from fheat_core.optimization.preprocess import N_BEHIND, POWER, SimplifiedNetwork, edge_key


@dataclass(frozen=True)
class ReferenceTree:
    """Shortest-path tree from the source to every building.

    ``n0`` and ``s0`` map tree sections (:func:`edge_key`) to the number of
    buildings and their summed connection power [kW] behind them;
    ``node_count`` maps every tree node to the buildings supplied through it.
    """

    n0: dict[tuple[int, int], int]
    s0: dict[tuple[int, int], float]
    node_count: dict[int, int]


def reference_tree(network: SimplifiedNetwork) -> ReferenceTree:
    """Shortest paths (weight ``cols.LENGTH``) from the source, as ``compute_network``."""
    H = network.graph
    paths = nx.single_source_dijkstra_path(H, network.source_node(), weight=cols.LENGTH)
    n0: dict[tuple[int, int], int] = {}
    s0: dict[tuple[int, int], float] = {}
    node_count: dict[int, int] = {}
    for building in network.building_nodes.values():
        path = paths[building]
        for node in path:
            node_count[node] = node_count.get(node, 0) + 1
        for u, v in zip(path[:-1], path[1:], strict=True):
            e = edge_key(u, v)
            n0[e] = n0.get(e, 0) + 1
            s0[e] = s0.get(e, 0.0) + H.nodes[building][POWER]
    return ReferenceTree(n0=n0, s0=s0, node_count=node_count)


def design_loads(network: SimplifiedNetwork) -> DesignLoads:
    """Largest design load [kW] per edge type, the end of the regression range.

    House connection: GLF(1) · max Q_k; street pipe: GLF(N) · ΣQ_k (the source
    connection). Used for both glf modes.
    """
    powers = [network.graph.nodes[n][POWER] for n in network.building_nodes.values()]
    return DesignLoads(
        house=calculate_glf(1) * max(powers),
        street=calculate_glf(len(powers)) * sum(powers),
    )


def glf_factors(network: SimplifiedNetwork, tree: ReferenceTree, glf_mode: str) -> dict:
    """Factor g_e per section (:func:`edge_key`) for C_e = g_e · S_e (block.py, (7))."""
    H = network.graph
    if glf_mode == GLF_OFF:
        return {edge_key(u, v): 1.0 for u, v in H.edges}
    return {edge_key(u, v): calculate_glf(_buildings_behind(u, v, d, tree)) for u, v, d in H.edges(data=True)}


def glf_estimated(network: SimplifiedNetwork, glf_mode: str, mode: str) -> dict:
    """Per section (:func:`edge_key`): is g_e an estimate rather than the exact GLF?

    Exact only with ``glf_mode = "referenz"`` on a bridge with a fixed
    direction whose number of buildings behind is known: always in the forced
    mode, in the economic mode only with one building behind (the upper bound
    is then the value).
    """
    return {
        edge_key(u, v): glf_mode == GLF_OFF or d[N_BEHIND] is None or (mode != MODE_FORCED and d[N_BEHIND] > 1)
        for u, v, d in network.graph.edges(data=True)
    }


def _buildings_behind(u, v, data, tree) -> int:
    """n of variant A on a bridge, n0 of variant B otherwise."""
    if data[N_BEHIND] is not None:
        return data[N_BEHIND]
    e = edge_key(u, v)
    if e in tree.n0:
        return tree.n0[e]
    counts = [c for c in (tree.node_count.get(u, 0), tree.node_count.get(v, 0)) if c > 0]
    return min(counts, default=1)
