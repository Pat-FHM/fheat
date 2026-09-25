"""Shortest-path reference tree for the MILP (plan section 7, details on B).

The reference tree is the network ``compute_network`` builds: the shortest
path from the source to every building, with the buildings and their summed
connection power accumulated per section.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from fheat_core import columns as cols
from fheat_core.optimization.preprocess import POWER, SimplifiedNetwork, edge_key


@dataclass(frozen=True)
class ReferenceTree:
    """Sections of the shortest-path tree with n0 buildings and S0 [kW] behind them.

    Keys are :func:`edge_key` tuples; sections outside the tree are missing.
    """

    n0: dict[tuple[int, int], int]
    s0: dict[tuple[int, int], float]


def reference_tree(network: SimplifiedNetwork) -> ReferenceTree:
    """Shortest paths (weight ``cols.LENGTH``) from the source to every building."""
    H = network.graph
    paths = nx.single_source_dijkstra_path(H, network.source_node(), weight=cols.LENGTH)
    n0: dict[tuple[int, int], int] = {}
    s0: dict[tuple[int, int], float] = {}
    for building in network.building_nodes.values():
        path = paths[building]
        for u, v in zip(path[:-1], path[1:], strict=True):
            e = edge_key(u, v)
            n0[e] = n0.get(e, 0) + 1
            s0[e] = s0.get(e, 0.0) + H.nodes[building][POWER]
    return ReferenceTree(n0=n0, s0=s0)
