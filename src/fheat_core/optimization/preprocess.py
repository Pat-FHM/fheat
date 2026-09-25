"""Graph simplification for the MILP network optimisation.

The street graph of ``steps/network.py`` has one node per street vertex. This
module turns it into the candidate graph of the MILP:

1. Integer node IDs (Pyomo cannot index by coordinate tuples); the mapping
   ID ↔ coordinate is kept.
2. Parts of the graph that cannot reach a heat source are removed; their
   buildings are reported as unreachable.
3. Dead ends without buildings are removed: every part of the graph that is
   attached to the rest through a single node and contains neither a building
   nor a source (dead-end streets, loops hanging off a single junction).
4. Chains of street nodes with exactly two street edges and no connection are
   merged into one edge per street section (summed length, joined geometry).
5. Parallel sections between the same two nodes are reduced to the shortest
   one (R3); self-loops created by step 4 are removed.
6. Bridges are identified. For every bridge the side without a source is
   known, hence also the flow direction and the buildings behind it.

Steps 3 to 5 do not change the optimum of the MILP: removed parts contain no
building or source, so a cost-minimal tree never uses them, and of two
parallel sections between the same nodes the shorter one is always cheaper
(pipe costs and losses scale with the length). Every removed edge is counted
in :class:`SimplificationReport`, whose lengths add up to the input length.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Hashable

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from fheat_core import columns as cols

logger = logging.getLogger(__name__)

HOUSE_CONNECTION = "Hausanschluss"
STREET_PIPE = "Straßenleitung"
SOURCE_CONNECTION = "Quellenanschluss"

# node attributes
COORD = "coord"
KIND = "kind"
KIND_BUILDING = "building"
KIND_SOURCE = "source"
KIND_JUNCTION = "junction"
BUILDING_KEY = "building_key"   # index of the building in buildings_gdf
SOURCE_KEY = "source_key"       # index of the source in source_gdf

# edge attributes (besides cols.TYPE and cols.LENGTH)
GEOMETRY = "geometry"
IS_BRIDGE = "is_bridge"
FLOW_FROM = "flow_from"         # fixed flow direction of a bridge (None if not fixed)
FLOW_TO = "flow_to"
N_BEHIND = "n_behind"           # buildings behind a bridge in flow direction
POWER_BEHIND = "power_behind"   # summed connection power behind a bridge [kW]


@dataclass
class SimplificationReport:
    """What the simplification removed or merged (lengths in m).

    ``length_before`` equals ``length_after`` plus all ``*_length_removed``.
    """

    nodes_before: int = 0
    edges_before: int = 0
    length_before: float = 0.0
    nodes_after: int = 0
    edges_after: int = 0
    length_after: float = 0.0
    disconnected_edges_removed: int = 0
    disconnected_length_removed: float = 0.0
    dead_end_edges_removed: int = 0
    dead_end_length_removed: float = 0.0
    parallel_edges_removed: int = 0
    parallel_length_removed: float = 0.0
    self_loops_removed: int = 0
    self_loop_length_removed: float = 0.0
    nodes_merged: int = 0
    bridges: int = 0
    bridges_with_fixed_direction: int = 0
    unreachable_buildings: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimplifiedNetwork:
    """Candidate graph for the MILP.

    ``graph`` has integer nodes with the attributes ``coord``, ``kind`` and
    ``building_key`` / ``source_key``; edges carry ``cols.TYPE``,
    ``cols.LENGTH``, ``geometry`` and the bridge attributes.
    ``node_ids`` maps every coordinate of the input graph to its integer ID,
    ``node_coords`` the other way round.
    """

    graph: nx.Graph
    node_coords: dict[int, tuple]
    node_ids: dict[tuple, int]
    building_nodes: dict[Hashable, int]
    source_nodes: dict[Hashable, int]
    unreachable_buildings: list = field(default_factory=list)
    report: SimplificationReport = field(default_factory=SimplificationReport)

    def bridges(self) -> list[tuple[int, int]]:
        return [(u, v) for u, v, d in self.graph.edges(data=True) if d[IS_BRIDGE]]


def simplify_network(
    G: nx.Graph,
    buildings_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    power_att: str = cols.THERMAL_POWER,
) -> SimplifiedNetwork:
    """Simplify the street graph built by ``steps/network.py``.

    Buildings are identified by their centroid (as in ``compute_network``),
    sources by their point geometry. ``G`` is not modified.
    """
    report = SimplificationReport(
        nodes_before=G.number_of_nodes(),
        edges_before=G.number_of_edges(),
        length_before=_total_length(G),
    )

    node_coords = dict(enumerate(G.nodes))
    node_ids = {coord: i for i, coord in node_coords.items()}
    H = nx.Graph()
    for i, coord in node_coords.items():
        H.add_node(i, **{COORD: coord, KIND: KIND_JUNCTION})
    for u, v, data in G.edges(data=True):
        attrs = dict(data)
        attrs.setdefault(GEOMETRY, LineString([u, v]))
        H.add_edge(node_ids[u], node_ids[v], **attrs)

    building_nodes, power, unreachable = _mark_buildings(H, node_ids, buildings_gdf, power_att)
    source_nodes = _mark_sources(H, node_ids, source_gdf)

    _remove_disconnected(H, source_nodes.values(), report)
    for key, node in list(building_nodes.items()):
        if node not in H:
            unreachable.append(key)
            del building_nodes[key]
    _check_leaves(H, building_nodes, source_nodes)

    while True:
        edges = H.number_of_edges()
        _remove_dead_ends(H, source_nodes.values(), report)
        H = _merge_chains(H, report)
        if H.number_of_edges() == edges:
            break

    _mark_bridges(H, source_nodes, {building_nodes[k]: power[k] for k in building_nodes}, report)

    report.nodes_after = H.number_of_nodes()
    report.edges_after = H.number_of_edges()
    report.length_after = _total_length(H)
    report.unreachable_buildings = len(unreachable)

    logger.info("Graph simplification: %s", report.as_dict())
    if unreachable:
        logger.warning(
            "%d building(s) cannot be reached from a heat source and are ignored: %s",
            len(unreachable), unreachable,
        )

    return SimplifiedNetwork(
        graph=H,
        node_coords=node_coords,
        node_ids=node_ids,
        building_nodes=building_nodes,
        source_nodes=source_nodes,
        unreachable_buildings=unreachable,
        report=report,
    )


# ---------------------------------------------------------------------------
# terminals
# ---------------------------------------------------------------------------


def _mark_buildings(H, node_ids, buildings_gdf, power_att):
    building_nodes: dict[Hashable, int] = {}
    power: dict[Hashable, float] = {}
    unreachable: list = []
    seen: dict[int, Hashable] = {}
    for key, row in buildings_gdf.iterrows():
        coord = (row[cols.CENTROID].x, row[cols.CENTROID].y)
        node = node_ids.get(coord)
        if node is None:
            unreachable.append(key)
            continue
        if node in seen:
            raise ValueError(
                f"Buildings {seen[node]!r} and {key!r} share the centroid {coord}; "
                "the network graph cannot tell them apart."
            )
        seen[node] = key
        H.nodes[node][KIND] = KIND_BUILDING
        H.nodes[node][BUILDING_KEY] = key
        building_nodes[key] = node
        power[key] = float(row[power_att])
    return building_nodes, power, unreachable


def _mark_sources(H, node_ids, source_gdf):
    source_nodes: dict[Hashable, int] = {}
    for key, row in source_gdf.iterrows():
        coord = row["geometry"].coords[0]
        node = node_ids.get(coord)
        if node is None:
            raise ValueError(f"Source {key!r} at {coord} is not connected to the network graph.")
        if H.nodes[node][KIND] != KIND_JUNCTION:
            raise ValueError(f"Source {key!r} at {coord} coincides with another building or source.")
        H.nodes[node][KIND] = KIND_SOURCE
        H.nodes[node][SOURCE_KEY] = key
        source_nodes[key] = node
    if not source_nodes:
        raise ValueError("source_gdf is empty: at least one heat source is required.")
    return source_nodes


def _check_leaves(H, building_nodes, source_nodes):
    """Buildings and sources must be leaves attached by their connection edge."""
    for kind, nodes, edge_type in (
        ("Building", building_nodes, HOUSE_CONNECTION),
        ("Source", source_nodes, SOURCE_CONNECTION),
    ):
        for key, node in nodes.items():
            edges = list(H.edges(node, data=True))
            if len(edges) != 1 or edges[0][2].get(cols.TYPE) != edge_type or edges[0][1] == node:
                raise ValueError(
                    f"{kind} {key!r} must be connected by exactly one '{edge_type}' edge "
                    f"(found {len(edges)} edge(s) at {H.nodes[node][COORD]})."
                )


# ---------------------------------------------------------------------------
# removal steps
# ---------------------------------------------------------------------------


def _remove_disconnected(H, sources, report):
    keep = set()
    for s in sources:
        keep |= nx.node_connected_component(H, s)
    drop = [n for n in H if n not in keep]
    sub = H.subgraph(drop)
    report.disconnected_edges_removed += sub.number_of_edges()
    report.disconnected_length_removed += _total_length(sub)
    H.remove_nodes_from(drop)


def _remove_dead_ends(H, sources, report):
    """Remove every part attached through one node without building or source.

    Uses the block-cut tree: rooted at a source, every subtree without a
    terminal (building or source) is dropped. Articulation points that still
    lead to a terminal are kept.
    """
    terminals = {n for n, k in H.nodes(data=KIND) if k != KIND_JUNCTION}
    cut_nodes = set(nx.articulation_points(H))
    blocks = [set(b) for b in nx.biconnected_components(H)]

    tree = nx.Graph()
    block_of = {}
    for i, block in enumerate(blocks):
        tree.add_node(("block", i))
        for n in block - cut_nodes:
            block_of[n] = i
        for a in block & cut_nodes:
            tree.add_edge(("block", i), ("cut", a))

    def own_nodes(t):
        kind, ref = t
        return {ref} if kind == "cut" else blocks[ref] - cut_nodes

    # every component contains a source (see _remove_disconnected); isolated
    # sources without edges have no tree node
    source_tree_nodes = [
        ("cut", s) if s in cut_nodes else ("block", block_of[s])
        for s in sources
        if s in cut_nodes or s in block_of
    ]
    drop = set()
    for component in nx.connected_components(tree):
        root = next(t for t in source_tree_nodes if t in component)
        parent = dict(nx.bfs_predecessors(tree, root))
        has_terminal = {}
        for t in nx.dfs_postorder_nodes(tree, root):
            has_terminal[t] = has_terminal.get(t, False) or bool(own_nodes(t) & terminals)
            if has_terminal[t] and t in parent:
                has_terminal[parent[t]] = True
        drop |= {n for t, keep in has_terminal.items() if not keep for n in own_nodes(t)}
    if not drop:
        return
    removed = [(u, v, d) for u, v, d in H.edges(drop, data=True)]
    report.dead_end_edges_removed += len(removed)
    report.dead_end_length_removed += sum(d[cols.LENGTH] for _, _, d in removed)
    H.remove_nodes_from(drop)


def _is_mergeable(H, n) -> bool:
    if H.nodes[n][KIND] != KIND_JUNCTION or H.degree(n) != 2:
        return False
    return all(d.get(cols.TYPE) == STREET_PIPE for _, _, d in H.edges(n, data=True))


def _merge_chains(H, report) -> nx.Graph:
    """Merge chains of mergeable nodes into one edge; keep the shortest parallel edge."""
    out = nx.Graph()
    out.add_nodes_from(n for n in H.nodes(data=True) if not _is_mergeable(H, n[0]))
    visited = set()

    def add(u, v, attrs):
        if u == v:
            report.self_loops_removed += 1
            report.self_loop_length_removed += attrs[cols.LENGTH]
            return
        if out.has_edge(u, v):
            report.parallel_edges_removed += 1
            old = out.edges[u, v]
            if attrs[cols.LENGTH] < old[cols.LENGTH]:
                report.parallel_length_removed += old[cols.LENGTH]
                out.remove_edge(u, v)
            else:
                report.parallel_length_removed += attrs[cols.LENGTH]
                return
        out.add_edge(u, v, **attrs)

    for start in list(out.nodes):
        for nbr in H.neighbors(start):
            if frozenset((start, nbr)) in visited:
                continue
            chain = [start, nbr]
            visited.add(frozenset((start, nbr)))
            while _is_mergeable(H, chain[-1]):
                nxt = next(n for n in H.neighbors(chain[-1]) if frozenset((chain[-1], n)) not in visited)
                visited.add(frozenset((chain[-1], nxt)))
                chain.append(nxt)
            add(chain[0], chain[-1], _chain_attrs(H, chain))
            report.nodes_merged += len(chain) - 2

    # edges never reached from a kept node form cycles of mergeable nodes only
    for u, v, d in H.edges(data=True):
        if frozenset((u, v)) not in visited:
            report.self_loops_removed += 1
            report.self_loop_length_removed += d[cols.LENGTH]
    return out


def _chain_attrs(H, chain) -> dict:
    if len(chain) == 2:
        return dict(H.edges[chain[0], chain[1]])
    coords = [H.nodes[chain[0]][COORD]]
    length = 0.0
    for u, v in zip(chain[:-1], chain[1:]):
        d = H.edges[u, v]
        length += d[cols.LENGTH]
        seg = list(d[GEOMETRY].coords)
        if seg[0] != coords[-1]:
            seg.reverse()
        coords.extend(seg[1:])
    return {cols.TYPE: STREET_PIPE, cols.LENGTH: length, GEOMETRY: LineString(coords)}


# ---------------------------------------------------------------------------
# bridges
# ---------------------------------------------------------------------------


def _mark_bridges(H, source_nodes, building_power, report):
    """Flag bridges; orient those with all sources on one side.

    Uses any spanning tree rooted at a source: removing a bridge (parent,
    child) separates exactly the subtree of ``child`` from the rest.
    """
    nx.set_edge_attributes(H, False, IS_BRIDGE)
    nx.set_edge_attributes(H, None, FLOW_FROM)
    nx.set_edge_attributes(H, None, FLOW_TO)
    nx.set_edge_attributes(H, None, N_BEHIND)
    nx.set_edge_attributes(H, None, POWER_BEHIND)

    sources = set(source_nodes.values())
    parent, n_sub, p_sub, s_sub, totals = {}, {}, {}, {}, {}
    for component in nx.connected_components(H):
        root = next(s for s in sources if s in component)
        tree = nx.bfs_tree(H, root)
        for n in nx.dfs_postorder_nodes(tree, root):
            n_sub[n] = n_sub.get(n, 0) + (n in building_power)
            p_sub[n] = p_sub.get(n, 0.0) + building_power.get(n, 0.0)
            s_sub[n] = s_sub.get(n, 0) + (n in sources)
            for p in tree.predecessors(n):
                parent[n] = p
                n_sub[p] = n_sub.get(p, 0) + n_sub[n]
                p_sub[p] = p_sub.get(p, 0.0) + p_sub[n]
                s_sub[p] = s_sub.get(p, 0) + s_sub[n]
        totals[root] = (n_sub[root], p_sub[root], s_sub[root])
        for n in component:
            totals[n] = totals[root]

    for u, v in nx.bridges(H):
        child, par = (v, u) if parent.get(v) == u else (u, v)
        n_total, p_total, s_total = totals[child]
        d = H.edges[u, v]
        d[IS_BRIDGE] = True
        report.bridges += 1
        if s_sub[child] == 0:
            d[FLOW_FROM], d[FLOW_TO] = par, child
            d[N_BEHIND], d[POWER_BEHIND] = n_sub[child], p_sub[child]
        elif s_sub[child] == s_total:
            d[FLOW_FROM], d[FLOW_TO] = child, par
            d[N_BEHIND], d[POWER_BEHIND] = n_total - n_sub[child], p_total - p_sub[child]
        else:
            continue
        report.bridges_with_fixed_direction += 1


def _total_length(G) -> float:
    return float(sum(d.get(cols.LENGTH, 0.0) for _, _, d in G.edges(data=True)))
