"""Graph simplification for the MILP network optimisation.

The street graph of ``steps/network.py`` has one node per street vertex. This
module turns it into the candidate graph of the MILP (block.py) in seven steps:

1. Integer node IDs (Pyomo cannot index by coordinate tuples); the mapping
   ID ↔ coordinate is kept.
2. Parts of the graph that cannot reach a heat source are removed. Dead ends
   are removed: every part attached to the rest through a single node that
   contains neither a building nor a source (dead-end streets, loops hanging
   off a single junction). The connecting node stays as long as the part
   before it stays.
3. Chains of street nodes with exactly two street edges and no connection are
   merged into one edge per street section (summed length, joined geometry).
4. Parallel sections between the same two nodes are reduced to the shortest
   one (one pipe per section); self-loops created by step 3 are removed.
5. Bridges are identified. For every bridge the side without a source is
   known, hence also the flow direction and the buildings behind it. With
   all buildings connected (forced mode) number and power behind a bridge
   are exact; in the economic mode they are only upper bounds, the direction
   stays fixed.
6. Buildings that cannot be reached from a source are reported
   (``on_unreachable``); without any reachable building a ``ValueError`` is
   raised.
7. Invalid input raises a ``ValueError`` with a clear message.

Steps 2 to 4 do not change the optimum of the MILP: removed parts contain no
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
from fheat_core.optimization import HOUSE_CONNECTION, SOURCE_CONNECTION, STREET_PIPE

logger = logging.getLogger(__name__)

# node attributes
COORD = "coord"
KIND = "kind"
KIND_BUILDING = "building"
KIND_SOURCE = "source"
KIND_JUNCTION = "junction"
BUILDING_KEY = "building_key"   # index of the building in buildings_gdf
POWER = "power"                 # connection power of a building [kW]
SOURCE_KEY = "source_key"       # index of the source in source_gdf

# edge attributes (besides cols.TYPE and cols.LENGTH)
GEOMETRY = "geometry"
IS_BRIDGE = "is_bridge"
FLOW_FROM = "flow_from"         # fixed flow direction of a bridge (None if not fixed)
FLOW_TO = "flow_to"
N_BEHIND = "n_behind"           # buildings behind a bridge in flow direction
POWER_BEHIND = "power_behind"   # summed connection power behind a bridge [kW]

_ON_UNREACHABLE = frozenset({"warn", "error"})


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
    ``building_key`` and ``power`` [kW] / ``source_key``; edges carry ``cols.TYPE``,
    ``cols.LENGTH`` [m], ``geometry`` and the bridge attributes.
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

    def source_node(self) -> int:
        """The node of the only heat source; the MILP supports exactly one."""
        if len(self.source_nodes) != 1:
            raise ValueError(
                f"The MILP network optimisation needs exactly one heat source "
                f"(found {len(self.source_nodes)})."
            )
        return next(iter(self.source_nodes.values()))


def edge_key(u: int, v: int) -> tuple[int, int]:
    """Orientation-free key of the section between ``u`` and ``v``."""
    return (u, v) if u < v else (v, u)


def simplify_network(
    G: nx.Graph,
    buildings_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    power_att: str = cols.THERMAL_POWER,
    on_unreachable: str = "warn",
) -> SimplifiedNetwork:
    """Simplify the street graph built by ``steps/network.py`` (steps 1 to 7 above).

    Buildings are identified by their centroid (as in ``compute_network``),
    sources by their point geometry. ``G`` is not modified.

    Buildings that cannot be reached from a heat source are left out and
    listed in ``unreachable_buildings`` (index of ``buildings_gdf``). With
    ``on_unreachable="warn"`` a warning names their IDs, with ``"error"`` a
    ``ValueError`` is raised instead.
    """
    if on_unreachable not in _ON_UNREACHABLE:
        raise ValueError(
            f"on_unreachable '{on_unreachable}' is not allowed. "
            f"Allowed values: {sorted(_ON_UNREACHABLE)}"
        )
    report = SimplificationReport(
        nodes_before=G.number_of_nodes(),
        edges_before=G.number_of_edges(),
        length_before=_total_length(G),
    )
    H, node_coords, node_ids = _integer_graph(G)
    building_nodes, unreachable = _mark_buildings(H, node_ids, buildings_gdf, power_att)
    source_nodes = _mark_sources(H, node_ids, source_gdf)

    _remove_disconnected(H, source_nodes.values(), report)
    unreachable += _drop_removed_buildings(H, building_nodes)
    _report_unreachable(buildings_gdf, unreachable, len(building_nodes), on_unreachable)
    _check_leaves(H, building_nodes, source_nodes)

    H = _reduce_until_stable(H, source_nodes.values(), report)
    _mark_bridges(H, source_nodes, {n: H.nodes[n][POWER] for n in building_nodes.values()}, report)

    report.nodes_after = H.number_of_nodes()
    report.edges_after = H.number_of_edges()
    report.length_after = _total_length(H)
    report.unreachable_buildings = len(unreachable)
    logger.info("Graph simplification: %s", report.as_dict())

    return SimplifiedNetwork(
        graph=H,
        node_coords=node_coords,
        node_ids=node_ids,
        building_nodes=building_nodes,
        source_nodes=source_nodes,
        unreachable_buildings=unreachable,
        report=report,
    )


def _integer_graph(G):
    """Copy of ``G`` with integer nodes (step 1); every edge gets a geometry."""
    node_coords = dict(enumerate(G.nodes))
    node_ids = {coord: i for i, coord in node_coords.items()}
    H = nx.Graph()
    for i, coord in node_coords.items():
        H.add_node(i, **{COORD: coord, KIND: KIND_JUNCTION})
    for u, v, data in G.edges(data=True):
        attrs = dict(data)
        attrs.setdefault(GEOMETRY, LineString([u, v]))
        H.add_edge(node_ids[u], node_ids[v], **attrs)
    return H, node_coords, node_ids


def _reduce_until_stable(H, sources, report) -> nx.Graph:
    """Repeat steps 2 to 4: dropping a parallel edge can create new chains or dead ends."""
    while True:
        edges = H.number_of_edges()
        _remove_dead_ends(H, sources, report)
        H = _merge_chains(H, report)
        if H.number_of_edges() == edges:
            return H


# ---------------------------------------------------------------------------
# terminals (buildings and sources)
# ---------------------------------------------------------------------------


def _mark_buildings(H, node_ids, buildings_gdf, power_att):
    """Mark building nodes with key and power; buildings missing in the graph are unreachable."""
    building_nodes: dict[Hashable, int] = {}
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
        H.nodes[node][POWER] = float(row[power_att])
        building_nodes[key] = node
    return building_nodes, unreachable


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


def _drop_removed_buildings(H, building_nodes) -> list:
    """Remove buildings whose node was removed from ``building_nodes``; return their keys."""
    removed = [key for key, node in building_nodes.items() if node not in H]
    for key in removed:
        del building_nodes[key]
    return removed


def _report_unreachable(buildings_gdf, unreachable, n_reachable, on_unreachable):
    """Step 6: warn about or refuse unreachable buildings; refuse an empty network."""
    if unreachable:
        ids = _building_ids(buildings_gdf, unreachable)
        msg = f"{len(unreachable)} building(s) cannot be reached from a heat source: {ids}"
        if on_unreachable == "error":
            raise ValueError(msg)
        logger.warning("%s; they are not part of the network.", msg)
    if n_reachable == 0:
        raise ValueError("No building can be reached from a heat source: there is no network to optimise.")


def _building_ids(buildings_gdf, keys) -> list:
    """``building_id`` of the given rows if the column exists, else their index."""
    if cols.BUILDING_ID in buildings_gdf.columns:
        return buildings_gdf.loc[keys, cols.BUILDING_ID].tolist()
    return list(keys)


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
# step 2: parts without source, dead ends
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


@dataclass
class _BlockCutTree:
    """Blocks (biconnected components) and cut nodes of a graph as a tree.

    Tree nodes are ``("block", index)`` and ``("cut", node)``; a block is
    linked to every cut node it contains.
    """

    tree: nx.Graph
    blocks: list[set]
    cut_nodes: set
    block_of: dict      # non-cut node → index of its only block

    @classmethod
    def of(cls, H) -> _BlockCutTree:
        cut_nodes = set(nx.articulation_points(H))
        blocks = [set(b) for b in nx.biconnected_components(H)]
        tree = nx.Graph()
        block_of = {}
        for i, block in enumerate(blocks):
            tree.add_node(("block", i))
            block_of.update({n: i for n in block - cut_nodes})
            tree.add_edges_from((("block", i), ("cut", a)) for a in block & cut_nodes)
        return cls(tree, blocks, cut_nodes, block_of)

    def tree_node(self, n):
        """Tree node holding graph node ``n`` (the graph has no isolated nodes)."""
        return ("cut", n) if n in self.cut_nodes else ("block", self.block_of[n])

    def own_nodes(self, t) -> set:
        """Graph nodes represented by tree node ``t`` (a block without its cut nodes)."""
        kind, ref = t
        return {ref} if kind == "cut" else self.blocks[ref] - self.cut_nodes


def _remove_dead_ends(H, sources, report):
    """Remove every part attached through one node without building or source.

    Uses the block-cut tree rooted at a source: every subtree without a
    terminal (building or source) is dropped.
    """
    bct = _BlockCutTree.of(H)
    terminals = {n for n, k in H.nodes(data=KIND) if k != KIND_JUNCTION}
    # every component contains a source (see _remove_disconnected)
    roots = [bct.tree_node(s) for s in sources]
    drop = set()
    for component in nx.connected_components(bct.tree):
        root = next(t for t in roots if t in component)
        drop |= _dead_end_nodes(bct, root, terminals)
    removed = list(H.edges(drop, data=True))
    report.dead_end_edges_removed += len(removed)
    report.dead_end_length_removed += sum(d[cols.LENGTH] for _, _, d in removed)
    H.remove_nodes_from(drop)


def _dead_end_nodes(bct, root, terminals) -> set:
    """Graph nodes in subtrees below ``root`` that contain no terminal.

    A cut node also belongs to its parent block and is only dropped together
    with that block.
    """
    parent = dict(nx.bfs_predecessors(bct.tree, root))
    has_terminal = _terminal_below(bct, root, parent, terminals)
    drop = set()
    for t, keep in has_terminal.items():
        if keep:
            continue
        if t[0] == "block":
            drop |= bct.own_nodes(t)
        elif not has_terminal[parent[t]]:
            drop.add(t[1])
    return drop


def _terminal_below(bct, root, parent, terminals) -> dict:
    """For every tree node below ``root``: does its subtree contain a terminal?"""
    has_terminal = {}
    for t in nx.dfs_postorder_nodes(bct.tree, root):
        has_terminal[t] = has_terminal.get(t, False) or bool(bct.own_nodes(t) & terminals)
        if has_terminal[t] and t in parent:
            has_terminal[parent[t]] = True
    return has_terminal


# ---------------------------------------------------------------------------
# steps 3 and 4: merge chains, parallel sections
# ---------------------------------------------------------------------------


def _is_mergeable(H, n) -> bool:
    if H.nodes[n][KIND] != KIND_JUNCTION or H.degree(n) != 2:
        return False
    return all(d.get(cols.TYPE) == STREET_PIPE for _, _, d in H.edges(n, data=True))


def _merge_chains(H, report) -> nx.Graph:
    """New graph with one edge per chain of mergeable nodes (steps 3 and 4).

    Every component contains a source (step 2), which is never mergeable, so
    every chain starts at a kept node.
    """
    out = nx.Graph()
    out.add_nodes_from(n for n in H.nodes(data=True) if not _is_mergeable(H, n[0]))
    visited = set()
    for start in list(out.nodes):
        for nbr in H.neighbors(start):
            if frozenset((start, nbr)) in visited:
                continue
            chain = _walk_chain(H, start, nbr, visited)
            _add_section(out, chain[0], chain[-1], _chain_attrs(H, chain), report)
            report.nodes_merged += len(chain) - 2
    return out


def _walk_chain(H, start, nbr, visited) -> list:
    """Nodes from ``start`` via ``nbr`` up to the next non-mergeable node."""
    chain = [start, nbr]
    visited.add(frozenset((start, nbr)))
    while _is_mergeable(H, chain[-1]):
        nxt = next(n for n in H.neighbors(chain[-1]) if frozenset((chain[-1], n)) not in visited)
        visited.add(frozenset((chain[-1], nxt)))
        chain.append(nxt)
    return chain


def _add_section(out, u, v, attrs, report):
    """Add a merged section; drop self-loops, keep the shorter of parallel sections.

    Self-loops come from repeated vertices in a street line (length 0 m).
    """
    length = attrs[cols.LENGTH]
    if u == v:
        report.self_loops_removed += 1
        report.self_loop_length_removed += length
        return
    if out.has_edge(u, v):
        old_length = out.edges[u, v][cols.LENGTH]
        report.parallel_edges_removed += 1
        report.parallel_length_removed += max(length, old_length)
        if length >= old_length:
            return
        out.remove_edge(u, v)
    out.add_edge(u, v, **attrs)


def _chain_attrs(H, chain) -> dict:
    """Attributes of the merged section: summed length [m], joined geometry."""
    if len(chain) == 2:
        return dict(H.edges[chain[0], chain[1]])
    coords = [H.nodes[chain[0]][COORD]]
    length = 0.0
    for u, v in zip(chain[:-1], chain[1:], strict=True):
        d = H.edges[u, v]
        length += d[cols.LENGTH]
        seg = list(d[GEOMETRY].coords)
        if seg[0] != coords[-1]:
            seg.reverse()
        coords.extend(seg[1:])
    return {cols.TYPE: STREET_PIPE, cols.LENGTH: length, GEOMETRY: LineString(coords)}


# ---------------------------------------------------------------------------
# step 5: bridges
# ---------------------------------------------------------------------------


@dataclass
class _SubtreeSums:
    """Per node of a spanning tree rooted at a source: parent and subtree sums.

    ``totals`` holds (buildings, power [kW], sources) of the node's component.
    """

    parent: dict = field(default_factory=dict)
    buildings: dict = field(default_factory=dict)
    power: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    @classmethod
    def of(cls, H, sources, building_power) -> _SubtreeSums:
        sums = cls()
        for component in nx.connected_components(H):
            root = next(s for s in sources if s in component)
            sums._add_tree(nx.bfs_tree(H, root), root, sources, building_power)
            total = (sums.buildings[root], sums.power[root], sums.sources[root])
            sums.totals.update(dict.fromkeys(component, total))
        return sums

    def _add_tree(self, tree, root, sources, building_power):
        for n in nx.dfs_postorder_nodes(tree, root):
            self.buildings[n] = self.buildings.get(n, 0) + (n in building_power)
            self.power[n] = self.power.get(n, 0.0) + building_power.get(n, 0.0)
            self.sources[n] = self.sources.get(n, 0) + (n in sources)
            for p in tree.predecessors(n):
                self.parent[n] = p
                self.buildings[p] = self.buildings.get(p, 0) + self.buildings[n]
                self.power[p] = self.power.get(p, 0.0) + self.power[n]
                self.sources[p] = self.sources.get(p, 0) + self.sources[n]


def _mark_bridges(H, source_nodes, building_power, report):
    """Flag bridges; orient those with all sources on one side.

    Uses any spanning tree rooted at a source: removing a bridge (parent,
    child) separates exactly the subtree of ``child`` from the rest.
    """
    for attr, default in ((IS_BRIDGE, False), (FLOW_FROM, None), (FLOW_TO, None),
                          (N_BEHIND, None), (POWER_BEHIND, None)):
        nx.set_edge_attributes(H, default, attr)
    sums = _SubtreeSums.of(H, set(source_nodes.values()), building_power)
    for u, v in nx.bridges(H):
        d = H.edges[u, v]
        d[IS_BRIDGE] = True
        report.bridges += 1
        child, par = (v, u) if sums.parent.get(v) == u else (u, v)
        orientation = _bridge_orientation(child, par, sums)
        if orientation is not None:
            d[FLOW_FROM], d[FLOW_TO], d[N_BEHIND], d[POWER_BEHIND] = orientation
            report.bridges_with_fixed_direction += 1


def _bridge_orientation(child, par, sums):
    """(from, to, buildings behind, power behind [kW]), or None with sources on both sides."""
    n_total, p_total, s_total = sums.totals[child]
    if sums.sources[child] == 0:
        return par, child, sums.buildings[child], sums.power[child]
    if sums.sources[child] == s_total:
        return child, par, n_total - sums.buildings[child], p_total - sums.power[child]
    return None


def _total_length(G) -> float:
    return float(sum(d.get(cols.LENGTH, 0.0) for _, _, d in G.edges(data=True)))
