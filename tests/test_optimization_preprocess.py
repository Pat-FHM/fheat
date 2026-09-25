"""Tests for fheat_core.optimization.preprocess (T2).

Covers:
- integer node IDs and the ID ↔ coordinate mapping
- removal of dead ends without buildings (streets and loops) and of parts
  without a heat source (unreachable buildings are reported)
- merging of street chains into one edge per section (length, geometry)
- reduction of parallel sections to the shortest one
- bridges: detection, fixed flow direction, buildings and power behind
- total length balance and unchanged shortest paths source → building,
  also on random graphs with dead ends attached to mesh nodes
- on_unreachable = "warn" | "error"

The graphs are built with the same functions as ``steps/network.py``.
"""
from __future__ import annotations

import math
import random

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point, Polygon

from fheat_core import columns as cols
from fheat_core.algorithms.geometry import (
    add_centroids,
    closest_points_to_streets,
    insert_connection_points,
)
from fheat_core.algorithms.network import (
    add_edge_lengths,
    build_street_graph,
    connect_buildings_to_graph,
    connect_source_to_graph,
)
from fheat_core.optimization.preprocess import (
    BUILDING_KEY,
    COORD,
    FLOW_FROM,
    FLOW_TO,
    GEOMETRY,
    HOUSE_CONNECTION,
    IS_BRIDGE,
    KIND,
    KIND_BUILDING,
    KIND_JUNCTION,
    KIND_SOURCE,
    N_BEHIND,
    POWER_BEHIND,
    SOURCE_CONNECTION,
    STREET_PIPE,
    simplify_network,
)

CRS = "EPSG:25832"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _graph(streets, buildings, sources):
    """Build the F|Heat street graph from explicit connection points.

    streets:   list of coordinate lists (connection points must be vertices)
    buildings: list of (centroid, connection_point, power)
    sources:   list of (point, connection_point)
    """
    streets_gdf = gpd.GeoDataFrame({"geometry": [LineString(s) for s in streets]}, crs=CRS)
    b = gpd.GeoDataFrame(
        {
            cols.CENTROID: [Point(c) for c, _, _ in buildings],
            cols.CONNECTION_POINT: [Point(cp) if cp is not None else None for _, cp, _ in buildings],
            cols.THERMAL_POWER: [p for _, _, p in buildings],
            "geometry": [Point(c) for c, _, _ in buildings],
        },
        crs=CRS,
    )
    s = gpd.GeoDataFrame(
        {
            cols.CONNECTION_POINT: [Point(cp) for _, cp in sources],
            "geometry": [Point(p) for p, _ in sources],
        },
        crs=CRS,
    )
    G = build_street_graph(streets_gdf, CRS)
    G = connect_buildings_to_graph(G, b)
    G = connect_source_to_graph(G, s)
    G = add_edge_lengths(G)
    return G, b, s


def _length(G):
    return sum(d[cols.LENGTH] for _, _, d in G.edges(data=True))


def _assert_length_balance(G, net):
    r = net.report
    removed = (
        r.disconnected_length_removed
        + r.dead_end_length_removed
        + r.parallel_length_removed
        + r.self_loop_length_removed
    )
    assert r.length_before == pytest.approx(_length(G))
    assert r.length_after == pytest.approx(_length(net.graph))
    assert r.length_before == pytest.approx(r.length_after + removed)


def _assert_shortest_paths_unchanged(G, b, s, net):
    src = s.geometry.iloc[0].coords[0]
    src_id = net.node_ids[src]
    for key, node in net.building_nodes.items():
        c = b.at[key, cols.CENTROID]
        before = nx.shortest_path_length(G, src, (c.x, c.y), weight=cols.LENGTH)
        after = nx.shortest_path_length(net.graph, src_id, node, weight=cols.LENGTH)
        assert after == pytest.approx(before), key


def _assert_bridges_match_brute_force(net):
    """Remove every edge once and compare with the stored bridge attributes."""
    H = net.graph
    sources = set(net.source_nodes.values())
    buildings = {n: H.nodes[n] for n in net.building_nodes.values()}
    for u, v, d in H.edges(data=True):
        K = H.copy()
        K.remove_edge(u, v)
        comp_u = nx.node_connected_component(K, u)
        is_bridge = v not in comp_u
        assert d[IS_BRIDGE] == is_bridge, (u, v)
        if not is_bridge:
            assert d[FLOW_FROM] is None
            continue
        comp_v = nx.node_connected_component(K, v)
        if not sources & comp_v:
            behind, expected_from = comp_v, u
        elif not sources & comp_u:
            behind, expected_from = comp_u, v
        else:
            assert d[FLOW_FROM] is None
            continue
        assert d[FLOW_FROM] == expected_from
        assert d[N_BEHIND] == len(behind & buildings.keys())


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def line_case():
    """Street with a tail, a dead-end branch and a loop without buildings.

        S ── (0,0) ─ (10,0) ─ (20,0) ─ (30,0) ─ (40,0)   tail (30→40) without building
                       │  ▲      │        │
                      B0 loop  branch    B1
    """
    streets = [
        [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)],
        [(20, 0), (20, -10), (20, -20)],                 # dead-end branch
        [(10, 0), (5, -10), (15, -10), (10, 0)],         # loop hanging off (10, 0)
    ]
    buildings = [((10, 10), (10, 0), 10.0), ((30, 10), (30, 0), 20.0)]
    sources = [((-10, 0), (0, 0))]
    return _graph(streets, buildings, sources)


@pytest.fixture
def ring_case():
    """Square ring (50 m) with a spur; source at (0,0), three buildings.

    Ring vertices every 25 m; spur (0,50) → (0,75) → (0,100).
    """
    streets = [
        [(0, 0), (25, 0), (50, 0), (50, 25), (50, 50), (25, 50), (0, 50), (0, 25), (0, 0)],
        [(0, 50), (0, 75), (0, 100)],
    ]
    buildings = [
        ((25, -10), (25, 0), 10.0),
        ((60, 25), (50, 25), 20.0),
        ((10, 100), (0, 100), 30.0),
    ]
    sources = [((-10, 0), (0, 0))]
    return _graph(streets, buildings, sources)


@pytest.fixture
def parallel_case():
    """Two routes between (0,0) and (100,0): 100 m straight, 200 m detour."""
    streets = [
        [(0, 0), (50, 0), (100, 0)],
        [(0, 0), (0, 50), (100, 50), (100, 0)],
    ]
    buildings = [((110, 0), (100, 0), 50.0)]
    sources = [((-10, 0), (0, 0))]
    return _graph(streets, buildings, sources)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestNodeIds:
    def test_integer_ids_and_mapping(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        assert all(isinstance(n, int) for n in net.graph.nodes)
        assert len(net.node_coords) == G.number_of_nodes()
        for i, coord in net.node_coords.items():
            assert net.node_ids[coord] == i
        for n, data in net.graph.nodes(data=True):
            assert data[COORD] == net.node_coords[n]

    def test_node_kinds(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        kinds = nx.get_node_attributes(net.graph, KIND)
        assert {kinds[n] for n in net.building_nodes.values()} == {KIND_BUILDING}
        assert [kinds[n] for n in net.source_nodes.values()] == [KIND_SOURCE]
        for key, n in net.building_nodes.items():
            assert net.graph.nodes[n][BUILDING_KEY] == key

    def test_input_graph_unchanged(self, ring_case):
        G, b, s = ring_case
        before = (G.number_of_nodes(), G.number_of_edges(), _length(G))
        simplify_network(G, b, s)
        assert (G.number_of_nodes(), G.number_of_edges(), _length(G)) == before


class TestDeadEnds:
    def test_dead_ends_removed(self, line_case):
        G, b, s = line_case
        net = simplify_network(G, b, s)
        coords = {net.node_coords[n] for n in net.graph}
        for gone in [(40, 0), (20, -10), (20, -20), (5, -10), (15, -10)]:
            assert gone not in coords, gone
        # tail 10 m + branch 20 m + loop (2 · √125 + 10 m)
        assert net.report.dead_end_length_removed == pytest.approx(30 + 2 * 125 ** 0.5 + 10)

    def test_no_junction_leaves_left(self, line_case):
        G, b, s = line_case
        net = simplify_network(G, b, s)
        for n, kind in net.graph.nodes(data=KIND):
            if kind == KIND_JUNCTION:
                assert net.graph.degree(n) >= 2

    def test_length_balance(self, line_case):
        G, b, s = line_case
        _assert_length_balance(G, simplify_network(G, b, s))

    def test_shortest_paths_unchanged(self, line_case):
        G, b, s = line_case
        _assert_shortest_paths_unchanged(G, b, s, simplify_network(G, b, s))

    def test_dead_end_at_mesh_node_keeps_mesh_node(self):
        """Regression: B carries a dead end but also belongs to the mesh A-B-C-D.

        A(0,0)-B(10,0) and B-C(10,10) are 10 m, A-D(0,10) and D-C 50 m each
        (via a detour vertex). Source at A, building at C, dead end B-E(20,0).
        Removing B together with its dead end raised the path 30 m → 110 m.
        """
        dx = math.sqrt(25 ** 2 - 5 ** 2)  # detour vertex: two 25 m segments
        G, b, s = _graph(
            streets=[
                [(0, 0), (10, 0)],
                [(10, 0), (10, 10)],
                [(0, 0), (-dx, 5), (0, 10)],
                [(0, 10), (5, 10 + dx), (10, 10)],
                [(10, 0), (20, 0)],
            ],
            buildings=[((15, 10), (10, 10), 10.0)],
            sources=[((-5, 0), (0, 0))],
        )
        assert nx.shortest_path_length(G, (-5, 0), (15, 10), weight=cols.LENGTH) == pytest.approx(30)
        net = simplify_network(G, b, s)
        coords = {net.node_coords[n] for n in net.graph}
        assert (20, 0) not in coords
        assert net.report.dead_end_length_removed == pytest.approx(10)
        # without the dead end B is a plain street vertex: A-B-C becomes one
        # 20 m section, which makes the 100 m detour via D parallel to it
        a, c = net.node_ids[(0, 0)], net.node_ids[(10, 10)]
        assert net.graph.edges[a, c][cols.LENGTH] == pytest.approx(20)
        assert list(net.graph.edges[a, c][GEOMETRY].coords) in (
            [(0, 0), (10, 0), (10, 10)], [(10, 10), (10, 0), (0, 0)]
        )
        assert net.report.parallel_length_removed == pytest.approx(100)
        _assert_shortest_paths_unchanged(G, b, s, net)
        _assert_length_balance(G, net)


class TestMergeChains:
    def test_one_edge_per_section(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        street = {
            frozenset((net.node_coords[u], net.node_coords[v])): d[cols.LENGTH]
            for u, v, d in net.graph.edges(data=True)
            if d[cols.TYPE] == STREET_PIPE
        }
        assert street == {
            frozenset(((0, 0), (25, 0))): pytest.approx(25),
            frozenset(((25, 0), (50, 25))): pytest.approx(50),
            frozenset(((50, 25), (0, 50))): pytest.approx(75),
            frozenset(((0, 50), (0, 0))): pytest.approx(50),
            frozenset(((0, 50), (0, 100))): pytest.approx(50),
        }
        assert net.report.nodes_merged == 5

    def test_geometry_matches_length_and_endpoints(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        for u, v, d in net.graph.edges(data=True):
            geom = d[GEOMETRY]
            assert geom.length == pytest.approx(d[cols.LENGTH])
            ends = {geom.coords[0], geom.coords[-1]}
            assert ends == {net.node_coords[u], net.node_coords[v]}

    def test_no_mergeable_node_left(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        H = net.graph
        for n, kind in H.nodes(data=KIND):
            if kind == KIND_JUNCTION and H.degree(n) == 2:
                types = {d[cols.TYPE] for _, _, d in H.edges(n, data=True)}
                assert types != {STREET_PIPE}

    def test_connection_edges_kept(self, ring_case):
        G, b, s = ring_case
        H = simplify_network(G, b, s).graph
        types = [d[cols.TYPE] for _, _, d in H.edges(data=True)]
        assert types.count(HOUSE_CONNECTION) == 3
        assert types.count(SOURCE_CONNECTION) == 1

    def test_length_balance(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        _assert_length_balance(G, net)
        assert net.report.length_after == pytest.approx(net.report.length_before)


class TestParallelEdges:
    def test_shortest_parallel_kept(self, parallel_case):
        G, b, s = parallel_case
        net = simplify_network(G, b, s)
        a, c = net.node_ids[(0, 0)], net.node_ids[(100, 0)]
        assert net.graph.edges[a, c][cols.LENGTH] == pytest.approx(100)
        assert net.report.parallel_edges_removed == 1
        assert net.report.parallel_length_removed == pytest.approx(200)
        _assert_length_balance(G, net)

    def test_simple_graph(self, parallel_case):
        G, b, s = parallel_case
        H = simplify_network(G, b, s).graph
        assert not H.is_multigraph()
        assert nx.number_of_selfloops(H) == 0

    def test_all_edges_become_bridges(self, parallel_case):
        G, b, s = parallel_case
        net = simplify_network(G, b, s)
        assert all(d[IS_BRIDGE] for _, _, d in net.graph.edges(data=True))


class TestBridges:
    def test_ring_bridges(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        H = net.graph
        bridges = {frozenset((net.node_coords[u], net.node_coords[v])) for u, v in net.bridges()}
        assert bridges == {
            frozenset(((25, -10), (25, 0))),
            frozenset(((60, 25), (50, 25))),
            frozenset(((10, 100), (0, 100))),
            frozenset(((-10, 0), (0, 0))),
            frozenset(((0, 50), (0, 100))),
        }
        assert net.report.bridges == 5
        assert net.report.bridges_with_fixed_direction == 5

    def test_house_connections_flow_to_building(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        H = net.graph
        for key, node in net.building_nodes.items():
            (street_node,) = H.neighbors(node)
            d = H.edges[node, street_node]
            assert d[IS_BRIDGE]
            assert (d[FLOW_FROM], d[FLOW_TO]) == (street_node, node)
            assert d[N_BEHIND] == 1
            assert d[POWER_BEHIND] == pytest.approx(b.at[key, cols.THERMAL_POWER])

    def test_source_connection_carries_all(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        src = net.source_nodes[0]
        (street_node,) = net.graph.neighbors(src)
        d = net.graph.edges[src, street_node]
        assert (d[FLOW_FROM], d[FLOW_TO]) == (src, street_node)
        assert d[N_BEHIND] == 3
        assert d[POWER_BEHIND] == pytest.approx(60.0)

    def test_spur_behind_ring(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        u, v = net.node_ids[(0, 50)], net.node_ids[(0, 100)]
        d = net.graph.edges[u, v]
        assert (d[FLOW_FROM], d[FLOW_TO]) == (u, v)
        assert (d[N_BEHIND], d[POWER_BEHIND]) == (1, pytest.approx(30.0))

    def test_ring_edges_are_no_bridges(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s)
        u, v = net.node_ids[(0, 0)], net.node_ids[(25, 0)]
        d = net.graph.edges[u, v]
        assert not d[IS_BRIDGE]
        assert d[FLOW_FROM] is None and d[N_BEHIND] is None

    def test_brute_force(self, line_case, ring_case, parallel_case):
        for G, b, s in (line_case, ring_case, parallel_case):
            _assert_bridges_match_brute_force(simplify_network(G, b, s))

    def test_two_sources(self):
        """A bridge with a source on both sides has no fixed direction."""
        G, b, s = _graph(
            streets=[[(0, 0), (50, 0), (100, 0)]],
            buildings=[((50, 10), (50, 0), 10.0)],
            sources=[((-10, 0), (0, 0)), ((110, 0), (100, 0))],
        )
        net = simplify_network(G, b, s)
        H = net.graph
        mid = net.node_ids[(50, 0)]
        for end in ((0, 0), (100, 0)):
            d = H.edges[mid, net.node_ids[end]]
            assert d[IS_BRIDGE] and d[FLOW_FROM] is None
        house = H.edges[mid, net.building_nodes[0]]
        assert (house[FLOW_FROM], house[N_BEHIND]) == (mid, 1)
        _assert_bridges_match_brute_force(net)


class TestUnreachable:
    def test_building_without_connection_point(self, ring_case):
        G, b, s = ring_case
        b = b.copy()
        extra = gpd.GeoDataFrame(
            {cols.CENTROID: [Point(500, 500)], cols.CONNECTION_POINT: [None],
             cols.THERMAL_POWER: [5.0], "geometry": [Point(500, 500)]},
            index=[99], crs=CRS,
        )
        b = gpd.GeoDataFrame(gpd.pd.concat([b, extra]), crs=CRS)
        net = simplify_network(G, b, s)
        assert net.unreachable_buildings == [99]
        assert 99 not in net.building_nodes

    def test_disconnected_part_removed(self, caplog):
        G, b, s = _graph(
            streets=[[(0, 0), (50, 0)], [(200, 0), (250, 0)]],
            buildings=[((50, 10), (50, 0), 10.0), ((250, 10), (250, 0), 10.0)],
            sources=[((-10, 0), (0, 0))],
        )
        with caplog.at_level("WARNING", logger="fheat_core.optimization.preprocess"):
            net = simplify_network(G, b, s)
        assert net.unreachable_buildings == [1]
        assert net.report.unreachable_buildings == 1
        assert net.report.disconnected_length_removed == pytest.approx(60)
        assert any("cannot be reached" in r.getMessage() for r in caplog.records)
        _assert_length_balance(G, net)

    def test_warning_names_building_ids(self, caplog):
        G, b, s = _graph(
            streets=[[(0, 0), (50, 0)], [(200, 0), (250, 0)]],
            buildings=[((50, 10), (50, 0), 10.0), ((250, 10), (250, 0), 10.0)],
            sources=[((-10, 0), (0, 0))],
        )
        b[cols.BUILDING_ID] = ["B-17", "B-42"]
        with caplog.at_level("WARNING", logger="fheat_core.optimization.preprocess"):
            simplify_network(G, b, s, on_unreachable="warn")
        (msg,) = [r.getMessage() for r in caplog.records if "cannot be reached" in r.getMessage()]
        assert "B-42" in msg and "B-17" not in msg

    def test_error_mode_raises(self):
        G, b, s = _graph(
            streets=[[(0, 0), (50, 0)], [(200, 0), (250, 0)]],
            buildings=[((50, 10), (50, 0), 10.0), ((250, 10), (250, 0), 10.0)],
            sources=[((-10, 0), (0, 0))],
        )
        with pytest.raises(ValueError, match="cannot be reached"):
            simplify_network(G, b, s, on_unreachable="error")

    def test_error_mode_passes_when_all_reachable(self, ring_case):
        G, b, s = ring_case
        net = simplify_network(G, b, s, on_unreachable="error")
        assert net.unreachable_buildings == []

    def test_invalid_mode_raises(self, ring_case):
        G, b, s = ring_case
        with pytest.raises(ValueError, match="on_unreachable"):
            simplify_network(G, b, s, on_unreachable="ignore")


class TestInputErrors:
    def test_duplicate_centroid_raises(self, ring_case):
        G, b, s = ring_case
        b = b.copy()
        b[cols.CENTROID] = [b.at[0, cols.CENTROID]] * len(b)
        with pytest.raises(ValueError, match="share the centroid"):
            simplify_network(G, b, s)

    def test_source_not_in_graph_raises(self, ring_case):
        G, b, _ = ring_case
        s = gpd.GeoDataFrame({"geometry": [Point(999, 999)]}, crs=CRS)
        with pytest.raises(ValueError, match="not connected"):
            simplify_network(G, b, s)

    def test_empty_source_raises(self, ring_case):
        G, b, s = ring_case
        with pytest.raises(ValueError, match="empty"):
            simplify_network(G, b, s.iloc[:0])


class TestPipelineGraph:
    """Graph built exactly like steps/network.py from streets, buildings and a source."""

    @pytest.fixture
    def pipeline_case(self):
        # 3 × 3 street grid (100 m blocks), every street split into 10 m vertices
        lines = []
        for k in range(3):
            lines.append([(x, 100.0 * k) for x in range(0, 201, 10)])
            lines.append([(100.0 * k, y) for y in range(0, 201, 10)])
        streets = gpd.GeoDataFrame({cols.ROUTABLE: [1] * len(lines),
                                    "geometry": [LineString(l) for l in lines]}, crs=CRS)
        polys, power = [], []
        for i, (x, y) in enumerate([(33, 12), (67, 112), (145, 188), (188, 55), (12, 160), (120, 88)]):
            polys.append(Polygon([(x - 4, y - 4), (x + 4, y - 4), (x + 4, y + 4), (x - 4, y + 4)]))
            power.append(10.0 + i)
        buildings = gpd.GeoDataFrame({cols.THERMAL_POWER: power, "geometry": polys}, crs=CRS)
        source = gpd.GeoDataFrame({"geometry": [Point(-15, -15)]}, crs=CRS)

        buildings = add_centroids(buildings)
        buildings = closest_points_to_streets(buildings, streets, centroid_col=cols.CENTROID)
        source[cols.CENTROID] = source.geometry
        source = closest_points_to_streets(source, streets, centroid_col=cols.CENTROID)
        streets = insert_connection_points(streets, buildings)
        streets = insert_connection_points(streets, source)
        G = build_street_graph(streets, CRS)
        G = connect_buildings_to_graph(G, buildings)
        G = connect_source_to_graph(G, source)
        G = add_edge_lengths(G)
        return G, buildings, source

    def test_reduces_graph(self, pipeline_case):
        G, b, s = pipeline_case
        net = simplify_network(G, b, s)
        assert net.graph.number_of_nodes() < G.number_of_nodes() / 3
        assert net.report.nodes_merged > 0
        assert set(net.building_nodes) == set(b.index)

    def test_length_balance(self, pipeline_case):
        G, b, s = pipeline_case
        _assert_length_balance(G, simplify_network(G, b, s))

    def test_shortest_paths_unchanged(self, pipeline_case):
        G, b, s = pipeline_case
        _assert_shortest_paths_unchanged(G, b, s, simplify_network(G, b, s))

    def test_bridges_brute_force(self, pipeline_case):
        G, b, s = pipeline_case
        _assert_bridges_match_brute_force(simplify_network(G, b, s))

    def test_simplification_is_idempotent(self, pipeline_case):
        """Running the simplification on its own output changes nothing."""
        G, b, s = pipeline_case
        net = simplify_network(G, b, s)
        coord_graph = nx.relabel_nodes(net.graph, net.node_coords)
        again = simplify_network(coord_graph, b, s)
        assert again.graph.number_of_edges() == net.graph.number_of_edges()
        assert again.report.dead_end_edges_removed == 0
        assert again.report.parallel_edges_removed == 0
        assert again.report.nodes_merged == 0


# ---------------------------------------------------------------------------
# random graphs
# ---------------------------------------------------------------------------


def _random_case(seed):
    """Irregular mesh with dead ends (chains, branches, loops) at mesh nodes.

    Buildings sit on mesh nodes and dead-end nodes; some dead ends and some
    mesh parts stay without buildings, some buildings end up unreachable.
    """
    rng = random.Random(seed)
    m = rng.randint(3, 6)
    pos = {
        (i, j): (i * 50 + rng.uniform(-10, 10), j * 50 + rng.uniform(-10, 10))
        for i in range(m) for j in range(m)
    }
    streets = []
    for i in range(m):
        for j in range(m):
            for di, dj in ((1, 0), (0, 1)):
                if (i + di, j + dj) in pos and rng.random() < 0.75:
                    streets.append([pos[i, j], pos[i + di, j + dj]])
    mesh = sorted({p for line in streets for p in line})
    if not mesh:
        return _random_case(seed + 1000)

    dead_end_nodes = []
    for _ in range(rng.randint(1, 3 * m)):
        base = rng.choice(mesh)
        chain = [base]
        for _ in range(rng.randint(1, 3)):
            x, y = chain[-1]
            chain.append((x + rng.uniform(-20, 20), y + rng.uniform(-20, 20)))
        streets.append(chain)
        dead_end_nodes += chain[1:]
        roll = rng.random()
        if roll < 0.3:                       # branch off the dead end
            x, y = rng.choice(chain[1:])
            branch = [(x, y), (x + rng.uniform(-15, 15), y + rng.uniform(-15, 15))]
            streets.append(branch)
            dead_end_nodes.append(branch[1])
        elif roll < 0.5:                     # loop at the end of the dead end
            x, y = chain[-1]
            a = (x + rng.uniform(5, 15), y + rng.uniform(5, 15))
            c = (x - rng.uniform(5, 15), y + rng.uniform(5, 15))
            streets.append([chain[-1], a, c, chain[-1]])
            dead_end_nodes += [a, c]

    candidates = mesh + dead_end_nodes
    buildings = []
    for k in range(rng.randint(1, 8)):
        x, y = rng.choice(candidates)
        buildings.append(((x + 3 + 0.01 * k, y + 4), (x, y), rng.uniform(5, 50)))
    x, y = rng.choice(mesh)
    sources = [((x - 7, y - 3), (x, y))]
    return _graph(streets, buildings, sources)


class TestRandomGraphs:
    @staticmethod
    def _simplify(G, b, s):
        src = s.geometry.iloc[0].coords[0]
        if not any(nx.has_path(G, src, (c.x, c.y)) for c in b[cols.CENTROID] if (c.x, c.y) in G):
            with pytest.raises(ValueError, match="No building"):
                simplify_network(G, b, s)
            return None
        return simplify_network(G, b, s)

    @pytest.mark.parametrize("seed", range(60))
    def test_shortest_paths_unchanged(self, seed):
        G, b, s = _random_case(seed)
        net = self._simplify(G, b, s)
        if net is None:
            return
        _assert_shortest_paths_unchanged(G, b, s, net)
        src = s.geometry.iloc[0].coords[0]
        for key in net.unreachable_buildings:
            c = b.at[key, cols.CENTROID]
            assert not nx.has_path(G, src, (c.x, c.y))

    @pytest.mark.parametrize("seed", range(60))
    def test_structure(self, seed):
        G, b, s = _random_case(seed)
        net = self._simplify(G, b, s)
        if net is None:
            return
        _assert_length_balance(G, net)
        _assert_bridges_match_brute_force(net)
        H = net.graph
        for n, kind in H.nodes(data=KIND):
            if kind == KIND_JUNCTION:
                assert H.degree(n) >= 2, n
        assert nx.number_of_selfloops(H) == 0
