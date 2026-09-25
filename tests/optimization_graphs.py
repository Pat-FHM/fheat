"""Graphs for the optimisation tests, built with the functions of ``steps/network.py``."""
from __future__ import annotations

import random

import geopandas as gpd
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

CRS = "EPSG:25832"


def street_graph(streets, buildings, sources):
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


def pipeline_graph(lines, centroids, powers, source_point, size=8.0):
    """Street graph exactly as ``steps/network.run`` builds it.

    lines: street coordinate lists; centroids: building centres (square
    buildings of ``size`` m); powers [kW]; source_point: heat source.
    """
    streets = gpd.GeoDataFrame({cols.ROUTABLE: [1] * len(lines),
                                "geometry": [LineString(line) for line in lines]}, crs=CRS)
    h = size / 2
    polys = [Polygon([(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]) for x, y in centroids]
    buildings = gpd.GeoDataFrame({cols.THERMAL_POWER: list(powers), "geometry": polys}, crs=CRS)
    source = gpd.GeoDataFrame({"geometry": [Point(source_point)]}, crs=CRS)

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


def grid_case(blocks, n_buildings, seed, block_length=100.0):
    """Street grid with ``blocks`` × ``blocks`` blocks and random buildings.

    Streets have a vertex every 10 m; buildings lie 6-20 m next to a random
    street, 20-80 kW each; the source sits at the lower left corner.
    """
    rng = random.Random(seed)
    extent = blocks * block_length
    ticks = [10.0 * i for i in range(int(extent / 10) + 1)]
    lines = []
    for k in range(blocks + 1):
        lines.append([(x, k * block_length) for x in ticks])
        lines.append([(k * block_length, y) for y in ticks])
    centroids = set()
    while len(centroids) < n_buildings:
        k = rng.randint(0, blocks)
        along = round(rng.uniform(5, extent - 5), 1)
        offset = rng.choice((-1, 1)) * round(rng.uniform(6, 20), 1)
        pos = (along, k * block_length + offset) if rng.random() < 0.5 else (k * block_length + offset, along)
        centroids.add(pos)
    centroids = sorted(centroids)
    powers = [round(rng.uniform(20, 80), 1) for _ in centroids]
    return pipeline_graph(lines, centroids, powers, (-15.0, -15.0))


def random_case(seed):
    """Irregular mesh with dead ends (chains, branches, loops) at mesh nodes.

    Buildings sit on mesh nodes and dead-end nodes; some dead ends and some
    mesh parts stay without buildings, some buildings end up unreachable.
    """
    rng = random.Random(seed)
    m, streets = _random_mesh(rng)
    mesh = sorted({p for line in streets for p in line})
    if not mesh:
        return random_case(seed + 1000)
    dead_end_nodes = []
    for _ in range(rng.randint(1, 3 * m)):
        lines, nodes = _random_dead_end(rng, rng.choice(mesh))
        streets += lines
        dead_end_nodes += nodes
    buildings, sources = _random_terminals(rng, mesh, mesh + dead_end_nodes)
    return street_graph(streets, buildings, sources)


def _random_mesh(rng):
    """m × m grid (50 m, jittered nodes), each grid street kept with 75 %."""
    m = rng.randint(3, 6)
    pos = {
        (i, j): (i * 50 + rng.uniform(-10, 10), j * 50 + rng.uniform(-10, 10))
        for i in range(m) for j in range(m)
    }
    streets = []
    for (i, j), p in pos.items():
        for di, dj in ((1, 0), (0, 1)):
            if (i + di, j + dj) in pos and rng.random() < 0.75:
                streets.append([p, pos[i + di, j + dj]])
    return m, streets


def _random_dead_end(rng, base):
    """Chain of 1-3 segments from ``base``, sometimes with a branch or an end loop."""
    chain = [base]
    for _ in range(rng.randint(1, 3)):
        x, y = chain[-1]
        chain.append((x + rng.uniform(-20, 20), y + rng.uniform(-20, 20)))
    lines, nodes = [chain], chain[1:]
    roll = rng.random()
    if roll < 0.3:                       # branch off the dead end
        x, y = rng.choice(chain[1:])
        branch = [(x, y), (x + rng.uniform(-15, 15), y + rng.uniform(-15, 15))]
        lines.append(branch)
        nodes.append(branch[1])
    elif roll < 0.5:                     # loop at the end of the dead end
        x, y = chain[-1]
        a = (x + rng.uniform(5, 15), y + rng.uniform(5, 15))
        c = (x - rng.uniform(5, 15), y + rng.uniform(5, 15))
        lines.append([chain[-1], a, c, chain[-1]])
        nodes += [a, c]
    return lines, nodes


def _random_terminals(rng, mesh, candidates):
    """1-8 buildings on random candidate nodes, one source on a mesh node."""
    buildings = []
    for k in range(rng.randint(1, 8)):
        x, y = rng.choice(candidates)
        buildings.append(((x + 3 + 0.01 * k, y + 4), (x, y), rng.uniform(5, 50)))
    x, y = rng.choice(mesh)
    return buildings, [((x - 7, y - 3), (x, y))]


def two_cluster_case(n_per_cluster=50, power=100.0):
    """Two clusters A and B with a direct feeder each or a shared trunk.

    Source at S(0, 0); S-A and S-B are 90 m, the trunk S-J 60 m and the
    branches J-A, J-B 50 m each. ``n_per_cluster`` buildings of ``power`` [kW]
    are connected at A and at B. Returns the graph data and the node
    coordinates {"S", "J", "A", "B"}.
    """
    x = (90 ** 2 - 50 ** 2 + 60 ** 2) / (2 * 60)
    y = (90 ** 2 - x ** 2) ** 0.5
    nodes = {"S": (0.0, 0.0), "J": (60.0, 0.0), "A": (x, y), "B": (x, -y)}
    streets = [[nodes[p], nodes[q]] for p, q in (("S", "A"), ("S", "B"), ("S", "J"), ("J", "A"), ("J", "B"))]
    buildings = [
        ((cx + 0.5 * (i % 10) + 1, cy + sign * (1 + 0.5 * (i // 10))), (cx, cy), power)
        for (cx, cy), sign in ((nodes["A"], 1), (nodes["B"], -1))
        for i in range(n_per_cluster)
    ]
    return street_graph(streets, buildings, [((-5.0, 0.0), nodes["S"])]), nodes
