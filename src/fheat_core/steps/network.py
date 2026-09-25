"""Step NETWORK: street graph + network calculation + pipe sizing."""
from __future__ import annotations

import geopandas as gpd

from fheat_core import columns as cols
from fheat_core.algorithms.geometry import (
    add_centroids,
    closest_points_to_streets,
    insert_connection_points,
)
from fheat_core.algorithms.network import (
    add_edge_lengths,
    build_street_graph,
    compute_network,
    connect_buildings_to_graph,
    connect_source_to_graph,
)
from fheat_core.resources import load_pipe_info
from fheat_core.schemas import NetSchema
from fheat_core.state import Phase, PipelineState


def run(state: PipelineState, config, adapter) -> PipelineState:
    if config.network_method == "milp":
        return _run_milp(state, config, adapter)

    pipe_info = adapter.provide_pipe_info()
    if pipe_info is None:
        pipe_info = load_pipe_info()

    G, buildings, source = prepare_graph(
        state.buildings_gdf.copy(), state.streets_gdf.copy(), state.source_gdf.copy()
    )

    net_gdf = compute_network(
        G,
        buildings,
        source,
        pipe_info,
        power_att=cols.THERMAL_POWER,
        htemp=config.supply_temperature,
        ltemp=config.return_temperature,
        crs=buildings.crs,
    )

    NetSchema.validate(net_gdf)

    state.net_gdf = net_gdf
    state.phase = Phase.NETWORK
    return state


def prepare_graph(buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame, source: gpd.GeoDataFrame):
    """Street graph with house and source connections for every network method.

    Returns ``(G, buildings, source)``: the graph and the buildings with
    ``connect == 1`` and the source, both with centroid and connection point.
    """
    # restrict to connectable routes and buildings with heat connection
    if cols.ROUTABLE in streets.columns:
        streets = streets[streets[cols.ROUTABLE] == 1]
    if cols.CONNECT in buildings.columns:
        buildings = buildings[buildings[cols.CONNECT] == 1]

    # CRS: source to buildings CRS
    if source.crs != buildings.crs:
        source = source.to_crs(buildings.crs)

    # geometry enrichment
    buildings = add_centroids(buildings)
    buildings = closest_points_to_streets(buildings, streets, centroid_col="centroid")

    source = source.copy()
    source[cols.CENTROID] = source.geometry
    source = closest_points_to_streets(source, streets, centroid_col=cols.CENTROID)

    streets = insert_connection_points(streets, buildings)
    streets = insert_connection_points(streets, source)

    G = build_street_graph(streets, buildings.crs)
    G = connect_buildings_to_graph(G, buildings)
    G = connect_source_to_graph(G, source)
    G = add_edge_lengths(G)
    return G, buildings, source


def _run_milp(state: PipelineState, config, adapter) -> PipelineState:
    """network_method = "milp": optimised network, ``connect`` written back."""
    from fheat_core.optimization.network import REPORT_KEY, build_network

    net_gdf, candidates = build_network(
        state.buildings_gdf.copy(), state.streets_gdf.copy(), state.source_gdf.copy(), config, adapter
    )
    NetSchema.validate(net_gdf)

    buildings = state.buildings_gdf.copy()
    buildings[cols.CONNECTION_STATUS] = None
    buildings.loc[candidates.index, cols.CONNECT] = candidates[cols.CONNECT]
    buildings.loc[candidates.index, cols.CONNECTION_STATUS] = candidates[cols.CONNECTION_STATUS]

    state.optimization_report = net_gdf.attrs.pop(REPORT_KEY)
    state.buildings_gdf = buildings
    state.net_gdf = net_gdf
    state.phase = Phase.NETWORK
    return state
