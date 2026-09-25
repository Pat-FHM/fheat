"""Entry point of the MILP network optimisation: :func:`build_network`.

``(buildings, streets, source, config, adapter) -> NetworkResult(net_gdf,
buildings, report)``; unpacking the first two gives the shape of a network
backend. Steps:

1. street graph as for the shortest-path network (``steps.network.prepare_graph``),
2. graph simplification (``preprocess``),
3. linearised pipe costs and losses (``linearize``, ``glf_terms.design_loads``),
4. oemof.solph system with the network block, solved once (``energysystem``),
5. post-calculation with exact GLF and real DN (``postprocess``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import geopandas as gpd
import pandas as pd

from fheat_core import columns as cols
from fheat_core.optimization import STATUS_CONNECTED, STATUS_UNREACHABLE
from fheat_core.optimization.energysystem import MilpResult, pipe_annuity, solve_network
from fheat_core.optimization.glf_terms import design_loads
from fheat_core.optimization.linearize import linearize_pipes
from fheat_core.optimization.postprocess import PostCalculation, postprocess
from fheat_core.optimization.preprocess import SimplificationReport, building_ids, simplify_network
from fheat_core.resources import load_pipe_costs, load_pipe_info

@dataclass(frozen=True)
class OptimizationReport:
    """Everything the MILP run reports besides ``net_gdf``.

    ``fit_quality`` is ``PipeLinearization.report()`` (one row per cost/loss
    line and DN), ``unreachable`` the ``building_id``s (or index) of buildings
    without a route to the source.
    """

    milp: MilpResult
    post: PostCalculation
    simplification: SimplificationReport
    fit_quality: pd.DataFrame
    fit_warnings: tuple[str, ...]
    unreachable: list

    def summary(self) -> dict:
        """Key figures for the result summary (``milp_*`` keys)."""
        m, p = self.milp, self.post
        cost_lines = self.fit_quality[self.fit_quality["quantity"] == "cost"]
        return {
            "milp_glf_mode": m.glf_mode,
            "milp_termination": m.termination,
            "milp_objective_eur_a": round(m.objective, 1),
            "milp_gap_eur_a": None if m.achieved_gap is None else round(m.achieved_gap, 1),
            "milp_solve_time_s": round(m.solve_time_s, 2),
            "milp_unreachable_buildings": len(self.unreachable),
            "milp_producer_capacity_kw": round(p.producer_capacity, 1),
            "milp_producer_capacity_model_kw": round(m.source_capacity, 1),
            "milp_pipe_invest_eur": round(p.invest_cost, 0),
            "milp_pipe_invest_model_eur": round(p.invest_cost_model, 0),
            "milp_cost_line_deviation": round(p.cost_line_deviation, 4),
            "milp_pipe_annuity_eur_a": round(p.annual_cost, 1),
            "milp_cost_source_eur_a": round(m.objective_parts.source_invest, 1),
            "milp_cost_heat_eur_a": round(m.objective_parts.heat_demand, 1),
            "milp_cost_losses_eur_a": round(m.objective_parts.heat_losses, 1),
            "milp_cost_pipes_eur_a": round(m.objective_parts.pipes, 1),
            "milp_glf_max_deviation": round(p.glf_max_deviation, 4),
            "milp_capacity_max_deviation": round(p.capacity_max_deviation, 4),
            "milp_sections_above_largest_dn": p.sections_above_largest_dn,
            "milp_cost_line_max_deviation_per_dn": round(float(cost_lines["deviation"].abs().max()), 4),
            "milp_fit_warnings": len(self.fit_warnings),
        }


class NetworkResult(NamedTuple):
    """Result of :func:`build_network`.

    ``net_gdf``: ``NetSchema`` plus the optional model columns. ``buildings``:
    the buildings with ``connect == 1`` on input, with ``cols.CONNECT`` and
    ``cols.CONNECTION_STATUS`` updated.
    """

    net_gdf: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    report: OptimizationReport


def build_network(
    buildings: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    source: gpd.GeoDataFrame,
    config,
    adapter,
) -> NetworkResult:
    """MILP network for ``config.network_method = "milp"``."""
    from fheat_core.steps.network import prepare_graph

    opt = config.optimization
    pipe_info = _or_default(adapter.provide_pipe_info(), load_pipe_info)
    pipe_costs = _or_default(adapter.provide_pipe_costs(), load_pipe_costs)
    G, candidates, source = prepare_graph(buildings, streets, source)
    network = simplify_network(G, candidates, source, on_unreachable=opt.on_unreachable)
    network.source_node()   # exactly one heat source, clear error otherwise
    lin = linearize_pipes(
        pipe_info, pipe_costs, design_loads(network),
        config.supply_temperature, config.return_temperature,
        max_deviation=opt.regression_max_deviation,
    )
    milp = solve_network(network, lin, candidates[cols.HEAT_DEMAND].to_dict(), opt)
    net_gdf, post = postprocess(network, milp.edges, lin, pipe_info, pipe_costs, pipe_annuity(opt), candidates.crs)
    report = OptimizationReport(
        milp=milp,
        post=post,
        simplification=network.report,
        fit_quality=lin.report(),
        fit_warnings=lin.warnings,
        unreachable=building_ids(candidates, network.unreachable_buildings),
    )
    return NetworkResult(
        net_gdf, _connection_status(buildings.loc[candidates.index], network.unreachable_buildings), report
    )


def _or_default(provided, load):
    return load() if provided is None else provided


def _connection_status(candidates, unreachable) -> gpd.GeoDataFrame:
    """Candidate buildings with ``cols.CONNECT`` and ``cols.CONNECTION_STATUS`` updated."""
    out = candidates.copy()
    out[cols.CONNECTION_STATUS] = STATUS_CONNECTED
    out.loc[unreachable, cols.CONNECT] = 0
    out.loc[unreachable, cols.CONNECTION_STATUS] = STATUS_UNREACHABLE
    return out
