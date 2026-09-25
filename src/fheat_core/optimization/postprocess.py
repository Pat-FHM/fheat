"""Post-calculation of the MILP network with exact GLF and real DN (one pass).

The solution is a tree, so every built section carries a known number of
buildings n and summed connection power S [kW]. Per built section:

1. exact simultaneity factor GLF(n) (``calculate_glf``) and design power
   GLF(n) · S [kW];
2. volume flow and the next larger DN (``calculate_volumeflow``,
   ``calculate_diameter_velocity_loss``, including its house connection rule);
3. heat loss [kWh/a] of the DN for both U-values (``linearize.trench_loss``
   with the soil temperature of the model), pipe investment [€] from the
   pipe costs of the DN and its annuity [€/a].

The model values (``cols.GLF_MODEL``, ``cols.CAPACITY_MODEL``,
``cols.INVEST_COST_MODEL``) stay next to the post-calculated ones. The
producer capacity is GLF(N) · ΣQ + Σ heat loss [kW].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from fheat_core import columns as cols
from fheat_core.algorithms.network import (
    calculate_diameter_velocity_loss,
    calculate_glf,
    calculate_volumeflow,
)
from fheat_core.optimization import HOURS_PER_YEAR, SOURCE_CONNECTION
from fheat_core.optimization.linearize import PipeLinearization, merge_pipe_costs, trench_loss
from fheat_core.optimization.preprocess import GEOMETRY, SimplifiedNetwork

logger = logging.getLogger(__name__)

_NET_COLUMNS = [
    cols.TYPE, cols.LENGTH, cols.THERMAL_POWER, cols.N_BUILDINGS, cols.GLF,
    cols.THERMAL_POWER_GLF, cols.VOLUME_FLOW, cols.NOMINAL_DIAMETER, cols.VELOCITY,
    cols.HEAT_LOSS, cols.HEAT_LOSS_EXTRA_INSULATION,
    cols.GLF_MODEL, cols.CAPACITY_MODEL, cols.INVEST_COST_MODEL, cols.INVEST_COST, cols.ANNUAL_COST,
]


@dataclass(frozen=True)
class PostCalculation:
    """Totals of the post-calculation."""

    producer_capacity: float        # GLF(N) · ΣQ + heat loss [kW]
    heat_loss: float                # [kW], standard U-value
    invest_cost: float              # pipe investment of the chosen DNs [€]
    invest_cost_model: float        # linearised pipe investment of the MILP [€]
    annual_cost: float              # annuity of the pipe investment [€/a]
    glf_max_deviation: float        # max |GLF_model / GLF − 1| over the built sections
    capacity_max_deviation: float   # max |C_model / (GLF · S) − 1| over the built sections
    sections_above_largest_dn: int  # design power above the largest DN of the catalogue

    @property
    def cost_line_deviation(self) -> float:
        """Linearised over real pipe investment minus 1; negative: the cost line is too low."""
        return self.invest_cost_model / self.invest_cost - 1


def postprocess(
    network: SimplifiedNetwork,
    edges: pd.DataFrame,
    linearization: PipeLinearization,
    pipe_info: pd.DataFrame,
    pipe_costs: pd.DataFrame,
    pipe_annuity: float,
    crs,
) -> tuple[gpd.GeoDataFrame, PostCalculation]:
    """``net_gdf`` of the built sections (``NetSchema``) and the totals.

    ``edges`` is ``MilpResult.edges``, ``pipe_annuity`` the annuity factor of
    the pipes [1/a]. Geometries point in flow direction.
    """
    catalogue = merge_pipe_costs(pipe_info, pipe_costs).set_index("DN")
    built = edges[edges["built"]].to_dict("records")
    rows = [_section(row, network, linearization, pipe_info, catalogue, pipe_annuity) for row in built]
    net_gdf = gpd.GeoDataFrame(
        [{k: row[k] for k in _NET_COLUMNS} for row in rows],
        geometry=[row["geometry"] for row in rows],
        crs=crs,
    )
    return net_gdf, _totals(net_gdf, rows)


def _section(row, network, lin, pipe_info, catalogue, pipe_annuity) -> dict:
    """Post-calculated values of one built section (steps 1 to 3)."""
    htemp, ltemp = lin.supply_temperature, lin.return_temperature
    length = row[cols.LENGTH]
    n = int(round(row[cols.N_BUILDINGS]))
    glf = calculate_glf(n)
    power_glf = glf * row[cols.THERMAL_POWER]
    volume_flow = calculate_volumeflow(power_glf, htemp, ltemp)
    dn, velocity, _, _ = calculate_diameter_velocity_loss(
        volume_flow, htemp, ltemp, length, pipe_info, row[cols.TYPE]
    )
    pipe = catalogue.loc[dn]
    invest = pipe["cost_eur_per_m"] * length
    return {
        cols.TYPE: row[cols.TYPE],
        cols.LENGTH: length,
        cols.THERMAL_POWER: row[cols.THERMAL_POWER],
        cols.N_BUILDINGS: n,
        cols.GLF: glf,
        cols.THERMAL_POWER_GLF: power_glf,
        cols.VOLUME_FLOW: volume_flow,
        cols.NOMINAL_DIAMETER: dn,
        cols.VELOCITY: velocity,
        cols.HEAT_LOSS: _annual_loss(pipe["U-Value"], lin, length),
        cols.HEAT_LOSS_EXTRA_INSULATION: _annual_loss(pipe["U-Value_extra_insulation"], lin, length),
        cols.GLF_MODEL: row[cols.GLF_MODEL],
        cols.CAPACITY_MODEL: row[cols.CAPACITY_MODEL],
        cols.INVEST_COST_MODEL: row[cols.INVEST_COST_MODEL],
        cols.INVEST_COST: invest,
        cols.ANNUAL_COST: pipe_annuity * invest,
        "above_largest_dn": volume_flow > pipe["max_volumeFlow"],
        "geometry": _flow_geometry(network, row),
    }


def _annual_loss(u_value, lin, length) -> float:
    """Heat loss [kWh/a] of a section."""
    w_per_m = trench_loss(u_value, lin.supply_temperature, lin.return_temperature, lin.soil_temperature)
    return w_per_m * length * HOURS_PER_YEAR / 1000


def _flow_geometry(network, row) -> LineString:
    geometry = network.graph.edges[row["u"], row["v"]][GEOMETRY]
    if geometry.coords[0] != network.node_coords[row["flow_from"]]:
        return LineString(list(geometry.coords)[::-1])
    return geometry


def _totals(net_gdf, rows) -> PostCalculation:
    loss_kw = net_gdf[cols.HEAT_LOSS].sum() / HOURS_PER_YEAR
    source = net_gdf[net_gdf[cols.TYPE] == SOURCE_CONNECTION].iloc[0]
    above = sum(row["above_largest_dn"] for row in rows)
    if above:
        logger.warning("%d section(s) need more than the largest DN of the pipe catalogue.", above)
    return PostCalculation(
        producer_capacity=float(source[cols.THERMAL_POWER_GLF] + loss_kw),
        heat_loss=float(loss_kw),
        invest_cost=float(net_gdf[cols.INVEST_COST].sum()),
        invest_cost_model=float(net_gdf[cols.INVEST_COST_MODEL].sum()),
        annual_cost=float(net_gdf[cols.ANNUAL_COST].sum()),
        glf_max_deviation=float((net_gdf[cols.GLF_MODEL] / net_gdf[cols.GLF] - 1).abs().max()),
        capacity_max_deviation=float(
            (net_gdf[cols.CAPACITY_MODEL] / net_gdf[cols.THERMAL_POWER_GLF] - 1).abs().max()
        ),
        sections_above_largest_dn=int(above),
    )
