"""Linearisation of pipe costs and heat losses over the transport capacity.

The MILP only knows a continuous design capacity C_e [kW] per section. Pipe
costs and heat losses are therefore approximated per edge type by straight
lines over the capacity of the catalogue DNs (block.py, objective and (8)):

    cost [€/m] = a_K · Q + b_K        loss [W/m] = a_V · Q + b_V

with Q_max = V̇_max · ρ · c_p · (T_VL − T_RL) per DN and the loss
2 · U · (T_mean − T_soil) per trench metre, U the standard ``U-Value`` of the
catalogue (as ``cols.HEAT_LOSS`` in ``compute_network``).

The regression range is limited automatically to the DNs the network can
actually need (smallest allowed DN up to the DN carrying the design load,
plus one step), which gives a tighter fit (cf. Lambert et al. 2024). The fit
quality (R², deviation per DN) is logged and returned with the result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fheat_core.algorithms.network import calculate_volumeflow
from fheat_core.optimization import HOUSE_CONNECTION, STREET_PIPE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesignLoads:
    """Largest design load [kW] of a house connection and of a street pipe.

    They set the end of the regression range of each edge type.
    """

    house: float
    street: float

    def of(self, edge_type: str) -> float:
        return self.house if edge_type == HOUSE_CONNECTION else self.street


def _start_index(edge_type: str) -> int:
    """First admissible catalogue row, same rule as ``calculate_diameter_velocity_loss``."""
    return 0 if edge_type == HOUSE_CONNECTION else 2


@dataclass(frozen=True)
class LinearFit:
    """Straight line ``y = slope · Q + intercept`` fitted over catalogue DNs.

    ``table`` holds one row per DN in the regression range with the columns
    DN, capacity [kW], actual, fitted and deviation (``(fitted - actual) / actual``).
    """

    slope: float
    intercept: float
    r_squared: float
    table: pd.DataFrame

    @property
    def max_abs_deviation(self) -> float:
        return float(self.table["deviation"].abs().max())

    @property
    def dn_range(self) -> tuple[str, str]:
        return str(self.table["DN"].iloc[0]), str(self.table["DN"].iloc[-1])


@dataclass(frozen=True)
class PipeLinearization:
    """Linearised cost and loss coefficients for house connections and street pipes.

    Units: cost slope €/(m·kW), cost intercept €/m, loss slope W/(m·kW),
    loss intercept W/m. ``capacities`` lists Q_max [kW] for every catalogue DN.
    ``warnings`` repeats every warning that was logged during the fit.
    """

    house_cost: LinearFit
    street_cost: LinearFit
    house_loss: LinearFit
    street_loss: LinearFit
    capacities: pd.DataFrame
    supply_temperature: float
    return_temperature: float
    soil_temperature: float
    max_deviation: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def fit(self, quantity: str, edge_type: str) -> LinearFit:
        """Return the fit for ``quantity`` in {"cost", "loss"} and an edge type.

        Every edge type other than a house connection uses the street pipe fit
        (same rule as ``calculate_diameter_velocity_loss``).
        """
        prefix = "house" if edge_type == HOUSE_CONNECTION else "street"
        return getattr(self, f"{prefix}_{quantity}")

    def invest_cost(self, edge_type: str, length, capacity, built):
        """Pipe investment [€] of a section: L · (a_K · C + b_K · y) (block.py, objective).

        ``length`` [m], ``capacity`` C [kW], ``built`` y (0/1). Accepts numbers
        and Pyomo expressions.
        """
        f = self.fit("cost", edge_type)
        return length * (f.slope * capacity + f.intercept * built)

    def heat_loss(self, edge_type: str, length, capacity, built):
        """Heat loss [kW] of a section: (a_V · C + b_V · y) · L / 1000 (block.py, (8)).

        ``length`` [m], ``capacity`` C [kW], ``built`` y (0/1). Accepts numbers
        and Pyomo expressions.
        """
        f = self.fit("loss", edge_type)
        return (f.slope * capacity + f.intercept * built) * length / 1000

    def report(self) -> pd.DataFrame:
        """Fit quality as one long table (one row per fit and DN)."""
        frames = []
        for quantity in ("cost", "loss"):
            for edge_type in (HOUSE_CONNECTION, STREET_PIPE):
                f = self.fit(quantity, edge_type)
                t = f.table.copy()
                t.insert(0, "edge_type", edge_type)
                t.insert(0, "quantity", quantity)
                t["slope"] = f.slope
                t["intercept"] = f.intercept
                t["r_squared"] = f.r_squared
                frames.append(t)
        return pd.concat(frames, ignore_index=True)


def trench_loss(u_value, htemp: float, ltemp: float, soil_temperature: float):
    """Heat loss [W/m] of supply and return pipe: 2 · U · (T_mean − T_soil).

    ``u_value`` [W/(m·K)] per pipe (number or Series). With T_soil = 10 °C this
    is the loss of ``calculate_diameter_velocity_loss`` per hour and metre.
    """
    return 2 * u_value * ((htemp + ltemp) / 2 - soil_temperature)


def pipe_capacities(pipe_info: pd.DataFrame, htemp: float, ltemp: float) -> pd.Series:
    """Maximum thermal capacity Q_max [kW] per catalogue DN.

    Q_max = V̇_max · ρ · c_p · (T_VL − T_RL), with ρ and c_p exactly as in
    :func:`calculate_volumeflow` (which is linear in the power).
    """
    return pipe_info["max_volumeFlow"] / calculate_volumeflow(1.0, htemp, ltemp)


def fit_linear(dn, capacity, values) -> LinearFit:
    """Ordinary least-squares line through (capacity, value) points."""
    q = np.asarray(capacity, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(q) < 2:
        raise ValueError("A linear fit needs at least two catalogue DNs.")
    slope, intercept = np.polyfit(q, y, 1)
    fitted = slope * q + intercept
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 1.0 if ss_res == 0 else float("nan")
    table = pd.DataFrame({
        "DN": list(dn),
        "capacity": q,
        "actual": y,
        "fitted": fitted,
        "deviation": (fitted - y) / y,
    })
    return LinearFit(float(slope), float(intercept), float(r_squared), table)


def regression_range(
    pipe_info: pd.DataFrame,
    edge_type: str,
    design_power: float,
    htemp: float,
    ltemp: float,
) -> tuple[int, int, bool]:
    """Catalogue rows ``[first, last]`` used for the regression of one edge type.

    ``first`` is the smallest admissible DN, ``last`` the DN that carries
    ``design_power`` [kW] (same sizing rule as ``calculate_diameter_velocity_loss``)
    plus one step, capped at the end of the catalogue. The third value is True
    if ``design_power`` exceeds the largest DN.
    """
    start = _start_index(edge_type)
    if start >= len(pipe_info) - 1:
        raise ValueError(
            f"Pipe catalogue has too few DNs for a regression of '{edge_type}' "
            f"(needs at least two rows from index {start})."
        )
    vf = calculate_volumeflow(design_power, htemp, ltemp)
    idx = int(pipe_info["max_volumeFlow"][start:].searchsorted(vf, side="right")) + start
    exceeded = idx >= len(pipe_info)
    return start, min(idx + 1, len(pipe_info) - 1), exceeded


def merge_pipe_costs(pipe_info: pd.DataFrame, pipe_costs: pd.DataFrame) -> pd.DataFrame:
    """Attach ``cost_eur_per_m`` to the catalogue by ``DN``; every DN needs a cost."""
    costs = pipe_costs.set_index("DN")["cost_eur_per_m"]
    if costs.index.duplicated().any():
        dup = sorted(costs.index[costs.index.duplicated()].astype(str))
        raise ValueError(f"pipe_costs contains duplicate DN entries: {dup}")
    missing = [dn for dn in pipe_info["DN"] if dn not in costs.index]
    if missing:
        raise ValueError(f"pipe_costs has no cost for DN {missing}.")
    out = pipe_info.copy()
    out["cost_eur_per_m"] = out["DN"].map(costs).astype(float)
    return out


def linearize_pipes(
    pipe_info: pd.DataFrame,
    pipe_costs: pd.DataFrame,
    design_loads: DesignLoads,
    htemp: float,
    ltemp: float,
    soil_temperature: float = 10.0,
    max_deviation: float = 0.15,
) -> PipeLinearization:
    """Linearise pipe costs and heat losses for house connections and street pipes.

    Parameters
    ----------
    pipe_info
        Pipe catalogue (``load_pipe_info()`` or adapter), sorted by capacity.
    pipe_costs
        Cost table with ``DN`` and ``cost_eur_per_m`` [€ per trench metre].
    design_loads
        Largest design load [kW] per edge type, see ``glf_terms.design_loads``.
    htemp, ltemp
        Supply and return temperature [°C].
    soil_temperature
        Soil temperature [°C] for the loss 2 · U · (T_mean − T_soil) [W/m].
    max_deviation
        A warning is issued for every fit whose relative deviation at one DN
        exceeds this value.
    """
    if min(design_loads.house, design_loads.street) < 0:
        raise ValueError(f"Design loads must not be negative: {design_loads}.")
    catalogue = _catalogue(pipe_info, pipe_costs, htemp, ltemp, soil_temperature)
    fits: dict[tuple[str, str], LinearFit] = {}
    warnings: list[str] = []
    for edge_type in (HOUSE_CONNECTION, STREET_PIPE):
        rows = _regression_rows(catalogue, edge_type, design_loads.of(edge_type), htemp, ltemp, warnings)
        for quantity, column in (("cost", "cost_eur_per_m"), ("loss", "loss_w_per_m")):
            f = fit_linear(rows["DN"], rows["capacity"], rows[column])
            fits[(quantity, edge_type)] = f
            _log_fit(f, quantity, edge_type, max_deviation, warnings)

    return PipeLinearization(
        house_cost=fits[("cost", HOUSE_CONNECTION)],
        street_cost=fits[("cost", STREET_PIPE)],
        house_loss=fits[("loss", HOUSE_CONNECTION)],
        street_loss=fits[("loss", STREET_PIPE)],
        capacities=catalogue[["DN", "capacity"]].copy(),
        supply_temperature=htemp,
        return_temperature=ltemp,
        soil_temperature=soil_temperature,
        max_deviation=max_deviation,
        warnings=tuple(warnings),
    )


def _catalogue(pipe_info, pipe_costs, htemp, ltemp, soil_temperature) -> pd.DataFrame:
    """Catalogue with cost [€/m], capacity Q_max [kW] and loss [W/m] per DN."""
    catalogue = merge_pipe_costs(pipe_info, pipe_costs).reset_index(drop=True)
    catalogue["capacity"] = pipe_capacities(catalogue, htemp, ltemp)
    catalogue["loss_w_per_m"] = trench_loss(catalogue["U-Value"], htemp, ltemp, soil_temperature)
    return catalogue


def _regression_rows(catalogue, edge_type, design_power, htemp, ltemp, warnings) -> pd.DataFrame:
    first, last, exceeded = regression_range(catalogue, edge_type, design_power, htemp, ltemp)
    if exceeded:
        msg = (
            f"{edge_type}: design load {design_power:.1f} kW exceeds the largest DN "
            f"{catalogue['DN'].iloc[-1]} ({catalogue['capacity'].iloc[-1]:.1f} kW)."
        )
        logger.warning(msg)
        warnings.append(msg)
    return catalogue.iloc[first:last + 1]


def _log_fit(f, quantity, edge_type, max_deviation, warnings):
    """Log R² and the deviation per DN; warn above ``max_deviation``."""
    lo, hi = f.dn_range
    logger.info(
        "Linearised %s (%s, %s to %s): slope=%.6g, intercept=%.6g, R²=%.3f, "
        "deviation per DN: %s",
        quantity, edge_type, lo, hi, f.slope, f.intercept, f.r_squared,
        ", ".join(f"{d}: {v:+.1%}" for d, v in zip(f.table["DN"], f.table["deviation"], strict=True)),
    )
    if f.max_abs_deviation > max_deviation:
        worst = f.table.loc[f.table["deviation"].abs().idxmax()]
        msg = (
            f"{quantity} regression ({edge_type}, {lo} to {hi}) deviates by "
            f"{worst['deviation']:+.1%} at {worst['DN']} "
            f"(limit ±{max_deviation:.0%})."
        )
        logger.warning(msg)
        warnings.append(msg)
