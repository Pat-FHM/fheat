"""oemof.solph energy system with the network block: coupling, objective, one solve.

    Source "erzeuger" ──▶ Bus "waermenetz" ──▶ Sink "verbraucher"
                                          └──▶ Sink "netzverluste"

One time step of 8760 h: flows are annual mean values [kW], variable costs
[€/kWh] are weighted with 8760 h. The coupling is block.py, (9) and (10); the
objective is the solph objective plus the pipe annuity (block.py, objective).

HiGHS stops only at the absolute MIP gap [€/a] or at the time limit. The
relative gap is set to 0: HiGHS stops at the first criterion met, and in the
forced mode the large constant heat cost would let a relative gap hide whole
route variants. HiGHS is called through the appsi interface directly: the
legacy ``SolverFactory("appsi_highs").solve()`` resets ``config.time_limit``
to its ``timelimit`` argument, which silently dropped the time limit.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from fheat_core import columns as cols
from fheat_core.config import OptimizationConfig
from fheat_core.optimization import MISSING_OPT_EXTRA
from fheat_core.optimization.block import build_network_block
from fheat_core.optimization.glf_terms import ReferenceTree, glf_factors, reference_tree
from fheat_core.optimization.linearize import PipeLinearization
from fheat_core.optimization.preprocess import SimplifiedNetwork, edge_key

try:
    import oemof.solph as solph
    from oemof.tools import economics
    import pyomo.environ as po
    from pyomo.contrib.appsi.base import TerminationCondition
    from pyomo.contrib.appsi.solvers import Highs
except ImportError as err:
    raise ImportError(MISSING_OPT_EXTRA) from err

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 8760
MIP_ABS_GAP_AUTO_SHARE = 0.005   # "auto": 0.5 % of the pipe annuity of the reference tree


@dataclass(frozen=True)
class ObjectiveParts:
    """Objective terms [€/a]."""

    source_invest: float      # annuity of the producer capacity
    heat_demand: float        # heat cost of the consumers' demand
    heat_losses: float        # heat cost of the network losses
    pipes: float              # annuity of the linearised pipe investment

    @property
    def total(self) -> float:
        return self.source_invest + self.heat_demand + self.heat_losses + self.pipes


@dataclass
class MilpResult:
    """Solution of one MILP run.

    ``edges`` has one row per section of the simplified graph: ``u``, ``v``,
    ``cols.TYPE``, ``cols.LENGTH`` [m], ``fixed`` (bridge built without a
    binary), ``built``, ``flow_from``, ``flow_to``, ``capacity`` [kW],
    ``cols.THERMAL_POWER`` (S [kW]), ``cols.N_BUILDINGS`` (n), ``glf_model``
    (g_e used in block.py, (7)), ``heat_loss`` [kW] and ``invest_cost`` [€]
    of the linearised model.
    """

    termination: str
    glf_mode: str
    objective: float            # [€/a]
    best_bound: float | None    # lower bound of the objective [€/a]
    objective_parts: ObjectiveParts
    mip_abs_gap: float          # [€/a], as passed to HiGHS
    source_capacity: float      # producer capacity [kW]
    producer_flow: float        # [kW]
    demand_flow: float          # [kW]
    loss_flow: float            # [kW]
    edges: pd.DataFrame
    build_time_s: float
    solve_time_s: float

    @property
    def achieved_gap(self) -> float | None:
        """Objective minus lower bound [€/a] at the end of the solve."""
        return None if self.best_bound is None else self.objective - self.best_bound


@dataclass
class _Components:
    bus: solph.Bus
    producer: solph.components.Source
    consumers: solph.components.Sink
    losses: solph.components.Sink


def solve_network(
    network: SimplifiedNetwork,
    linearization: PipeLinearization,
    heat_demand: Mapping,
    config: OptimizationConfig,
) -> MilpResult:
    """Build the energy system with the network block and solve it once.

    ``heat_demand`` maps every building key of ``network`` to W_k [kWh/a].
    ``linearization`` is built from ``glf_terms.design_loads`` (the same cost
    line for both glf modes).
    """
    t0 = time.perf_counter()
    tree = reference_tree(network)
    glf = glf_factors(network, tree, config.glf_mode)
    demand_flow = sum(heat_demand[k] for k in network.building_nodes) / HOURS_PER_YEAR
    comp = _components(config)
    es = solph.EnergySystem(
        timeindex=pd.date_range("2025-01-01", periods=2, freq=f"{HOURS_PER_YEAR}h"),
        infer_last_interval=False,
    )
    es.add(comp.bus, comp.producer, comp.consumers, comp.losses)
    model = solph.Model(es)
    model.netz = build_network_block(network, linearization, glf)
    _couple(model, comp, demand_flow)
    _add_pipe_annuity(model, _pipe_annuity(config))
    abs_gap = _mip_abs_gap(network, linearization, tree, glf, config)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = _run_highs(model, abs_gap, config.time_limit_s)
    solve_time = time.perf_counter() - t0
    return _result(model, comp, network, glf, results, config, abs_gap, (build_time, solve_time))


def _pipe_annuity(config) -> float:
    """Annuity factor of the pipes [1/a]."""
    return economics.annuity(1.0, config.lifetime_pipes, config.interest_rate)


def _components(config) -> _Components:
    bus = solph.Bus(label="waermenetz")
    source_ep_costs = economics.annuity(
        config.source_capex_eur_per_kw, config.lifetime_source, config.interest_rate
    )
    producer = solph.components.Source(
        label="erzeuger",
        outputs={bus: solph.Flow(
            nominal_capacity=solph.Investment(ep_costs=source_ep_costs),
            variable_costs=config.heat_cost_eur_per_kwh,
        )},
    )
    consumers = solph.components.Sink(label="verbraucher", inputs={bus: solph.Flow()})
    losses = solph.components.Sink(label="netzverluste", inputs={bus: solph.Flow()})
    return _Components(bus, producer, consumers, losses)


def _couple(model, comp, demand_flow):
    """block.py, (9) and (10)."""
    netz = model.netz
    invest = model.InvestmentFlowBlock.invest[comp.producer, comp.bus, 0]
    model.demand_coupling = po.Constraint(
        model.TIMESTEPS, rule=lambda m, t: m.flow[comp.bus, comp.consumers, t] == demand_flow
    )
    model.loss_coupling = po.Constraint(
        model.TIMESTEPS, rule=lambda m, t: m.flow[comp.bus, comp.losses, t] == netz.heat_loss
    )
    model.capacity_coupling = po.Constraint(expr=invest >= netz.source_capacity + netz.heat_loss)


def _add_pipe_annuity(model, pipe_annuity):
    solph_objective = model.objective.expr
    model.del_component("objective")
    model.objective = po.Objective(
        expr=solph_objective + pipe_annuity * model.netz.invest_cost, sense=po.minimize
    )


def _mip_abs_gap(network, linearization, tree: ReferenceTree, glf, config) -> float:
    """Absolute MIP gap [€/a].

    "auto": ``MIP_ABS_GAP_AUTO_SHARE`` of the pipe annuity of the reference
    tree, each tree section sized with C = g_e · S0 as in block.py, (7).
    """
    if config.mip_abs_gap != "auto":
        return float(config.mip_abs_gap)
    H = network.graph
    invest = sum(
        linearization.invest_cost(H.edges[e][cols.TYPE], H.edges[e][cols.LENGTH], glf[e] * s0, 1)
        for e, s0 in tree.s0.items()
    )
    return MIP_ABS_GAP_AUTO_SHARE * _pipe_annuity(config) * invest


def _run_highs(model, abs_gap, time_limit_s):
    """One HiGHS run; stops at ``abs_gap`` [€/a] or ``time_limit_s`` [s] only."""
    opt = Highs()
    opt.config.load_solution = False
    opt.config.time_limit = time_limit_s
    opt.highs_options = {"mip_rel_gap": 0.0, "mip_abs_gap": abs_gap}
    results = opt.solve(model)
    if results.best_feasible_objective is None:
        raise RuntimeError(f"HiGHS found no feasible network (termination: {results.termination_condition}).")
    results.solution_loader.load_vars()
    if results.termination_condition != TerminationCondition.optimal:
        logger.warning(
            "HiGHS stopped with '%s'; the network is feasible but not within the MIP gap.",
            results.termination_condition,
        )
    return results


def _result(model, comp, network, glf, results, config, abs_gap, run_times) -> MilpResult:
    netz = model.netz
    heat_cost = config.heat_cost_eur_per_kwh * HOURS_PER_YEAR
    invest = model.InvestmentFlowBlock.invest[comp.producer, comp.bus, 0]
    ep_costs = comp.producer.outputs[comp.bus].investment.ep_costs[0]
    demand_flow = po.value(model.flow[comp.bus, comp.consumers, 0])
    loss_flow = po.value(model.flow[comp.bus, comp.losses, 0])
    parts = ObjectiveParts(
        source_invest=ep_costs * po.value(invest),
        heat_demand=heat_cost * demand_flow,
        heat_losses=heat_cost * loss_flow,
        pipes=_pipe_annuity(config) * po.value(netz.invest_cost),
    )
    return MilpResult(
        termination=str(results.termination_condition.name),
        glf_mode=config.glf_mode,
        objective=po.value(model.objective),
        best_bound=results.best_objective_bound,
        objective_parts=parts,
        mip_abs_gap=abs_gap,
        source_capacity=po.value(invest),
        producer_flow=po.value(model.flow[comp.producer, comp.bus, 0]),
        demand_flow=demand_flow,
        loss_flow=loss_flow,
        edges=_edge_table(netz, network, glf),
        build_time_s=run_times[0],
        solve_time_s=run_times[1],
    )


def _edge_table(netz, network, glf) -> pd.DataFrame:
    built_arc = {edge_key(*a): a for a in netz.ARCS if po.value(netz.direction[a]) > 0.5}
    rows = []
    for u, v, data in network.graph.edges(data=True):
        e = edge_key(u, v)
        arc = built_arc.get(e)
        rows.append({
            "u": e[0], "v": e[1],
            cols.TYPE: data[cols.TYPE], cols.LENGTH: data[cols.LENGTH],
            "fixed": netz.built[e].fixed,
            "built": po.value(netz.built[e]) > 0.5,
            "flow_from": arc[0] if arc else None,
            "flow_to": arc[1] if arc else None,
            "capacity": po.value(netz.capacity[e]),
            cols.THERMAL_POWER: po.value(netz.power_flow[arc]) if arc else 0.0,
            cols.N_BUILDINGS: po.value(netz.count_flow[arc]) if arc else 0.0,
            "glf_model": glf[e],
            "heat_loss": po.value(netz.section_loss[e]),
            "invest_cost": po.value(netz.section_invest[e]),
        })
    return pd.DataFrame(rows)
