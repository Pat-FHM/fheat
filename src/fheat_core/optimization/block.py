"""Pyomo block "netz": route, flow direction and capacity of every section (plan section 6).

Forced mode: every reachable building is connected (x_k = 1). Symbols and
constraint numbers as in plan section 6:

1. λ_ij + λ_ji = y_e: one pipe with one flow direction per section (R3)
2. Σ_i λ_ij ≤ 1 at every junction j: radial network
3. λ_(j→k) = 1 at every house connection
4. count flow n: inflow − outflow = 1 at a building, 0 at a junction
5. power flow S [kW]: inflow − outflow = Q_k at a building, 0 at a junction
6. n_ij ≤ N · λ_ij and S_ij ≤ ΣQ · λ_ij: flow only in the built direction.
   In addition λ_ij ≤ n_ij: every built pipe supplies at least one building.
   This keeps the optimum and makes every feasible solution a tree, also one
   the solver returns at the MIP gap or the time limit.
7. C_e ≥ S_e (design capacity without simultaneity) and C_e ≤ Q_max · y_e [kW]
8. heat loss_e = (a_V · C_e + b_V · y_e) · L_e / 1000 [kW]

Bridges with a fixed direction (plan section 5) must be built in the forced
mode: only their fixed direction exists and y, λ are fixed to 1, so they need
no binary decision. House connections are such bridges, which gives no. 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fheat_core import columns as cols
from fheat_core.optimization import MISSING_OPT_EXTRA
from fheat_core.optimization.linearize import PipeLinearization
from fheat_core.optimization.preprocess import (
    FLOW_FROM,
    FLOW_TO,
    KIND,
    KIND_BUILDING,
    KIND_JUNCTION,
    POWER,
    POWER_BEHIND,
    SimplifiedNetwork,
    edge_key,
)

try:
    import pyomo.environ as po
except ImportError as err:
    raise ImportError(MISSING_OPT_EXTRA) from err


@dataclass
class NetworkSets:
    """Index sets and parameters of the block, taken from the simplified graph.

    ``edges`` are :func:`edge_key` tuples, ``arcs`` directed (from, to) pairs.
    Lengths in m, powers in kW.
    """

    edges: list[tuple[int, int]] = field(default_factory=list)
    arcs: list[tuple[int, int]] = field(default_factory=list)
    arcs_of_edge: dict = field(default_factory=dict)
    arcs_in: dict = field(default_factory=dict)
    arcs_out: dict = field(default_factory=dict)
    fixed_edges: set = field(default_factory=set)
    edge_type: dict = field(default_factory=dict)
    length: dict = field(default_factory=dict)
    building_power: dict = field(default_factory=dict)
    junctions: list[int] = field(default_factory=list)
    source_edge: tuple[int, int] | None = None
    capacity_max: float = 0.0

    @property
    def n_buildings(self) -> int:
        return len(self.building_power)

    @property
    def total_power(self) -> float:
        return sum(self.building_power.values())

    @classmethod
    def of(cls, network: SimplifiedNetwork, capacity_max: float) -> NetworkSets:
        H = network.graph
        source = network.source_node()
        sets = cls(capacity_max=capacity_max)
        for n, data in H.nodes(data=True):
            sets.arcs_in[n], sets.arcs_out[n] = [], []
            if data[KIND] == KIND_BUILDING:
                sets.building_power[n] = data[POWER]
            elif data[KIND] == KIND_JUNCTION:
                sets.junctions.append(n)
        for u, v, data in H.edges(data=True):
            sets._add_edge(u, v, data, source)
        return sets

    def _add_edge(self, u, v, data, source):
        e = edge_key(u, v)
        self.edges.append(e)
        self.edge_type[e] = data[cols.TYPE]
        self.length[e] = data[cols.LENGTH]
        if source in e:
            self.source_edge = e
        if data[FLOW_FROM] is None:
            arcs = [(u, v), (v, u)]
        else:
            arcs = [(data[FLOW_FROM], data[FLOW_TO])]
            self.fixed_edges.add(e)
            self._check_capacity(e, data[POWER_BEHIND])
        self.arcs_of_edge[e] = arcs
        for i, j in arcs:
            self.arcs.append((i, j))
            self.arcs_out[i].append((i, j))
            self.arcs_in[j].append((i, j))

    def _check_capacity(self, e, power):
        if power > self.capacity_max:
            raise ValueError(
                f"Section {e} must carry {power:.1f} kW, more than the largest DN "
                f"({self.capacity_max:.1f} kW)."
            )


def build_network_block(network: SimplifiedNetwork, linearization: PipeLinearization) -> po.Block:
    """Pyomo block with variables, constraints 1 to 8 and the cost expressions.

    Expressions for the energy system: ``heat_loss`` [kW] (sum of no. 8),
    ``invest_cost`` [€] (pipe investment, linearised) and
    ``source_capacity`` [kW] (C of the source connection).
    """
    sets = NetworkSets.of(network, float(linearization.capacities["capacity"].max()))
    b = po.Block(concrete=True)
    b.EDGES = po.Set(initialize=sets.edges, dimen=2, ordered=True)
    b.ARCS = po.Set(initialize=sets.arcs, dimen=2, ordered=True)
    _add_variables(b, sets)
    _add_direction_constraints(b, sets)
    _add_flow_constraints(b, sets)
    _add_capacity_constraints(b, sets)
    _add_expressions(b, sets, linearization)
    return b


def _add_variables(b, sets):
    b.built = po.Var(b.EDGES, domain=po.Binary)                   # y_e
    b.direction = po.Var(b.ARCS, domain=po.Binary)                # λ_ij
    b.count_flow = po.Var(b.ARCS, domain=po.NonNegativeReals)     # n_ij
    b.power_flow = po.Var(b.ARCS, domain=po.NonNegativeReals)     # S_ij [kW]
    b.capacity = po.Var(b.EDGES, domain=po.NonNegativeReals)      # C_e [kW]
    for e in sets.fixed_edges:
        b.built[e].fix(1)
        for a in sets.arcs_of_edge[e]:
            b.direction[a].fix(1)


def _add_direction_constraints(b, sets):
    """Constraints 1 and 2."""
    def one_direction(b, u, v):
        if (u, v) in sets.fixed_edges:
            return po.Constraint.Skip
        return sum(b.direction[a] for a in sets.arcs_of_edge[u, v]) == b.built[u, v]

    def radial(b, j):
        if not sets.arcs_in[j]:
            return po.Constraint.Skip
        return sum(b.direction[a] for a in sets.arcs_in[j]) <= 1

    b.one_direction = po.Constraint(b.EDGES, rule=one_direction)
    b.radial = po.Constraint(sets.junctions, rule=radial)


def _add_flow_constraints(b, sets):
    """Constraints 4 to 6."""
    nodes = sets.junctions + list(sets.building_power)

    def count_balance(b, j):
        inflow = sum(b.count_flow[a] for a in sets.arcs_in[j])
        outflow = sum(b.count_flow[a] for a in sets.arcs_out[j])
        return inflow - outflow == (1 if j in sets.building_power else 0)

    def power_balance(b, j):
        inflow = sum(b.power_flow[a] for a in sets.arcs_in[j])
        outflow = sum(b.power_flow[a] for a in sets.arcs_out[j])
        return inflow - outflow == sets.building_power.get(j, 0.0)

    b.count_balance = po.Constraint(nodes, rule=count_balance)
    b.power_balance = po.Constraint(nodes, rule=power_balance)
    b.count_bound = po.Constraint(
        b.ARCS, rule=lambda b, i, j: b.count_flow[i, j] <= sets.n_buildings * b.direction[i, j]
    )
    b.power_bound = po.Constraint(
        b.ARCS, rule=lambda b, i, j: b.power_flow[i, j] <= sets.total_power * b.direction[i, j]
    )
    b.supplies_building = po.Constraint(
        b.ARCS, rule=lambda b, i, j: b.direction[i, j] <= b.count_flow[i, j]
    )


def _add_capacity_constraints(b, sets):
    """Constraint 7 (without simultaneity)."""
    def design(b, u, v):
        return b.capacity[u, v] >= sum(b.power_flow[a] for a in sets.arcs_of_edge[u, v])

    b.capacity_design = po.Constraint(b.EDGES, rule=design)
    b.capacity_bound = po.Constraint(
        b.EDGES, rule=lambda b, u, v: b.capacity[u, v] <= sets.capacity_max * b.built[u, v]
    )


def _add_expressions(b, sets, linearization):
    """Constraint 8 and the linearised pipe investment."""
    def section_loss(b, u, v):
        e = (u, v)
        return linearization.heat_loss(sets.edge_type[e], sets.length[e], b.capacity[e], b.built[e])

    def section_invest(b, u, v):
        e = (u, v)
        return linearization.invest_cost(sets.edge_type[e], sets.length[e], b.capacity[e], b.built[e])

    b.section_loss = po.Expression(b.EDGES, rule=section_loss)            # [kW]
    b.section_invest = po.Expression(b.EDGES, rule=section_invest)        # [€]
    b.heat_loss = po.Expression(expr=sum(b.section_loss[e] for e in b.EDGES))
    b.invest_cost = po.Expression(expr=sum(b.section_invest[e] for e in b.EDGES))
    b.source_capacity = po.Expression(expr=b.capacity[sets.source_edge])
