"""Tests for fheat_core.optimization.linearize (T1).

Covers:
- pipe_capacities: Q_max per DN consistent with calculate_volumeflow
- fit_linear: least-squares line, R², deviation per DN
- regression_range: automatic DN range (smallest DN to design DN plus one step)
- merge_pipe_costs: every catalogue DN needs a cost
- linearize_pipes: reference values, monotone costs, warnings, report
- invest_cost / heat_loss: the linearised terms of the MILP

The reference values (70/50 °C, street DN32 to KMR 150: a_K ≈ 0.123 €/(m·kW),
b_K ≈ 640 €/m, R² ≈ 0.84, deviation −10 % to +13 %) are test references only.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from fheat_core.algorithms.network import calculate_glf, calculate_volumeflow
from fheat_core.optimization import HOUSE_CONNECTION, STREET_PIPE
from fheat_core.optimization.linearize import (
    fit_linear,
    DesignLoads,
    linearize_pipes,
    merge_pipe_costs,
    pipe_capacities,
    regression_range,
)
from fheat_core.resources import load_pipe_costs, load_pipe_info


@pytest.fixture
def pipe_info():
    return load_pipe_info()


@pytest.fixture
def pipe_costs():
    return load_pipe_costs()


@pytest.fixture
def reference(pipe_info, pipe_costs):
    """50 buildings à 50 kW at 70/50 °C → street range PEX 32 to KMR 150."""
    return linearize_pipes(pipe_info, pipe_costs, _loads([50.0] * 50), htemp=70, ltemp=50)


def _loads(powers):
    """Design loads with simultaneity: GLF(1) · max Q and GLF(N) · ΣQ."""
    return DesignLoads(house=calculate_glf(1) * max(powers), street=calculate_glf(len(powers)) * sum(powers))


# ---------------------------------------------------------------------------
# pipe_capacities
# ---------------------------------------------------------------------------


class TestPipeCapacities:
    def test_round_trip_with_volumeflow(self, pipe_info):
        q = pipe_capacities(pipe_info, 70, 50)
        for q_max, vf_max in zip(q, pipe_info["max_volumeFlow"], strict=True):
            assert calculate_volumeflow(q_max, 70, 50) == pytest.approx(vf_max)

    def test_monotonically_increasing(self, pipe_info):
        assert pipe_capacities(pipe_info, 70, 50).is_monotonic_increasing

    def test_scales_with_temperature_spread(self, pipe_info):
        q_20 = pipe_capacities(pipe_info, 70, 50)
        q_30 = pipe_capacities(pipe_info, 70, 40)
        np.testing.assert_allclose(q_30 / q_20, 1.5)


# ---------------------------------------------------------------------------
# fit_linear
# ---------------------------------------------------------------------------


class TestFitLinear:
    def test_exact_line(self):
        f = fit_linear(["a", "b", "c"], [1.0, 2.0, 3.0], [12.0, 14.0, 16.0])
        assert f.slope == pytest.approx(2.0)
        assert f.intercept == pytest.approx(10.0)
        assert f.r_squared == pytest.approx(1.0)
        assert f.max_abs_deviation == pytest.approx(0.0, abs=1e-12)

    def test_deviation_relative_to_actual(self):
        f = fit_linear(["a", "b", "c"], [0.0, 1.0, 2.0], [10.0, 13.0, 10.0])
        # fitted = 11 everywhere
        np.testing.assert_allclose(f.table["deviation"], [0.1, 11 / 13 - 1, 0.1])

    def test_needs_two_points(self):
        with pytest.raises(ValueError, match="at least two"):
            fit_linear(["a"], [1.0], [1.0])

    def test_dn_range(self):
        f = fit_linear(["PEX 20", "PEX 25"], [1.0, 2.0], [1.0, 2.0])
        assert f.dn_range == ("PEX 20", "PEX 25")


# ---------------------------------------------------------------------------
# regression_range
# ---------------------------------------------------------------------------


class TestRegressionRange:
    def test_street_starts_at_index_2(self, pipe_info):
        first, _, _ = regression_range(pipe_info, STREET_PIPE, 10.0, 70, 50)
        assert first == 2

    def test_house_connection_starts_at_index_0(self, pipe_info):
        first, _, _ = regression_range(pipe_info, HOUSE_CONNECTION, 10.0, 70, 50)
        assert first == 0

    def test_design_dn_plus_one_step(self, pipe_info):
        """Upper end is the DN chosen by the sizing rule, plus one step."""
        from fheat_core.algorithms.network import calculate_diameter_velocity_loss

        power = 1500.0
        vf = calculate_volumeflow(power, 70, 50)
        dn, *_ = calculate_diameter_velocity_loss(vf, 70, 50, 1.0, pipe_info, STREET_PIPE)
        design_idx = int(pipe_info.index[pipe_info["DN"] == dn][0])
        _, last, exceeded = regression_range(pipe_info, STREET_PIPE, power, 70, 50)
        assert last == design_idx + 1
        assert not exceeded

    def test_small_load_still_two_points(self, pipe_info):
        first, last, _ = regression_range(pipe_info, STREET_PIPE, 0.0, 70, 50)
        assert last - first == 1

    def test_capped_at_catalogue_end(self, pipe_info):
        _, last, exceeded = regression_range(pipe_info, STREET_PIPE, 1e9, 70, 50)
        assert last == len(pipe_info) - 1
        assert exceeded

    def test_too_small_catalogue_raises(self, pipe_info):
        with pytest.raises(ValueError, match="too few DNs"):
            regression_range(pipe_info.iloc[:3], STREET_PIPE, 10.0, 70, 50)


# ---------------------------------------------------------------------------
# merge_pipe_costs
# ---------------------------------------------------------------------------


class TestMergePipeCosts:
    def test_all_dn_have_costs(self, pipe_info, pipe_costs):
        merged = merge_pipe_costs(pipe_info, pipe_costs)
        assert merged["cost_eur_per_m"].notna().all()
        assert len(merged) == len(pipe_info)

    def test_missing_dn_raises(self, pipe_info, pipe_costs):
        with pytest.raises(ValueError, match="no cost for DN"):
            merge_pipe_costs(pipe_info, pipe_costs[pipe_costs["DN"] != "KMR 200"])

    def test_duplicate_dn_raises(self, pipe_info, pipe_costs):
        with pytest.raises(ValueError, match="duplicate"):
            merge_pipe_costs(pipe_info, pd.concat([pipe_costs, pipe_costs.iloc[:1]]))


# ---------------------------------------------------------------------------
# linearize_pipes
# ---------------------------------------------------------------------------


class TestLinearizePipes:
    def test_reference_street_range(self, reference):
        assert reference.street_cost.dn_range == ("PEX 32", "KMR 150")

    def test_reference_street_cost(self, reference):
        f = reference.street_cost
        assert f.slope == pytest.approx(0.123, rel=0.01)
        assert f.intercept == pytest.approx(640, abs=1.0)
        assert f.r_squared == pytest.approx(0.84, abs=0.005)
        assert f.table["deviation"].min() == pytest.approx(-0.10, abs=0.005)
        assert f.table["deviation"].max() == pytest.approx(0.13, abs=0.005)

    def test_house_range_follows_largest_building(self, pipe_info, reference):
        """GLF(1) · 50 kW is carried by PEX 32, plus one step → PEX 40."""
        assert reference.house_cost.dn_range == ("PEX 20", "PEX 40")

    @pytest.mark.parametrize("attr", ["house_cost", "street_cost"])
    def test_costs_increase_with_capacity(self, reference, attr):
        f = getattr(reference, attr)
        assert f.slope > 0
        assert f.intercept > 0

    def test_bundled_costs_increase_with_capacity(self, pipe_info, pipe_costs):
        merged = merge_pipe_costs(pipe_info, pipe_costs)
        assert merged["cost_eur_per_m"].is_monotonic_increasing

    def test_loss_values_match_fheat_formula(self, pipe_info, reference):
        """Actual loss per DN is 2 · U · (T_mean − T_soil) [W/m], as in F|Heat."""
        t = reference.street_loss.table.set_index("DN")
        u = pipe_info.set_index("DN")["U-Value"]
        for dn, actual in t["actual"].items():
            assert actual == pytest.approx(2 * u[dn] * (60 - 10))

    def test_loss_consistent_with_calculate_diameter_velocity_loss(self, pipe_info, reference):
        """W/m · 8760 h / 1000 equals the kWh/(a·m) of the Dijkstra sizing."""
        from fheat_core.algorithms.network import calculate_diameter_velocity_loss

        t = reference.street_loss.table.set_index("DN")
        vf = pipe_info["max_volumeFlow"].iloc[4] * 0.99  # sized into row 4
        dn, _, loss_kwh, _ = calculate_diameter_velocity_loss(vf, 70, 50, 1.0, pipe_info, STREET_PIPE)
        assert t.loc[dn, "actual"] * 8760 / 1000 == pytest.approx(loss_kwh)

    def test_soil_temperature_changes_losses(self, pipe_info, pipe_costs):
        warm = linearize_pipes(pipe_info, pipe_costs, _loads([50.0] * 50), 70, 50, soil_temperature=15.0)
        cold = linearize_pipes(pipe_info, pipe_costs, _loads([50.0] * 50), 70, 50, soil_temperature=5.0)
        assert warm.street_loss.intercept < cold.street_loss.intercept
        assert warm.soil_temperature == 15.0

    def test_negative_design_load_raises(self, pipe_info, pipe_costs):
        with pytest.raises(ValueError, match="negative"):
            linearize_pipes(pipe_info, pipe_costs, DesignLoads(house=-1.0, street=100.0), 70, 50)

    def test_tighter_range_fits_better(self, pipe_info, pipe_costs, reference):
        """The automatic range beats a regression over the whole street catalogue."""
        merged = merge_pipe_costs(pipe_info, pipe_costs).iloc[2:]
        full = fit_linear(merged["DN"], pipe_capacities(merged, 70, 50), merged["cost_eur_per_m"])
        assert reference.street_cost.max_abs_deviation < full.max_abs_deviation

    def test_range_ends_follow_design_loads(self, pipe_info, pipe_costs):
        loads = DesignLoads(house=40.0, street=1500.0)
        lin = linearize_pipes(pipe_info, pipe_costs, loads, 70, 50)
        for edge_type, fit in ((HOUSE_CONNECTION, lin.house_cost), (STREET_PIPE, lin.street_cost)):
            _, last, _ = regression_range(pipe_info, edge_type, loads.of(edge_type), 70, 50)
            assert fit.dn_range[1] == pipe_info["DN"].iloc[last]


class TestLinearizeWarnings:
    def test_deviation_warning_logged_and_returned(self, pipe_info, pipe_costs, caplog):
        with caplog.at_level(logging.WARNING, logger="fheat_core.optimization.linearize"):
            lin = linearize_pipes(pipe_info, pipe_costs, _loads([50.0] * 50), 70, 50, max_deviation=0.05)
        cost_warnings = [w for w in lin.warnings if w.startswith("cost regression (Straßenleitung")]
        assert len(cost_warnings) == 1
        assert "+13.0%" in cost_warnings[0]
        assert any(r.getMessage() == cost_warnings[0] for r in caplog.records)

    def test_no_cost_warning_within_limit(self, reference):
        assert not [w for w in reference.warnings if w.startswith("cost regression")]

    def test_exceeding_largest_dn_warns(self, pipe_info, pipe_costs):
        lin = linearize_pipes(pipe_info, pipe_costs, _loads([1e6]), 70, 50)
        assert any("exceeds the largest DN" in w for w in lin.warnings)
        assert lin.street_cost.dn_range[1] == pipe_info["DN"].iloc[-1]

    def test_fit_quality_logged(self, pipe_info, pipe_costs, caplog):
        with caplog.at_level(logging.INFO, logger="fheat_core.optimization.linearize"):
            linearize_pipes(pipe_info, pipe_costs, _loads([50.0] * 50), 70, 50)
        msgs = [r.getMessage() for r in caplog.records]
        assert sum("R²=" in m and "deviation per DN" in m for m in msgs) == 4


class TestReport:
    def test_report_contains_all_fits(self, reference):
        rep = reference.report()
        assert set(zip(rep["quantity"], rep["edge_type"], strict=True)) == {
            ("cost", HOUSE_CONNECTION), ("cost", STREET_PIPE),
            ("loss", HOUSE_CONNECTION), ("loss", STREET_PIPE),
        }
        for col in ("DN", "capacity", "actual", "fitted", "deviation", "slope", "intercept", "r_squared"):
            assert col in rep.columns, col

    def test_capacities_cover_catalogue(self, pipe_info, reference):
        assert list(reference.capacities["DN"]) == list(pipe_info["DN"])
        assert reference.capacities["capacity"].is_monotonic_increasing


class TestModelTerms:
    """invest_cost and heat_loss are the linearised terms used in the MILP."""

    def test_invest_cost(self, reference):
        f = reference.street_cost
        assert reference.invest_cost(STREET_PIPE, 10.0, 100.0, 1) == pytest.approx(
            10.0 * (f.slope * 100.0 + f.intercept)
        )

    def test_not_built_costs_nothing(self, reference):
        assert reference.invest_cost(HOUSE_CONNECTION, 25.0, 0.0, 0) == 0.0
        assert reference.heat_loss(HOUSE_CONNECTION, 25.0, 0.0, 0) == 0.0

    def test_heat_loss_in_kw(self, reference):
        f = reference.house_loss
        assert reference.heat_loss(HOUSE_CONNECTION, 20.0, 30.0, 1) == pytest.approx(
            (f.slope * 30.0 + f.intercept) * 20.0 / 1000
        )

    def test_source_connection_uses_street_fits(self, reference):
        from fheat_core.optimization import SOURCE_CONNECTION

        assert reference.fit("cost", SOURCE_CONNECTION) is reference.street_cost
        assert reference.fit("loss", SOURCE_CONNECTION) is reference.street_loss
