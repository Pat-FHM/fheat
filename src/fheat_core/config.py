from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from fheat_core.optimization import GLF_OFF, GLF_REFERENCE, ON_UNREACHABLE


@dataclass
class FHeatConfig:
    """Pipeline parameters.

    Contains only calculation parameters, NOT data-source parameters.
    Data-source-specific configuration (paths, column mappings, regional
    parameters such as municipality name) belongs in the adapter constructor.
    """

    # Network parameters
    supply_temperature: float = 80.0
    return_temperature: float = 50.0
    network_method: str = "shortest_path"   # "shortest_path" (Dijkstra) or "milp" (needs extra [opt])
    optimization: Optional[OptimizationConfig] = None   # "milp" only; None → defaults

    # WLD / suitability polygons
    wld_threshold: float = 500.0
    buffer_distance: float = 50.0

    # BDEW SLP parameters (German standard load profile)
    building_class: int = 3      # NRW default per BGW 2006
    wind_class: int = 1
    year: int = 2022

    # Output
    output_dir: str = "./output"
    output_format: str = "gpkg"
    output_language: str = "de"  # "de" = German column labels, "raw" = canonical IDs

    _ALLOWED_FORMATS: ClassVar[frozenset] = frozenset({"gpkg", "fgb", "geojson", "gml"})
    _ALLOWED_NETWORK_METHODS: ClassVar[frozenset] = frozenset({"shortest_path", "milp"})
    _ALLOWED_LANGUAGES: ClassVar[frozenset] = frozenset({"de", "raw"})

    def __post_init__(self) -> None:
        if self.supply_temperature <= self.return_temperature:
            raise ValueError(
                f"supply_temperature ({self.supply_temperature} °C) must be greater than "
                f"return_temperature ({self.return_temperature} °C)."
            )
        if self.output_format not in self._ALLOWED_FORMATS:
            raise ValueError(
                f"output_format '{self.output_format}' is not allowed. "
                f"Allowed formats: {sorted(self._ALLOWED_FORMATS)}"
            )
        if self.output_language not in self._ALLOWED_LANGUAGES:
            raise ValueError(
                f"output_language '{self.output_language}' is not allowed. "
                f"Allowed values: {sorted(self._ALLOWED_LANGUAGES)}"
            )
        if self.network_method not in self._ALLOWED_NETWORK_METHODS:
            raise ValueError(
                f"network_method '{self.network_method}' is not allowed. "
                f"Allowed values: {sorted(self._ALLOWED_NETWORK_METHODS)}"
            )
        if self.network_method == "milp" and self.optimization is None:
            self.optimization = OptimizationConfig()


@dataclass
class OptimizationConfig:
    """Parameters of the MILP network optimisation (``fheat_core.optimization``).

    The economic defaults are PLATZHALTER values from the literature and must
    be replaced by project-specific values.
    """

    # Simultaneity in the route choice: "referenz" (exact on bridges,
    # reference-tree GLF elsewhere) or "aus" (no GLF, same cost line, for comparison)
    glf_mode: str = GLF_REFERENCE

    # Economics — PLATZHALTER, see sources
    interest_rate: float = 0.08              # Lambert et al. 2025 (internal rate of return)
    lifetime_pipes: int = 20                 # [a] Lambert et al. 2025
    source_capex_eur_per_kw: float = 598.0   # [€/kW] Lambert et al. 2025, Tab. 7 (central air-water HP)
    lifetime_source: int = 20                # [a] Lambert et al. 2025, Tab. 7
    heat_cost_eur_per_kwh: float = 0.08      # [€/kWh] Lambert et al. 2024, Tab. 1

    # Pre-processing
    soil_temperature: float = 10.0           # [°C] heat loss 2 · U · (T_mean − T_soil), as F|Heat
    regression_max_deviation: float = 0.15   # warn if a cost/loss line deviates more at one DN
    on_unreachable: str = "warn"             # buildings without a route to the source: "warn" | "error"

    # Solver (HiGHS): stops at the absolute gap or the time limit only
    mip_abs_gap: float | str = "auto"        # [€/a]; "auto": 0.5 % of the pipe annuity of the shortest-path tree
    time_limit_s: float = 300.0

    _ALLOWED_GLF_MODES: ClassVar[frozenset] = frozenset({GLF_REFERENCE, GLF_OFF})
    _ALLOWED_ON_UNREACHABLE: ClassVar[frozenset] = ON_UNREACHABLE

    def __post_init__(self) -> None:
        if self.glf_mode not in self._ALLOWED_GLF_MODES:
            raise ValueError(
                f"glf_mode '{self.glf_mode}' is not allowed. "
                f"Allowed values: {sorted(self._ALLOWED_GLF_MODES)}"
            )
        if self.on_unreachable not in self._ALLOWED_ON_UNREACHABLE:
            raise ValueError(
                f"on_unreachable '{self.on_unreachable}' is not allowed. "
                f"Allowed values: {sorted(self._ALLOWED_ON_UNREACHABLE)}"
            )
        if self.regression_max_deviation <= 0:
            raise ValueError(
                f"regression_max_deviation ({self.regression_max_deviation}) must be greater than 0."
            )
        if self.interest_rate <= 0:
            raise ValueError(f"interest_rate ({self.interest_rate}) must be greater than 0.")
        for name in ("lifetime_pipes", "lifetime_source", "time_limit_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be greater than 0.")
        for name in ("source_capex_eur_per_kw", "heat_cost_eur_per_kwh"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must not be negative.")
        if self.mip_abs_gap != "auto" and (
            isinstance(self.mip_abs_gap, str) or self.mip_abs_gap < 0
        ):
            raise ValueError(
                f"mip_abs_gap ({self.mip_abs_gap!r}) must be 'auto' or a non-negative number [€/a]."
            )
