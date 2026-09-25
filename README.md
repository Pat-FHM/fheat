# F|Heat

**F|Heat** is a Python toolkit for **district-heating network planning from geodata**. Given buildings, streets, parcels and a heat-source location, it computes heat-line density (*Wärmeliniendichte*, WLD), derives suitability polygons (*Eignungspolygone*), dimensions a pipe network (diameters, flow velocities, heat losses, simultaneity factor / *Gleichzeitigkeitsfaktor*), and produces an hourly load profile and a result summary.

> Domain terms are German because the tool targets German municipal heat planning (*kommunale Wärmeplanung*), in particular the federal state of North Rhine-Westphalia (NRW).

## Packages

The repository is a monorepo of three installable packages:

| Package | Role |
|---|---|
| [`fheat_core`](src/fheat_core/) | Adapter-agnostic pipeline: orchestrator, data schemas, geometry/network/SLP algorithms, and the step functions. This is the engine. |
| [`fheat_nrw`](src/fheat_nrw/) | Adapter that **downloads NRW open geodata** (building heat model via OpenGeoData NRW, cadastral parcels via the ALKIS WFS) and processes it into schema-compliant frames. |
| [`fheat_flex`](src/fheat_flex/) | Adapter for **user-supplied** GeoPackages, with a column mapping onto the core schema. Use this when you bring your own data. |

All three are import packages shipped from a single distribution named `fheat` (see [Installation](#installation)). The whole logic is derived from the former QGIS plugin to adress the flexibility with other applications.

## Architecture

```
DataAdapter ──fetch──▶ PipelineState ──▶ FHeatOrchestrator ──▶ outputs (.gpkg + summary)
(source of data)        (frames)          (runs the phases)
```

- A **`DataAdapter`** (`fheat_nrw` or `fheat_flex`) supplies four schema-compliant GeoDataFrames: `buildings`, `streets`, `parcels`, `source`. All region- and source-specific logic lives in the adapter.
- **`FHeatConfig`** holds only *calculation* parameters (temperatures, WLD threshold, buffer distance, SLP year/class).
- **`FHeatOrchestrator`** runs the pipeline phase by phase, and can resume from any phase:

  | Phase | Step | Produces |
  |---|---|---|
  | `INITIAL` | download | input frames from the adapter |
  | `DOWNLOADED` | adjust | cleaned geometry, schema-validated frames |
  | `ADJUSTED` | status | heat-line density + suitability polygons |
  | `STATUS` | network | pipe network with sizing & losses (shortest path, or [MILP](#milp-network-optimisation-optional)) |
  | `NETWORK` | results | hourly load profile + result summary |

Adapters must produce data conforming to the contracts in [`schemas.py`](src/fheat_core/schemas.py); the core validates against the same schemas as it goes.

## Installation

Requires **Python ≥ 3.10**. The geospatial stack (GeoPandas, Shapely, etc.) is easiest to install with `conda`/`mamba`, but `pip` works on most platforms.

Not yet published to PyPI, so install from the repository. From the repo root:

```bash
pip install -e .              # core pipeline + flexible adapter (your own data)
pip install -e ".[nrw]"       # + NRW auto-download adapter (owslib, lxml)
pip install -e ".[full]"      # everything: NRW adapter + German holidays
pip install -e ".[full,dev]"  # everything + pytest, for development
pip install -e ".[opt]"       # + optional MILP network optimisation (oemof.solph, HiGHS)
```

All three import packages — `fheat_core`, `fheat_nrw`, `fheat_flex` — ship from the single `fheat` distribution. The extras only add the optional third-party dependencies a given adapter needs: the NRW adapter pulls in `owslib`/`lxml`, and holiday-aware load profiles pull in `workalendar`. All bundled reference data ships as plain text — CSV for tabular tables (pipe catalogue, example temperature year, NRW city index) and JSON for the keyed building-typology lookups (`fheat_nrw/data/*.json`) — so no package reads Excel. The separate `[excel]` extra adds `openpyxl` only for the optional `.xlsx` *export* in the examples.

## Quick start

### NRW adapter — download and analyse automatically

```python
from fheat_core.config import FHeatConfig
from fheat_core.orchestrator import FHeatOrchestrator
from fheat_nrw.adapter.data_adapter import NRWDataAdapter

adapter = NRWDataAdapter(
    city_name="Burgsteinfurt",              # Stadtteil/Gemarkung
    source_coordinates=(52.1592, 7.3268),   # (lat, lon) WGS84 — heat-source location
    heat_attribute="RW_WW",                 # NRW raw heat-demand column
)

config = FHeatConfig(
    supply_temperature=70.0,
    return_temperature=50.0,
    wld_threshold=500.0,     # kWh/(a·m)
    buffer_distance=50.0,    # m
    year=2022,
    output_dir="./output",
)

orch = FHeatOrchestrator(config=config, adapter=adapter)
orch.run_all()                       # download → adjust → status → network → results
saved = orch.save_outputs()          # writes GeoPackages to output_dir
print(orch.state.result_summary)
```

Running the NRW adapter requires internet access (NRW WFS and ZIP downloads).

### Flexible adapter — bring your own data

This is a feature which was most requested by users outside NRW. The `fheat_flex` adapter takes **user-supplied GeoPackages** and a column mapping onto the canonical schema. The example below assumes you have a directory `data/` with four GeoPackages: `buildings.gpkg`, `streets.gpkg`, `parcels.gpkg`, and `source.gpkg`. The column names in your data can be arbitrary; the adapter maps them onto the canonical schema.

```python
from fheat_flex.adapter.data_adapter import FlexDataAdapter

adapter = FlexDataAdapter(
    buildings_path="data/buildings.gpkg",
    streets_path="data/streets.gpkg",
    parcels_path="data/parcels.gpkg",
    source="data/source.gpkg",
    column_map={                  # map YOUR columns onto the canonical schema
        "waermebedarf":          "heat_demand",       # see fheat_core.columns
        "vollbenutzungsstunden": "full_load_hours",
        "lastprofil":            "load_profile",
    },
)
# ... same FHeatConfig + FHeatOrchestrator usage as above
```

The pipeline uses canonical, language-neutral column names internally (see
`fheat_core/columns.py`). On export, `save_outputs()` translates them back to
German display labels by default (`output_language="de"`); set
`output_language="raw"` to keep the canonical identifiers.

Worked examples are in [`examples/`](examples/): [`burgsteinfurt.py`](examples/burgsteinfurt.py) (NRW adapter, runnable with the bundled planning area `planungsgebiet.gpkg`) and an introductory notebook [`fheat_einfuehrung.ipynb`](examples/fheat_einfuehrung.ipynb). If you want to add an own area of interest for the analysis you can import it by exporting a polygon with using QGIS.

## MILP network optimisation (optional)

By default the network step connects every building along its shortest path
(Dijkstra). With `network_method="milp"` it designs a cost-optimal radial
network instead: a mixed-integer linear program on an
[oemof.solph](https://github.com/oemof/oemof-solph) energy system (producer →
network → consumers and losses), solved once with HiGHS. It needs the `[opt]`
extra; without it the default step is unchanged.

```python
from fheat_core.config import FHeatConfig, OptimizationConfig

config = FHeatConfig(
    network_method="milp",
    optimization=OptimizationConfig(time_limit_s=300),   # None → defaults
)
```

What the step does:

1. builds the street graph as the shortest-path step does, then simplifies it
   (integer node IDs, dead ends removed, street chains merged, parallel
   sections reduced, bridges fixed) without changing the optimum;
2. linearises pipe cost [€/m] and heat loss [W/m] over the design capacity;
3. takes the simultaneity factor (GLF) into the route choice:
   `glf_mode="referenz"` uses the exact GLF behind every bridge and the GLF of
   the shortest-path tree elsewhere; `"aus"` sizes without GLF, for comparison;
4. solves the MILP once; HiGHS stops at the absolute gap `mip_abs_gap` [€/a]
   or at `time_limit_s`;
5. re-calculates the chosen network with the exact GLF, the real DN
   (`calculate_diameter_velocity_loss`) and the real pipe costs.

Every reachable building with `connect == 1` is connected (forced mode).
Buildings without a route to the source get `connect = 0` and
`connection_status = "nicht erreichbar"` (`on_unreachable="error"` stops
instead). The formulation (sets, variables, numbered constraints, objective) is
documented in [`optimization/block.py`](src/fheat_core/optimization/block.py).

**Outputs.** `net_gdf` satisfies `NetSchema` and has additional columns that
put model and post-calculated values side by side:

| Column | German label | Meaning |
|---|---|---|
| `glf` | `GLF` | exact GLF of the section |
| `glf_model` | `GLF_Modell` | GLF used in the MILP |
| `capacity_model` | `Kapazitaet_Modell [kW]` | design capacity in the MILP (compare with `thermal_power_glf`) |
| `invest_cost_model` | `Investition_Modell [EUR]` | linearised pipe investment |
| `invest_cost` | `Investition [EUR]` | pipe investment of the chosen DN |
| `annual_cost` | `Annuitaet [EUR/a]` | annuity of `invest_cost` |

The buildings get `connection_status` (`Anschlussstatus`). The result summary
adds `milp_*` key figures (objective, gap, solve time, producer capacity
GLF(N) · ΣQ + losses, real and linearised pipe investment and their deviation,
unreachable buildings). `state.optimization_report` holds the full report,
including the quality of every cost and loss line per DN.

**Parameters** (`OptimizationConfig`):

| Field | Default | Note |
|---|---|---|
| `glf_mode` | `"referenz"` | `"aus"`: no simultaneity, for comparison |
| `interest_rate`, `lifetime_pipes` | 0.08, 20 a | placeholder, Lambert et al. 2025 |
| `source_capex_eur_per_kw`, `lifetime_source` | 598 €/kW, 20 a | placeholder, Lambert et al. 2025, Tab. 7 (central air-water heat pump) |
| `heat_cost_eur_per_kwh` | 0.08 | placeholder, Lambert et al. 2024, Tab. 1 |
| `soil_temperature` | 10 °C | heat loss 2 · U · (T_mean − T_soil), as the shortest-path step |
| `regression_max_deviation` | 0.15 | warning if a cost or loss line deviates more at one DN |
| `on_unreachable` | `"warn"` | `"error"`: stop if a building cannot be reached |
| `mip_abs_gap` | `"auto"` | [€/a]; `"auto"` = 0.5 % of the pipe annuity of the shortest-path tree |
| `time_limit_s` | 300 | solver time limit |

The pipe costs in `fheat_core/data/pipe_costs.csv` (Lambert et al. 2025,
Tab. 8) and the economic defaults are **placeholders** (`PLATZHALTER`) and must
be replaced by project-specific values; an adapter can supply its own costs
via `provide_pipe_costs()`.

**Limits.** One heat source, which must lie beside the street network (not
exactly on a street vertex). One time step (annual energy, design case). The
GLF of sections that are not bridges is estimated from the shortest-path tree;
the post-calculation reports the exact value.

## Tests

```bash
pip install -e ".[dev]"
pytest                       # runs the offline suite
pytest -m "not network"      # explicitly skip tests that hit the live NRW services
```

Tests marked `network` require internet; `slow` tests are long-running.

## License

Distributed under the **GNU General Public License v3.0 or later** — see [LICENSE](LICENSE).

## Funding notice

![Förderlogo](https://www-backend.fh-muenster.de/iep/fheat/f-heat.connect-start.php.media/73867/Foerdermittelgeber_Logo.jpg.scaled/b159e0672c75c3bc726462d37fb2efd2.jpg)

This project is co-funded by the European Union and the State of North Rhine-Westphalia under the EFRE/JTF Programme NRW 2021–2027, supported by the Ministry of Economic Affairs, Industry, Climate Action and Energy of North Rhine-Westphalia. Project duration: 1 December 2025 – 30 November 2028.
