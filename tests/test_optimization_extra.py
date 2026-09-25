"""The MILP modules need the optional extra [opt] and say so if it is missing.

Runs without oemof.solph: the missing packages are simulated. T11: the
shortest-path network step works without the extra.
"""
from __future__ import annotations

import builtins
import importlib
import subprocess
import sys
import textwrap

import pytest

from fheat_core.optimization import MISSING_OPT_EXTRA


@pytest.fixture
def without_opt_packages(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in {"oemof", "pyomo"}:
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod in ("fheat_core.optimization.block", "fheat_core.optimization.energysystem",
                "fheat_core.optimization.network"):
        monkeypatch.delitem(sys.modules, mod, raising=False)


@pytest.mark.parametrize("module", ["block", "energysystem", "network"])
def test_clear_message_without_extra(without_opt_packages, module):
    with pytest.raises(ImportError, match="fheat\\[opt\\]") as info:
        importlib.import_module(f"fheat_core.optimization.{module}")
    assert str(info.value) == MISSING_OPT_EXTRA


_BLOCK_OPT = """
import sys

class BlockOpt:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"oemof", "pyomo"}:
            raise ImportError(name)

sys.meta_path.insert(0, BlockOpt())
"""


def test_preprocessing_needs_no_extra():
    """linearize, preprocess and glf_terms import in a process without oemof and Pyomo."""
    code = _BLOCK_OPT + textwrap.dedent("""
        import fheat_core.optimization.glf_terms
        import fheat_core.optimization.linearize
        import fheat_core.optimization.postprocess
        import fheat_core.optimization.preprocess
        assert not any(m.split(".")[0] in {"oemof", "pyomo"} for m in sys.modules)
    """)
    subprocess.run([sys.executable, "-c", code], check=True)


def test_t11_shortest_path_step_without_extra(tmp_path):
    """The default network step runs in a process without oemof and Pyomo."""
    code = _BLOCK_OPT + textwrap.dedent("""
        import geopandas as gpd
        from shapely.geometry import LineString, Point, Polygon
        from fheat_core import columns as cols
        from fheat_core.config import FHeatConfig
        from fheat_core.state import PipelineState
        from fheat_core.steps import network

        crs = "EPSG:25832"
        buildings = gpd.GeoDataFrame({
            cols.BUILDING_ID: [0, 1], cols.CONNECT: [1, 1],
            cols.HEAT_DEMAND: [15000.0, 25000.0], cols.THERMAL_POWER: [10.0, 15.0],
            cols.FULL_LOAD_HOURS: [1500.0, 1666.0], cols.LOAD_PROFILE: ["EFH", "EFH"],
            "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                         Polygon([(50, 0), (60, 0), (60, 10), (50, 10)])],
        }, crs=crs)
        streets = gpd.GeoDataFrame({cols.ROUTABLE: [1], "geometry": [LineString([(-10, -5), (100, -5)])]}, crs=crs)
        source = gpd.GeoDataFrame({"geometry": [Point(-10, -5)]}, crs=crs)

        class Adapter:
            def provide_pipe_info(self):
                return None

        state = PipelineState(buildings_gdf=buildings, streets_gdf=streets, source_gdf=source)
        network.run(state, FHeatConfig(output_dir=r"%s"), Adapter())
        assert len(state.net_gdf) > 0
        assert not any(m.split(".")[0] in {"oemof", "pyomo"} for m in sys.modules)
        assert "fheat_core.optimization.network" not in sys.modules
    """ % tmp_path)
    subprocess.run([sys.executable, "-c", code], check=True)
