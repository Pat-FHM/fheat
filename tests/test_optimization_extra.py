"""The MILP modules need the optional extra [opt] and say so if it is missing.

Runs without oemof.solph: the missing packages are simulated.
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
    for mod in ("fheat_core.optimization.block", "fheat_core.optimization.energysystem"):
        monkeypatch.delitem(sys.modules, mod, raising=False)


@pytest.mark.parametrize("module", ["block", "energysystem"])
def test_clear_message_without_extra(without_opt_packages, module):
    with pytest.raises(ImportError, match="fheat\\[opt\\]") as info:
        importlib.import_module(f"fheat_core.optimization.{module}")
    assert str(info.value) == MISSING_OPT_EXTRA


def test_preprocessing_needs_no_extra():
    """linearize, preprocess and glf_terms import in a process without oemof and Pyomo."""
    code = textwrap.dedent("""
        import sys

        class BlockOpt:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"oemof", "pyomo"}:
                    raise ImportError(name)

        sys.meta_path.insert(0, BlockOpt())
        import fheat_core.optimization.glf_terms
        import fheat_core.optimization.linearize
        import fheat_core.optimization.preprocess
        assert not any(m.split(".")[0] in {"oemof", "pyomo"} for m in sys.modules)
    """)
    subprocess.run([sys.executable, "-c", code], check=True)
