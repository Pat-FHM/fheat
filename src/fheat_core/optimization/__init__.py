"""Optional MILP network optimisation (``network_method = "milp"``).

Solver-dependent modules import oemof.solph lazily; install the ``[opt]``
extra to use them. Pre-processing such as :mod:`.linearize` has no extra
dependencies.
"""

MISSING_OPT_EXTRA = (
    'The MILP network optimisation needs the optional extra "opt" '
    '(oemof.solph, HiGHS): pip install "fheat[opt]"'
)

# Edge types (``cols.TYPE``) as set by ``fheat_core.algorithms.network``.
HOUSE_CONNECTION = "Hausanschluss"
STREET_PIPE = "Straßenleitung"
SOURCE_CONNECTION = "Quellenanschluss"
