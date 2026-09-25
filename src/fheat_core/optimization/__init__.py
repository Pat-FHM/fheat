"""Optional MILP network optimisation (``network_method = "milp"``).

Solver-dependent modules import oemof.solph lazily; install the ``[opt]``
extra to use them. Pre-processing such as :mod:`.linearize` has no extra
dependencies.
"""

MISSING_OPT_EXTRA = (
    'The MILP network optimisation needs the optional extra "opt" '
    '(oemof.solph, HiGHS): pip install "fheat[opt]"'
)

HOURS_PER_YEAR = 8760   # one time step of the energy system; loss kW ↔ kWh/a
SOIL_TEMPERATURE = 10.0  # [°C], fixed as in calculate_diameter_velocity_loss

# Which buildings are connected, see block.py.
MODE_FORCED = "erzwungen"         # every reachable building
MODE_ECONOMIC = "wirtschaftlich"  # a building if its revenue pays for it

# Simultaneity in the design capacity, see ``glf_terms``.
GLF_REFERENCE = "referenz"   # exact GLF on bridges, reference-tree GLF elsewhere
GLF_OFF = "aus"              # no simultaneity (C = S), for comparison

# Buildings without a route to the source (preprocess): warn or stop.
ON_UNREACHABLE = frozenset({"warn", "error"})

# Values of cols.CONNECTION_STATUS
STATUS_CONNECTED = "angeschlossen"
STATUS_UNREACHABLE = "nicht erreichbar"
STATUS_NOT_ECONOMIC = "wirtschaftlich nicht angeschlossen"

# Edge types (``cols.TYPE``) as set by ``fheat_core.algorithms.network``.
HOUSE_CONNECTION = "Hausanschluss"
STREET_PIPE = "Straßenleitung"
SOURCE_CONNECTION = "Quellenanschluss"
