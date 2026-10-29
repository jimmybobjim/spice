from collections.abc import Callable, Hashable
from dataclasses import dataclass

from circuit.terminals import Terminals
from solver.waveformHelpers import FloatArray

# ---- nonlinear device models -------------------------------------------------
#
# Every nonlinear device is described in a normalised frame so that one piece of
# Newton/stamping code serves all of them:
#
#   terminals  the nodes it touches, in role order
#   controls   the voltages the model depends on, c = sign * (T @ terminal voltages)
#   model      c -> (terminal currents into the device, Jacobian d(currents)/d(c))
#   limit      (new c, previous c) -> a c whose step is safe for Newton
#
# `sign` is +1 for NPN/NMOS/diode and -1 for PNP/PMOS, which are mirror images.

type DeviceModel = Callable[[FloatArray], tuple[FloatArray, FloatArray]]
type Limiter = Callable[[FloatArray, FloatArray], FloatArray]

@dataclass(frozen=True, slots=True)
class Nonlinear[E: Hashable]:
    edge: E
    roles: type[Terminals]
    nodes: tuple[int | None, ...]  # unknown index per role, None for ground
    sign: float
    T: FloatArray                  # (controls x terminals) raw voltage differences
    model: DeviceModel
    limit: Limiter
