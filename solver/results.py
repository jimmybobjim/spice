from collections.abc import Hashable, Mapping
from dataclasses import dataclass

from circuit.terminals import Terminals
from solver.waveformHelpers import ComplexArray, FloatArray

@dataclass(frozen=True, slots=True)
class DcResult[V: Hashable, E: Hashable]:
    voltages: Mapping[V, float]  # every vertex, ground = 0
    currents: Mapping[E, float]  # every two-terminal device, first role -> second role
    terminalCurrents: Mapping[E, Mapping[Terminals, float]]  # every device: current into each role

@dataclass(frozen=True, slots=True)
class AcResult[V: Hashable, E: Hashable]:
    frequencies: FloatArray  # hertz
    voltages: Mapping[V, ComplexArray]  # complex phasors, one value per frequency
    currents: Mapping[E, ComplexArray]
    terminalCurrents: Mapping[E, Mapping[Terminals, ComplexArray]]

@dataclass(frozen=True, slots=True)
class TransientResult[V: Hashable, E: Hashable]:
    times: FloatArray  # seconds
    voltages: Mapping[V, FloatArray]  # one value per time point
    currents: Mapping[E, FloatArray]
    terminalCurrents: Mapping[E, Mapping[Terminals, FloatArray]]