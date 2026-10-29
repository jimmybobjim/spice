"""Evaluate probe expressions such as ``v(out)``, ``v(a,b)``, ``i(v1)``, ``vdb(out)`` or ``ic(q1)``.

Voltage/current functions take a suffix that picks the part of a complex (AC) value:

    v, i      magnitude for .ac, the value itself otherwise
    vm  im    magnitude          vp  ip    phase in degrees
    vdb idb   20 log10 |x|       vr  ir    real part          vi  ii    imaginary part

Transistor terminal currents: ``ic ib ie`` (BJT) and ``id ig is`` (MOSFET), flowing *into* the device.
"""
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from circuit.components import Bjt, Mosfet
from circuit.terminals import BjtTerminals, MosfetTerminals, Terminals
from errors import ProbeError
from simulate import Solution
from spiceParser import PrintRequest
from util.spicetext import canonicalNode

_CALL = re.compile(r"^([a-z]+)\(([^()]*)\)$")
_SUFFIXES = ("", "m", "p", "db", "r", "i")
_TERMINAL_CURRENTS: dict[str, tuple[type, Terminals]] = {
    "ic": (Bjt, BjtTerminals.COLLECTOR), "ib": (Bjt, BjtTerminals.BASE), "ie": (Bjt, BjtTerminals.EMITTER),
    "id": (Mosfet, MosfetTerminals.DRAIN), "ig": (Mosfet, MosfetTerminals.GATE), "is": (Mosfet, MosfetTerminals.SOURCE),
}

def _part(values: NDArray[Any], suffix: str, isComplex: bool) -> NDArray[np.float64]:
    z = np.asarray(values, dtype=np.complex128)
    match suffix:
        case "":
            return np.abs(z) if isComplex else z.real
        case "m":
            return np.abs(z)
        case "p":
            return np.degrees(np.angle(z))
        case "db":
            with np.errstate(divide="ignore"):
                return 20.0 * np.log10(np.abs(z))
        case "r":
            return z.real
        case _:
            return z.imag

def evaluate(probe: str, solution: Solution) -> NDArray[np.float64]:
    """The value of `probe` at every point of the solution's axis."""
    match = _CALL.match(probe.strip().lower())
    if match is None:
        raise ProbeError(f"{probe!r} is not a probe expression like v(out) or i(r1)")
    function, argument = match.groups()
    names = [name.strip() for name in argument.split(",")]

    if function in _TERMINAL_CURRENTS:
        kind, terminal = _TERMINAL_CURRENTS[function]
        device = solution.devices.get(names[0]) if len(names) == 1 else None
        if not isinstance(device, kind):
            raise ProbeError(f"{probe}: {names[0]!r} is not a {kind.__name__.upper()} in the circuit")
        return _part(solution.terminalCurrents[device.name][terminal], "", solution.isComplex)

    for base in ("v", "i"):
        if function.startswith(base) and function[1:] in _SUFFIXES:
            suffix = function[1:]
            break
    else:
        raise ProbeError(f"{probe}: unknown function {function!r}")

    if base == "v":
        if len(names) not in (1, 2) or not all(names):
            raise ProbeError(f"{probe}: v() takes one node, or two for a difference")
        nodes = [canonicalNode(name) for name in names]
        for node in nodes:
            if node not in solution.voltages:
                raise ProbeError(f"{probe}: there is no node {node!r}")
        value = solution.voltages[nodes[0]] - (solution.voltages[nodes[1]] if len(nodes) == 2 else 0.0)
        return _part(value, suffix, solution.isComplex)

    if len(names) != 1 or not names[0]:
        raise ProbeError(f"{probe}: i() takes one device name")
    name = names[0]
    if name not in solution.devices:
        raise ProbeError(f"{probe}: there is no device {name!r}")
    if name not in solution.currents:
        raise ProbeError(f"{probe}: {name!r} has more than two terminals; use ic/ib/ie (BJT) or id/ig/is (MOSFET)")
    return _part(solution.currents[name], suffix, solution.isComplex)

def defaultProbes(solution: Solution) -> tuple[str, ...]:
    """Every node voltage, plus the current through each voltage source."""
    nodes = [f"v({node})" for node in solution.voltages if node != "0"]
    sources = [f"i({name})" for name, device in solution.devices.items() if device.letter == "V"]
    return tuple(nodes + sources)

def probesFor(solution: Solution, requests: list[PrintRequest]) -> tuple[str, ...]:
    """The probes to show for a solution: those of matching ``.print/.plot/.probe`` lines, else the defaults."""
    chosen: list[str] = []
    for request in requests:
        if request.analysis in (None, solution.analysis.kind):
            chosen.extend(probe for probe in request.probes if probe not in chosen)
    return tuple(chosen) or defaultProbes(solution)
