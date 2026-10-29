"""Run a parsed netlist's analyses and collect the answers in one uniform shape (`Solution`)."""
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from circuit.circuit import buildCircuit
from circuit.components import Component, CurrentSource, VoltageSource
from circuit.terminals import Terminals
from errors import SimulationError
from solver.options import NewtonOptions
from solver.solver import solveAc, solveDc, solveTransient
from solver.waveformHelpers import FloatArray, linearSweep, logSweep, octaveSweep
from spiceParser import AcAnalysis, Analysis, DcSweep, Netlist, OpAnalysis, TranAnalysis, flatten

GROUND = "0"
MAX_POINTS = 1_000_000   # refuse sweeps and transients bigger than this rather than run out of memory

@dataclass(frozen=True)
class Solution:
    """The result of one analysis. Every quantity is an array with one entry per point of `axis`
    (a single point for ``.op``); the arrays are complex for ``.ac`` and real otherwise."""
    analysis: Analysis
    axisName: str                                   # "" for .op, else "time", "frequency" or the swept source
    axis: FloatArray
    voltages: Mapping[str, NDArray[Any]]            # by node
    currents: Mapping[str, NDArray[Any]]            # by two-terminal device: first terminal -> second
    terminalCurrents: Mapping[str, Mapping[Terminals, NDArray[Any]]]
    devices: Mapping[str, Component]

    @property
    def isComplex(self) -> bool:
        return isinstance(self.analysis, AcAnalysis)

def solverOptions(netlist: Netlist) -> tuple[float, NewtonOptions]:
    """``(gmin, NewtonOptions)`` from the netlist's ``.options`` line."""
    o = netlist.options
    defaults = NewtonOptions()
    return o.get("gmin", 1e-12), NewtonOptions(
        maxIterations=int(o.get("itl1", defaults.maxIterations)),
        reltol=o.get("reltol", defaults.reltol),
        vntol=o.get("vntol", defaults.vntol),
        abstol=o.get("abstol", defaults.abstol),
    )

def _requireGround(components: list[Component]) -> None:
    if not any(GROUND in component.nodes for component in components):
        raise SimulationError("the circuit has no ground node: connect something to node 0 (or gnd)")

def _acFrequencies(analysis: AcAnalysis) -> FloatArray:
    try:
        match analysis.variation:
            case "dec":
                return logSweep(analysis.start, analysis.stop, analysis.points)
            case "oct":
                return octaveSweep(analysis.start, analysis.stop, analysis.points)
            case _:
                return linearSweep(analysis.start, analysis.stop, analysis.points)
    except ValueError as error:
        raise SimulationError(f".ac (line {analysis.line}): {error}") from None

def _sweepValues(analysis: DcSweep) -> FloatArray:
    count = math.floor((analysis.stop - analysis.start) / analysis.step + 1e-9) + 1
    if count > MAX_POINTS:
        raise SimulationError(f".dc (line {analysis.line}) would need {count} points; the limit is {MAX_POINTS}")
    return analysis.start + analysis.step * np.arange(count, dtype=np.float64)

def runAnalysis(
    netlist: Netlist, analysis: Analysis, components: list[Component], *,
    method: Literal["trapezoidal", "backward_euler"] = "trapezoidal",
) -> Solution:
    """Run one analysis on the flat device list."""
    _requireGround(components)
    gmin, options = solverOptions(netlist)
    devices = {component.name: component for component in components}
    circuit = buildCircuit(components)

    match analysis:
        case OpAnalysis():
            r = solveDc(circuit, GROUND, gmin=gmin, options=options)
            return Solution(
                analysis, "", np.zeros(1),
                {node: np.array([v]) for node, v in r.voltages.items()},
                {edge: np.array([i]) for edge, i in r.currents.items()},
                {edge: {role: np.array([i]) for role, i in roles.items()} for edge, roles in r.terminalCurrents.items()},
                devices,
            )

        case DcSweep(source=name):
            target = devices.get(name)
            if not isinstance(target, (VoltageSource, CurrentSource)):
                raise SimulationError(f".dc (line {analysis.line}): {name!r} is not an independent voltage or current source in the circuit")
            values = _sweepValues(analysis)
            results = []
            for value in values:
                swept = replace(target, source=replace(target.source, dc=float(value)))
                point = buildCircuit(swept if c.name == name else c for c in components)
                results.append(solveDc(point, GROUND, gmin=gmin, options=options))
            first = results[0]
            return Solution(
                analysis, name, values,
                {node: np.array([r.voltages[node] for r in results]) for node in first.voltages},
                {edge: np.array([r.currents[edge] for r in results]) for edge in first.currents},
                {edge: {role: np.array([r.terminalCurrents[edge][role] for r in results]) for role in roles}
                 for edge, roles in first.terminalCurrents.items()},
                devices,
            )

        case AcAnalysis():
            freqs = _acFrequencies(analysis)
            if freqs.size > MAX_POINTS:
                raise SimulationError(f".ac (line {analysis.line}) would need {freqs.size} points; the limit is {MAX_POINTS}")
            r = solveAc(circuit, GROUND, freqs, gmin=gmin, options=options)
            return Solution(analysis, "frequency", r.frequencies, r.voltages, r.currents, r.terminalCurrents, devices)

        case TranAnalysis():
            step = analysis.step if analysis.max_step is None else min(analysis.step, analysis.max_step)
            if analysis.stop / step > MAX_POINTS:
                raise SimulationError(
                    f".tran (line {analysis.line}) would need {analysis.stop / step:.3g} time points; the limit is {MAX_POINTS}. "
                    "Increase tstep or shorten tstop")
            r = solveTransient(
                circuit, GROUND, stop=analysis.stop, step=step, method=method,
                useInitialConditions=analysis.use_initial_conditions, gmin=gmin, options=options,
            )
            keep = r.times >= analysis.start - 1e-12 * analysis.stop   # tstart only hides the early part of the output
            return Solution(
                analysis, "time", r.times[keep],
                {node: v[keep] for node, v in r.voltages.items()},
                {edge: i[keep] for edge, i in r.currents.items()},
                {edge: {role: i[keep] for role, i in roles.items()} for edge, roles in r.terminalCurrents.items()},
                devices,
            )

def simulate(
    netlist: Netlist, *, method: Literal["trapezoidal", "backward_euler"] = "trapezoidal",
) -> tuple[list[Solution], list[str]]:
    """Run every analysis in the netlist (an ``.op`` if it has none). Returns the solutions and any warnings."""
    components = flatten(netlist)
    warnings = list(netlist.warnings)

    analyses = list(netlist.analyses)
    if not analyses:
        warnings.append("no analysis (.op, .dc, .ac, .tran) in the netlist; running .op")
        analyses = [OpAnalysis()]

    if any(isinstance(a, AcAnalysis) for a in analyses) and not any(
        isinstance(c, (VoltageSource, CurrentSource)) and c.source.ac != 0 for c in components
    ):
        warnings.append(".ac analysis requested but no source has an AC value (add e.g. 'ac 1' to a source), so every result will be zero")

    return [runAnalysis(netlist, analysis, components, method=method) for analysis in analyses], warnings
