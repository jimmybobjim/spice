"""Circuit solver (modified nodal analysis + Newton iteration) on top of `Hypergraph`.

Devices
-------
    Resistor, Capacitor, Inductor, VoltageSource, CurrentSource     linear
    Diode (with optional Zener breakdown), Bjt (NPN/PNP), Mosfet (NMOS/PMOS)

Analyses
--------
    solveDc         operating point (capacitors open, inductors shorted)
    solveAc         small-signal frequency response around the DC operating point
    solveTransient  time-domain response (fixed step, trapezoidal or backward Euler)

Describing a circuit
--------------------
Use a `Circuit` (a `Hypergraph` whose roles are terminal enums and whose payloads are
`Component`s). Vertices are nets (any hashable), edges are devices (any hashable key) and
every device must be attached through exactly the terminals in its class's `terminals` enum:

    Resistor, Capacitor, Inductor,
    VoltageSource, CurrentSource     BranchTerminals.POSITIVE, NEGATIVE
    Diode                            DiodeTerminals.ANODE, CATHODE
    Bjt                              BjtTerminals.COLLECTOR, BASE, EMITTER
    Mosfet                           MosfetTerminals.DRAIN, GATE, SOURCE

`buildCircuit(components)` does this wiring from each component's `nodes`. By hand:

    g = buildCircuit([
        VoltageSource("v1", 0, ("in", "0"), Source(dc=5.0, ac=1.0)),
        Resistor("r1", 0, ("in", "out"), 1e3),
        Diode("d1", 0, ("out", "0")),
    ])

    op = solveDc(g, ground="0")
    ac = solveAc(g, ground="0", frequencies=logSweep(1, 1e6, 20))
    tr = solveTransient(g, ground="0", stop=5e-3, step=1e-6)

Results
-------
Each result holds `voltages` (per vertex, ground = 0), `currents` (per two-terminal
device, flowing from its first role through the device to its second) and
`terminalCurrents` (per device, per role: the current flowing *into* the device
through that terminal, which is the only way to describe a transistor).
A voltage source that is delivering power therefore reports a negative current.
A current source pushes `value` from "p" through itself to "n".

How it works
------------
Unknowns are every non-ground node voltage plus one branch current for each
voltage source, inductor and capacitor. Linear devices stamp into two matrices,

        G x + C dx/dt + i_nl(x) = b(t)

where i_nl are the nonlinear device currents. With no nonlinear devices this is
solved directly (DC: one solve; AC: one solve per frequency; transient: one LU
factorisation reused every step).

With nonlinear devices each solve becomes a Newton iteration. Every iteration
replaces each device by its tangent: a conductance matrix (the Jacobian) plus an
equivalent current source, i.e. a linear circuit, which is stamped and solved like
any other. To stop the exponentials overflowing, junction voltages are limited to
a bounded change per iteration (SPICE's `pnjlim`; MOSFET voltages are step-clamped).
If plain Newton fails for the DC point, the solver retries with gmin stepping and
then source stepping; a failed transient step is retried as two half steps.

Model limitations
-----------------
Device models are purely resistive (static). Add `Capacitor` edges for junction or
gate capacitance, and `Resistor` edges for series resistance. The MOSFET is level 1
(square law, channel-length modulation, no body effect, so no "b" terminal) and
the BJT is a transport (Ebers-Moll) model with an optional Early effect. All are
evaluated at a fixed temperature through `thermalVoltage`.

Requires Python 3.12+, numpy and scipy.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

from circuit.circuit import Circuit
from errors import CircuitError, ConvergenceError
from solver.nonlinearDeviceModels import Nonlinear
from solver.options import NewtonOptions
from solver.results import AcResult, DcResult, TransientResult
from solver.system import SINGULAR as _SINGULAR, System, solve
from solver.waveformHelpers import FloatArray


# ---- analyses --------------------------------------------------------------

def solveDc[V: Hashable, E: Hashable](
    circuit: Circuit[V, E], ground: V, *, gmin: float = 1e-12,
    options: NewtonOptions = NewtonOptions(),
) -> DcResult[V, E]:
    """DC operating point using each source's `dc` value.

    `gmin` is a tiny conductance from every node to ground (and across every
    device junction) that keeps otherwise floating nodes solvable. It perturbs
    results by roughly gmin * (resistance), about 1e-9 relative for kilohm circuits.
    """
    system = System(circuit, ground, gmin)
    x = system.operatingPoint(system.G, system.rhs(lambda s: s.dc), options)
    voltages, currents, terminals = system.unpack(
        x[None, :], lambda s: np.array([s.dc]), system.exactCurrents,
    )
    return DcResult(
        {vertex: float(values[0]) for vertex, values in voltages.items()},
        {edge: float(values[0]) for edge, values in currents.items()},
        {edge: {role: float(values[0]) for role, values in roles.items()} for edge, roles in terminals.items()},
    )


def solveAc[V: Hashable, E: Hashable](
    circuit: Circuit[V, E], ground: V, frequencies: Sequence[float] | FloatArray,
    *, gmin: float = 1e-12, options: NewtonOptions = NewtonOptions(),
) -> AcResult[V, E]:
    """Small-signal response at each frequency (hertz), driven by each source's `ac` phasor.

    Nonlinear devices are linearised around the DC operating point (found from each
    source's `dc` value), so a transistor contributes its gm and output conductance.
    """
    f: FloatArray = np.asarray(frequencies, dtype=np.float64)
    if f.ndim != 1 or f.size == 0 or not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise CircuitError("frequencies must be a non-empty 1-D sequence of finite values >= 0")

    system = System(circuit, ground, gmin)
    g = system.G
    jacobians: dict[E, FloatArray] = {}
    if system.nonlinear:
        operating = system.operatingPoint(system.G, system.rhs(lambda s: s.dc), options)
        lin = system.linearize(operating, None)
        g = g + lin.jacobian
        jacobians = {dev.edge: lin.perDevice[k] for k, dev in enumerate(system.nonlinear)}

    omega = 2.0 * np.pi * f
    A = g[None, :, :] + 1j * omega[:, None, None] * system.C[None, :, :]
    b = np.broadcast_to(system.rhsAc(), (f.size, system.size))
    X = solve(A, b[..., None])[..., 0]

    def smallSignalCurrents(dev: Nonlinear[E], X: NDArray[Any]) -> NDArray[Any]:
        v = np.array([np.zeros(X.shape[0]) if node is None else X[:, node] for node in dev.nodes])
        out: NDArray[Any] = jacobians[dev.edge] @ v
        return out

    voltages, currents, terminals = system.unpack(
        X, lambda s: np.full(f.size, s.ac, dtype=np.complex128), smallSignalCurrents,
    )
    return AcResult(f, voltages, currents, terminals)


def _transientLinear(system: System[Any, Any], x0: FloatArray, times: FloatArray,
                     step: float, trapezoidal: bool) -> FloatArray:
    """Fixed-step integration of G x + C dx/dt = b, factorising the step matrix once."""
    G, C = system.G, system.C
    scale = (2.0 if trapezoidal else 1.0) / step
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            lu = lu_factor(G + scale * C)
        except (LinAlgWarning, ValueError) as error:
            raise CircuitError(_SINGULAR) from error
    history = scale * C - (G if trapezoidal else 0.0)

    X: FloatArray = np.empty((times.size, system.size))
    X[0] = x0
    bPrevious = system.rhsAt(0.0)
    for k in range(1, times.size):
        b = system.rhsAt(float(times[k]))
        rhs = b + bPrevious + history @ X[k - 1] if trapezoidal else b + history @ X[k - 1]
        X[k] = lu_solve(lu, rhs)
        bPrevious = b
    return X


def _transientNonlinear(system: System[Any, Any], x0: FloatArray, times: FloatArray, step: float,
                        trapezoidal: bool, options: NewtonOptions) -> FloatArray:
    """Fixed-step integration with a Newton solve per step; a failing step is retried as two half steps."""
    G, C = system.G, system.C

    def advance(
        xPrev: FloatArray, bPrev: FloatArray, iPrev: FloatArray, t0: float, h: float, depth: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        b = system.rhsAt(t0 + h)
        if trapezoidal:
            a = G + (2.0 / h) * C
            r = b + bPrev + ((2.0 / h) * C - G) @ xPrev - iPrev
        else:
            a = G + C / h
            r = b + (C / h) @ xPrev
        try:
            x1 = system.newton(a, r, xPrev, options)
        except ConvergenceError as error:
            if depth >= options.maxStepHalvings:
                raise ConvergenceError(
                    f"transient analysis failed to converge near t = {t0 + h:g} s ({error})"
                ) from error
            xMid, bMid, iMid = advance(xPrev, bPrev, iPrev, t0, h / 2.0, depth + 1)
            return advance(xMid, bMid, iMid, t0 + h / 2.0, h / 2.0, depth + 1)
        return x1, b, system.nonlinearCurrents(x1)

    X: FloatArray = np.empty((times.size, system.size))
    x, b, i = x0, system.rhsAt(0.0), system.nonlinearCurrents(x0)
    X[0] = x
    for k in range(1, times.size):
        x, b, i = advance(x, b, i, float(times[k - 1]), float(times[k] - times[k - 1]), 0)
        X[k] = x
    return X


def solveTransient[V: Hashable, E: Hashable](
    circuit: Circuit[V, E], ground: V, *, stop: float, step: float,
    method: Literal["trapezoidal", "backward_euler"] = "trapezoidal",
    useInitialConditions: bool = False, gmin: float = 1e-12,
    options: NewtonOptions = NewtonOptions(),
) -> TransientResult[V, E]:
    """Time-domain response from t = 0 with a fixed time `step`.

    Sources follow their `waveform` (or `dc` if they have none). `stop` is
    rounded up to a whole number of steps.

    Initial state:
        useInitialConditions=False  start from the DC operating point at t = 0
        useInitialConditions=True   start from each capacitor's `initialVoltage`
                                    and each inductor's `initialCurrent`

    Trapezoidal is second-order accurate but rings if the source jumps at t = 0
    or between steps; backward_euler is first-order but heavily damped, so it is
    the safer choice for stiff circuits or discontinuous drive.

    With nonlinear devices every step runs a Newton iteration (see NewtonOptions),
    and a step that fails to converge is automatically retried as two half steps.
    """
    if not (math.isfinite(stop) and math.isfinite(step) and 0.0 < step <= stop):
        raise CircuitError("need finite values with 0 < step <= stop")
    if method not in ("trapezoidal", "backward_euler"):
        raise CircuitError(f"unknown method {method!r}")
    trapezoidal = method == "trapezoidal"

    system = System(circuit, ground, gmin)
    steps = max(1, math.ceil(stop / step - 1e-9))
    times: FloatArray = np.arange(steps + 1, dtype=np.float64) * step

    if useInitialConditions:
        Gic, bic = system.initialConditionSystem()
        x0 = system.operatingPoint(Gic, bic, options)
    else:
        x0 = system.operatingPoint(system.G, system.rhsAt(0.0), options)

    if system.nonlinear:
        x = _transientNonlinear(system, x0, times, step, trapezoidal, options)
    else:
        x = _transientLinear(system, x0, times, step, trapezoidal)

    if not np.all(np.isfinite(x)):
        raise CircuitError(_SINGULAR)

    return TransientResult(times, *system.unpack(
        x, lambda s: np.array([s.at(float(t)) for t in times]), system.exactCurrents,
    ))