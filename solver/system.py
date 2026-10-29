import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from circuit.circuit import Circuit
from circuit.components import Branch, CurrentSource, Resistor
from circuit.source import Source
from circuit.terminals import BranchTerminals, Terminals
from errors import CircuitError, ConvergenceError
from solver.nonlinearDeviceModels import Nonlinear
from solver.options import NewtonOptions
from solver.waveformHelpers import ComplexArray, FloatArray

SINGULAR = (
    "the circuit equations have no unique solution. Typical causes: a loop made only of "
    "voltage sources and/or inductors, conflicting ideal sources, a net that is not "
    "connected to ground, or a cut-set made only of current sources"
)

def solve(matrix: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    try:
        x: NDArray[Any] = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise CircuitError(SINGULAR) from error
    if not np.all(np.isfinite(x)):
        raise CircuitError(SINGULAR)
    return x

@dataclass(slots=True)
class _Linearization:
    controls: list[FloatArray]       # the (limited) control voltages each device was linearised at
    limited: bool                    # whether limiting changed any of them
    jacobian: FloatArray             # d(i_nl)/dx
    constant: FloatArray             # the Norton equivalent currents: i_nl(x) ~ constant + jacobian x
    perDevice: list[FloatArray]      # each device's terminals x terminals Jacobian


class System[V: Hashable, E: Hashable]:
    """The assembled MNA equations: G x + C dx/dt + i_nl(x) = b."""

    def __init__(self, circuit: Circuit[V, E], ground: V, gmin: float) -> None:
        if not circuit.hasVertex(ground):
            raise CircuitError(f"ground vertex {ground!r} is not in the circuit")
        if not (math.isfinite(gmin) and gmin >= 0.0):
            raise CircuitError(f"gmin must be finite and non-negative, got {gmin!r}")

        self.ground = ground
        self.nodeIndex: dict[V, int] = {
            vertex: i for i, vertex in enumerate(v for v in circuit.vertices() if v != ground)
        }
        self.nodeCount = len(self.nodeIndex)

        size = self.nodeCount
        self.devices: list[Branch[E] | Nonlinear[E]] = []  # in circuit order
        self.branches: list[Branch[E]] = []
        self.nonlinear: list[Nonlinear[E]] = []
        for edge, device in circuit.edges():
            roles = device.terminals
            links = circuit.links(edge)
            if set(links) != set(roles):
                raise CircuitError(
                    f"{edge!r}: a {type(device).__name__} needs exactly the roles {list(roles)!r}, "
                    f"got {sorted(links, key=lambda link: link.value)!r}"
                )
            nodes = tuple(self.nodeIndex.get(links[role]) for role in roles)

            device.validate(edge)
            row: int | None = None
            if device.branchCurrent:
                row, size = size, size + 1   # the branch current is the next unknown
            item = device.build(edge, nodes, row, gmin)

            self.devices.append(item)
            if isinstance(item, Nonlinear):
                self.nonlinear.append(item)
            else:
                self.branches.append(item)

        self.size = size
        self.G: FloatArray = np.zeros((size, size))
        self.C: FloatArray = np.zeros((size, size))
        # (row, sign, source): b[row] += sign * value(source)
        self.contributions: list[tuple[int, float, Source]] = []

        for i in range(self.nodeCount):
            self.G[i, i] += gmin  # keeps floating nodes solvable

        for branch in self.branches:
            branch.device.stamp(branch, self.G, self.C, self.contributions)

    # ---- right-hand sides --------------------------------------------------

    def rhs(self, value: Callable[[Source], float]) -> FloatArray:
        b: FloatArray = np.zeros(self.size)
        for row, sign, source in self.contributions:
            b[row] += sign * value(source)
        return b

    def rhsAt(self, time: float) -> FloatArray:
        return self.rhs(lambda source: source.at(time))

    def rhsAc(self) -> ComplexArray:
        b: ComplexArray = np.zeros(self.size, dtype=np.complex128)
        for row, sign, source in self.contributions:
            b[row] += sign * source.ac
        return b

    def initialConditionSystem(self) -> tuple[FloatArray, FloatArray]:
        """A DC-style system with capacitor voltages and inductor currents pinned to their initial values."""
        G = self.G.copy()
        b = self.rhsAt(0.0)
        for branch in self.branches:
            branch.device.handleInitialConditions(branch, G, b)
        return G, b

    # ---- nonlinear devices -------------------------------------------------

    @staticmethod
    def _terminalVoltages(dev: Nonlinear[E], x: NDArray[Any]) -> FloatArray:
        return np.array([0.0 if node is None else x[node] for node in dev.nodes], dtype=np.float64)

    def controls(self, x: FloatArray) -> list[FloatArray]:
        return [dev.sign * (dev.T @ self._terminalVoltages(dev, x)) for dev in self.nonlinear]

    def linearize(self, x: FloatArray, previous: list[FloatArray] | None) -> _Linearization:
        """Replace every nonlinear device by its tangent (conductances + equivalent current) at x.

        With `previous` given, each device's control voltages are limited relative to it.
        """
        jacobian: FloatArray = np.zeros((self.size, self.size))
        constant: FloatArray = np.zeros(self.size)
        used: list[FloatArray] = []
        perDevice: list[FloatArray] = []
        limited = False

        for k, dev in enumerate(self.nonlinear):
            raw = dev.sign * (dev.T @ self._terminalVoltages(dev, x))
            c = raw if previous is None else dev.limit(raw, previous[k])
            if previous is not None and bool(np.any(c != raw)):
                limited = True
            currents, jc = dev.model(c)
            jn = jc @ dev.T                          # d(terminal currents)/d(terminal voltages)
            offset = dev.sign * (currents - jc @ c)  # what is left once the Jacobian term is removed
            for a, na in enumerate(dev.nodes):
                if na is None:
                    continue
                constant[na] += offset[a]
                for b, nb in enumerate(dev.nodes):
                    if nb is not None:
                        jacobian[na, nb] += jn[a, b]
            used.append(c)
            perDevice.append(jn)
        return _Linearization(used, limited, jacobian, constant, perDevice)

    def nonlinearCurrents(self, x: FloatArray) -> FloatArray:
        """The nonlinear devices' contribution to the KCL rows, evaluated exactly at x."""
        vec: FloatArray = np.zeros(self.size)
        for dev in self.nonlinear:
            c = dev.sign * (dev.T @ self._terminalVoltages(dev, x))
            currents, _ = dev.model(c)
            for node, value in zip(dev.nodes, dev.sign * currents):
                if node is not None:
                    vec[node] += value
        return vec

    def exactCurrents(self, dev: Nonlinear[E], X: NDArray[Any]) -> NDArray[Any]:
        """Terminal currents (terminals x points) of one device for each solution row of X."""
        out: FloatArray = np.empty((len(dev.nodes), X.shape[0]))
        for k in range(X.shape[0]):
            c = dev.sign * (dev.T @ self._terminalVoltages(dev, X[k]))
            currents, _ = dev.model(c)
            out[:, k] = dev.sign * currents
        return out

    def newton(self, A: FloatArray, r: FloatArray, start: FloatArray, options: NewtonOptions) -> FloatArray:
        """Solve A x + i_nl(x) = r by Newton iteration from `start`."""
        tolerance: FloatArray = np.full(self.size, options.abstol)
        tolerance[:self.nodeCount] = options.vntol

        x = start.copy()
        previous = self.controls(x)
        for _ in range(options.maxIterations):
            lin = self.linearize(x, previous)
            try:
                xNew: FloatArray = np.linalg.solve(A + lin.jacobian, r - lin.constant)
            except np.linalg.LinAlgError as error:
                raise ConvergenceError("singular matrix during Newton iteration") from error
            if not np.all(np.isfinite(xNew)):
                raise ConvergenceError("Newton iteration diverged")
            step = np.abs(xNew - x)
            converged = (not lin.limited) and bool(
                np.all(step <= options.reltol * np.maximum(np.abs(xNew), np.abs(x)) + tolerance)
            )
            x, previous = xNew, lin.controls
            if converged:
                return x
        raise ConvergenceError(f"Newton iteration did not converge in {options.maxIterations} iterations")

    def operatingPoint(self, G: FloatArray, b: FloatArray, options: NewtonOptions) -> FloatArray:
        """Solve the static equations G x + i_nl(x) = b, with fallbacks if plain Newton fails."""
        if not self.nonlinear:
            return solve(G, b)

        zero: FloatArray = np.zeros(self.size)
        try:
            return self.newton(G, b, zero, options)
        except ConvergenceError:
            pass

        # gmin stepping: start heavily damped (every node strongly tied to ground), then relax
        try:
            x = zero
            for exponent in range(2, 13):
                extra: FloatArray = np.zeros(self.size)
                extra[:self.nodeCount] = 10.0 ** -exponent
                x = self.newton(G + np.diag(extra), b, x, options)
            return self.newton(G, b, x, options)
        except ConvergenceError:
            pass

        # source stepping: ramp every source up from 10% to 100%
        try:
            x = zero
            for alpha in np.linspace(0.1, 1.0, 10):
                x = self.newton(G, float(alpha) * b, x, options)
            return x
        except ConvergenceError as error:
            raise ConvergenceError(
                "could not find the DC operating point: Newton iteration, gmin stepping and "
                f"source stepping all failed ({error}); check the circuit for a missing bias path "
                "or a device with a poor model"
            ) from error

    # ---- results -----------------------------------------------------------

    def unpack(
        self, X: NDArray[Any], sourceValue: Callable[[Source], NDArray[Any]],
        nonlinearCurrents: Callable[[Nonlinear[E], NDArray[Any]], NDArray[Any]],
    ) -> tuple[dict[V, NDArray[Any]], dict[E, NDArray[Any]], dict[E, dict[Terminals, NDArray[Any]]]]:
        """Turn solution rows (shape: points x size) into voltages, two-terminal currents and terminal currents."""
        zero = np.zeros(X.shape[0], dtype=X.dtype)

        def voltage(index: int | None) -> NDArray[Any]:
            return zero if index is None else np.array(X[:, index])

        voltages: dict[V, NDArray[Any]] = {self.ground: zero}
        for vertex, index in self.nodeIndex.items():
            voltages[vertex] = np.array(X[:, index])

        currents: dict[E, NDArray[Any]] = {}
        terminals: dict[E, dict[Terminals, NDArray[Any]]] = {}
        for item in self.devices:
            if isinstance(item, Nonlinear):
                out = nonlinearCurrents(item, X)
                terminals[item.edge] = {role: np.array(out[k]) for k, role in enumerate(item.roles)}
                if len(item.roles) == 2:
                    currents[item.edge] = terminals[item.edge][next(iter(item.roles))]
                continue

            match item.device:
                case Resistor(resistance=r):
                    i = (voltage(item.p) - voltage(item.n)) / r
                case CurrentSource(source=s):
                    i = sourceValue(s)
                case _:
                    assert item.row is not None
                    i = np.array(X[:, item.row])
            currents[item.edge] = i
            terminals[item.edge] = {BranchTerminals.POSITIVE: i, BranchTerminals.NEGATIVE: -i}
        return voltages, currents, terminals