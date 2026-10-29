"""Circuit devices: how each is written in a netlist (`parse`), validated, and stamped into the MNA equations.

Every device is an immutable dataclass holding its name, the netlist line it came from, the names of the
nodes it touches (in the order of its ``terminals`` enum) and its own parameters. A netlist line is turned
into a device by the matching class's `parse`; `COMPONENT_TYPES` maps the leading letter of a name to the class.
"""
from __future__ import annotations

import cmath
import math
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from circuit.circuitMath import criticalVoltage, exp, pnjlim, squareLaw
from circuit.modelcard import ModelCard
from circuit.source import Source
from circuit.terminals import BjtTerminals, BranchTerminals, DiodeTerminals, MosfetTerminals, Terminals
from errors import CircuitError, SpiceError
from solver.nonlinearDeviceModels import Nonlinear
from solver.waveformHelpers import FloatArray, Waveform, exponential, pulse, pwl, sine
from util.spicetext import canonicalNode, isValue, parseValue
from util.validation import requireFinite, requireNonNegative, requirePositive, requireValues

_FET_STEP = 2.0  # largest change in a MOSFET control voltage per Newton iteration, volts

def _add(matrix: FloatArray, i: int | None, j: int | None, value: float) -> None:
    """matrix[i, j] += value, where None means the ground row/column (which is dropped)."""
    if i is not None and j is not None:
        matrix[i, j] += value


# ---- parsing helpers ---------------------------------------------------------

@dataclass(slots=True)
class ParseContext:
    """What a device needs to know about the rest of the netlist while it is being parsed."""
    models: Mapping[str, ModelCard]
    warnings: list[str] = field(default_factory=list[str])
    _checked: set[str] = field(default_factory=set[str])

    def warn(self, line: int, message: str) -> None:
        self.warnings.append(f"line {line}: {message}")

    def model(self, device: str, line: int, modelName: str, kinds: tuple[str, ...]) -> ModelCard:
        card = self.models.get(modelName)
        if card is None:
            raise SpiceError(f"{device}: model {modelName!r} is not defined", line)
        if card.kind not in kinds:
            raise SpiceError(f"{device}: model {modelName!r} is a {card.kind} model, expected {' or '.join(kinds)}", line)
        return card

    def checkModel(self, card: ModelCard, supported: frozenset[str], ignored: frozenset[str] = frozenset()) -> None:
        """Warn (once per card) about non-zero model parameters this simulator does not model."""
        if card.name in self._checked:
            return
        self._checked.add(card.name)
        for key, value in card.params.items():
            if key not in supported and key not in ignored and value != 0.0:
                self.warn(card.line, f".model {card.name}: parameter {key!r} is not modelled and was ignored")

def _splitNodes(name: str, line: int, positional: list[str], count: int) -> tuple[tuple[str, ...], list[str]]:
    if len(positional) < count:
        raise SpiceError(f"{name}: expected {count} nodes, got {len(positional)}", line)
    return tuple(canonicalNode(token) for token in positional[:count]), positional[count:]

def _number(text: str, line: int, what: str) -> float:
    try:
        return parseValue(text)
    except ValueError:
        raise SpiceError(f"{what} {text!r} is not a number", line) from None

def _only(name: str, line: int, parameters: dict[str, str], allowed: set[str]) -> None:
    for key in parameters:
        if key not in allowed:
            raise SpiceError(f"{name}: unsupported parameter {key!r}", line)

def _single(name: str, line: int, rest: list[str], what: str) -> str:
    if len(rest) != 1:
        raise SpiceError(f"{name}: expected exactly one {what}, got {len(rest)} values", line)
    return rest[0]


# ---- independent source specification ---------------------------------------

_SOURCE_FUNCTIONS = ("sin", "pulse", "pwl", "exp")
_SOURCE_UNSUPPORTED = ("sffm", "am", "trnoise", "trrandom")

def _waveform(name: str, line: int, kind: str, args: list[float]) -> Waveform:
    def need(low: int, high: int, usage: str) -> None:
        if not low <= len(args) <= high:
            raise SpiceError(f"{name}: {kind.upper()} needs {usage}", line)

    try:
        match kind:
            case "sin":
                need(3, 6, "vo va freq [td [theta [phase]]]")
                vo, va, freq, td, theta, phase = (*args, *(0.0,) * (6 - len(args)))
                if freq <= 0.0 or td < 0.0:
                    raise SpiceError(f"{name}: SIN needs a positive frequency and a non-negative delay", line)
                return sine(va, freq, offset=vo, phase=math.radians(phase), delay=td, damping=theta)
            case "pulse":
                need(2, 7, "v1 v2 [td [tr [tf [pw [per]]]]]")
                v1, v2, td, tr, tf = (*args[:5], *(0.0,) * (5 - len(args[:5])))
                pw = args[5] if len(args) > 5 else math.inf   # SPICE defaults this to TSTOP; "forever" is the closest equivalent
                per = args[6] if len(args) > 6 and args[6] > 0.0 else None
                if min(td, tr, tf, pw) < 0.0:
                    raise SpiceError(f"{name}: PULSE times must not be negative", line)
                return pulse(v1, v2, width=pw, delay=td, rise=tr, fall=tf, period=per)
            case "pwl":
                if len(args) < 2 or len(args) % 2:
                    raise SpiceError(f"{name}: PWL needs time/value pairs", line)
                return pwl(list(zip(args[0::2], args[1::2])))
            case "exp":
                need(6, 6, "v1 v2 td1 tau1 td2 tau2")
                v1, v2, td1, tau1, td2, tau2 = args
                return exponential(v1, v2, riseDelay=td1, riseTau=tau1, fallDelay=td2, fallTau=tau2)
    except SpiceError:
        raise
    except ValueError as error:   # from the waveform constructors, e.g. PWL times out of order
        raise SpiceError(f"{name}: {error}", line) from None
    raise SpiceError(f"{name}: unsupported source function {kind!r}", line)

def parseSource(name: str, line: int, tokens: list[str]) -> Source:
    """Parse ``[dc] value [ac mag [phase]] [sin(...) | pulse(...) | pwl(...) | exp(...)]`` (lower-cased tokens)."""
    dc: float | None = None
    ac: complex = 0.0
    waveform: Waveform | None = None

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "dc":
            if dc is not None or i + 1 >= len(tokens):
                raise SpiceError(f"{name}: DC needs exactly one value", line)
            dc = _number(tokens[i + 1], line, "DC value")
            i += 2
        elif token == "ac":
            if i + 1 >= len(tokens) or not isValue(tokens[i + 1]):
                raise SpiceError(f"{name}: AC needs a magnitude", line)
            magnitude = _number(tokens[i + 1], line, "AC magnitude")
            i += 2
            phase = 0.0
            if i < len(tokens) and isValue(tokens[i]):
                phase = _number(tokens[i], line, "AC phase")
                i += 1
            ac = cmath.rect(magnitude, math.radians(phase))
        elif token in _SOURCE_FUNCTIONS:
            if waveform is not None:
                raise SpiceError(f"{name}: only one time-dependent function is allowed", line)
            i += 1
            args: list[float] = []
            while i < len(tokens) and isValue(tokens[i]):
                args.append(_number(tokens[i], line, f"{token.upper()} argument"))
                i += 1
            waveform = _waveform(name, line, token, args)
        elif token in _SOURCE_UNSUPPORTED:
            raise SpiceError(f"{name}: {token.upper()} sources are not supported", line)
        elif i == 0 and isValue(token):
            dc = _number(token, line, "DC value")
            i += 1
        else:
            raise SpiceError(f"{name}: unexpected {token!r} in source specification", line)

    if dc is None:
        dc = waveform(0.0) if waveform is not None else 0.0   # as SPICE: a time-dependent source's DC value is its value at t = 0
    return Source(dc, ac, waveform)


# ---- devices ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Branch[E: Hashable]:
    """A linear device placed in the equations."""
    edge: E
    device: Component
    row: int | None  # row/column of this device's branch current, if it has one
    p: int | None    # index of the positive node, None if it is ground
    n: int | None    # index of the negative node, None if it is ground

@dataclass(frozen=True, slots=True)
class Component(ABC):
    name: str
    line: int
    nodes: tuple[str, ...]  # one per entry of `terminals`, same order

    letter: ClassVar[str]
    terminals: ClassVar[type[Terminals]]
    branchCurrent: ClassVar[bool] = False  # whether the device adds a branch-current unknown

    @classmethod
    @abstractmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> Component:
        """Build the device from a netlist line already split into tokens.

        `positional` holds the tokens without ``=`` (nodes first), `parameters` the ``key=value`` ones.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, edge: object) -> None:
        """Check the device's parameters, raising `CircuitError` if they cannot be simulated."""
        raise NotImplementedError

    def build[E: Hashable](self, edge: E, nodes: tuple[int | None, ...], row: int | None, gmin: float) -> Branch[E] | Nonlinear[E]:
        """Place the device in the equations. `nodes` are unknown indices in terminal order (None = ground)
        and `row` is the unknown reserved for its branch current (only given if `branchCurrent`)."""
        return Branch(edge, self, row, nodes[0], nodes[1])

    def stamp[E: Hashable](self, branch: Branch[E], G: FloatArray, C: FloatArray, contributions: list[tuple[int, float, Source]]) -> None:
        """Write one linear device's equations into G, C and the right-hand-side description.

        "Stamping" means each device adds a small fixed pattern to the matrices, so any circuit is
        just the sum of its devices' stamps. Nodes on ground (None) are skipped by `_add`.
        """
        raise CircuitError(f"{branch.edge!r}: unsupported device {branch.device}")

    def handleInitialConditions[E: Hashable](self, branch: Branch[E], G: FloatArray, b: FloatArray) -> None:
        pass

@dataclass(frozen=True, slots=True)
class Resistor(Component):
    resistance: float   # ohms

    letter = "R"
    terminals = BranchTerminals

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> Resistor:
        nodes, rest = _splitNodes(name, line, positional, 2)
        _only(name, line, parameters, set())
        return cls(name, line, nodes, _number(_single(name, line, rest, "resistance"), line, "resistance"))

    def validate(self, edge: object) -> None:
        requirePositive(edge, "resistance", self.resistance)

    def stamp[E: Hashable](self, branch: Branch[E], G: FloatArray, C: FloatArray, contributions: list[tuple[int, float, Source]]) -> None:
        p, n = branch.p, branch.n
        g: float = 1.0 / self.resistance
        _add(G, p, p, g)
        _add(G, n, n, g)
        _add(G, p, n, -g)
        _add(G, n, p, -g)

@dataclass(frozen=True, slots=True)
class Capacitor(Component):
    capacitance: float  # farads
    initialVoltage: float = 0.0   # volts across p/n

    letter = "C"
    terminals = BranchTerminals
    branchCurrent = True

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> Capacitor:
        nodes, rest = _splitNodes(name, line, positional, 2)
        _only(name, line, parameters, {"ic"})
        ic = _number(parameters["ic"], line, "initial voltage") if "ic" in parameters else 0.0
        return cls(name, line, nodes, _number(_single(name, line, rest, "capacitance"), line, "capacitance"), ic)

    def validate(self, edge: object) -> None:
        requirePositive(edge, "capacitance", self.capacitance)
        requireFinite(edge, "initial voltage", self.initialVoltage)

    def stamp[E: Hashable](self, branch: Branch[E], G: FloatArray, C: FloatArray, contributions: list[tuple[int, float, Source]]) -> None:
        p, n, row = branch.p, branch.n, branch.row
        assert row is not None
        _add(G, p, row, 1.0)
        _add(G, n, row, -1.0)
        G[row, row] += 1.0      # i - C d(v(p) - v(n))/dt = 0
        _add(C, row, p, -self.capacitance)
        _add(C, row, n, +self.capacitance)

    def handleInitialConditions[E: Hashable](self, branch: Branch[E], G: FloatArray, b: FloatArray) -> None:
        assert branch.row is not None
        G[branch.row, :] = 0.0
        _add(G, branch.row, branch.p, 1.0)
        _add(G, branch.row, branch.n, -1.0)
        b[branch.row] = self.initialVoltage

@dataclass(frozen=True, slots=True)
class Inductor(Component):
    inductance: float   # henries
    initialCurrent: float = 0.0   # amps across p/n

    letter = "L"
    terminals = BranchTerminals
    branchCurrent = True

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> Inductor:
        nodes, rest = _splitNodes(name, line, positional, 2)
        _only(name, line, parameters, {"ic"})
        ic = _number(parameters["ic"], line, "initial current") if "ic" in parameters else 0.0
        return cls(name, line, nodes, _number(_single(name, line, rest, "inductance"), line, "inductance"), ic)

    def validate(self, edge: object) -> None:
        requirePositive(edge, "inductance", self.inductance)
        requireFinite(edge, "initial current", self.initialCurrent)

    def stamp[E: Hashable](self, branch: Branch[E], G: FloatArray, C: FloatArray, contributions: list[tuple[int, float, Source]]) -> None:
        p, n, row = branch.p, branch.n, branch.row
        assert row is not None
        _add(G, p, row, 1.0)
        _add(G, n, row, -1.0)
        _add(G, row, p, 1.0)    # v(p) - v(n) - L di/dt = 0
        _add(G, row, n, -1.0)
        C[row, row] -= self.inductance

    def handleInitialConditions[E: Hashable](self, branch: Branch[E], G: FloatArray, b: FloatArray) -> None:
        assert branch.row is not None
        G[branch.row, :] = 0.0
        G[branch.row, branch.row] = 1.0
        b[branch.row] = self.initialCurrent

@dataclass(frozen=True, slots=True)
class VoltageSource(Component):
    source: Source

    letter = "V"
    terminals = BranchTerminals
    branchCurrent = True

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> VoltageSource:
        nodes, rest = _splitNodes(name, line, positional, 2)
        _only(name, line, parameters, set())
        return cls(name, line, nodes, parseSource(name, line, rest))

    def validate(self, edge: object) -> None:
        requireFinite(edge, "dc value", self.source.dc)

    def stamp[E: Hashable](self, branch: Branch[E], G: FloatArray, C: FloatArray, contributions: list[tuple[int, float, Source]]) -> None:
        p, n, row = branch.p, branch.n, branch.row
        assert row is not None
        _add(G, p, row, 1.0)    # branch current leaves p ...
        _add(G, n, row, -1.0)   # ... and enters n
        _add(G, row, p, 1.0)    # v(p) - v(n) = V
        _add(G, row, n, -1.0)
        contributions.append((row, 1.0, self.source))

@dataclass(frozen=True, slots=True)
class CurrentSource(Component):
    source: Source

    letter = "I"
    terminals = BranchTerminals

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> CurrentSource:
        nodes, rest = _splitNodes(name, line, positional, 2)
        _only(name, line, parameters, set())
        return cls(name, line, nodes, parseSource(name, line, rest))

    def validate(self, edge: object) -> None:
        requireFinite(edge, "dc value", self.source.dc)

    def stamp[E: Hashable](self, branch: Branch[E], G: FloatArray, C: FloatArray, contributions: list[tuple[int, float, Source]]) -> None:
        # the source pushes `value` from p through itself to n: it draws it out of node p and delivers it into node n
        if branch.p is not None:
            contributions.append((branch.p, -1.0, self.source))
        if branch.n is not None:
            contributions.append((branch.n, +1.0, self.source))

@dataclass(frozen=True, slots=True)
class Diode(Component):
    """Shockley diode: I = Is (exp(V / (n Vt)) - 1), anode -> cathode.

    If `breakdownVoltage` is set the diode also conducts in reverse, reaching
    `breakdownCurrent` when the reverse voltage equals `breakdownVoltage` (a Zener).
    """
    saturationCurrent: float = 1e-14  # Is, amps
    emissionCoefficient: float = 1.0  # n
    thermalVoltage: float = 0.02585   # Vt = kT/q, volts (about 300 K)
    breakdownVoltage: float | None = None   # positive reverse voltage, volts
    breakdownCurrent: float = 1e-3          # amps at the breakdown voltage

    letter = "D"
    terminals = DiodeTerminals

    _MODEL: ClassVar[frozenset[str]] = frozenset({"is", "n", "bv", "ibv"})
    _IGNORED: ClassVar[frozenset[str]] = frozenset({"level"})

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> Diode:
        nodes, rest = _splitNodes(name, line, positional, 2)
        _only(name, line, parameters, set())
        if len(rest) > 1:
            raise SpiceError(f"{name}: unexpected {rest[1]!r} (the area factor is not supported)", line)
        if not rest:
            return cls(name, line, nodes)
        card = context.model(name, line, rest[0], ("d",))
        context.checkModel(card, cls._MODEL, cls._IGNORED)
        p = card.params
        return cls(
            name, line, nodes,
            saturationCurrent=p.get("is", 1e-14), emissionCoefficient=p.get("n", 1.0),
            breakdownVoltage=p.get("bv"), breakdownCurrent=p.get("ibv", 1e-3),
        )

    def validate(self, edge: object) -> None:
        requirePositive(edge, "saturation current", self.saturationCurrent)
        requirePositive(edge, "emission coefficient", self.emissionCoefficient)
        requirePositive(edge, "thermal voltage", self.thermalVoltage)
        if self.breakdownVoltage is not None:
            requirePositive(edge, "breakdown voltage", self.breakdownVoltage)
        requirePositive(edge, "breakdown current", self.breakdownCurrent)

    def build[E: Hashable](self, edge: E, nodes: tuple[int | None, ...], row: int | None, gmin: float) -> Nonlinear[E]:
        vt = self.emissionCoefficient * self.thermalVoltage
        isat, bv, ibv = self.saturationCurrent, self.breakdownVoltage, self.breakdownCurrent
        vcrit = criticalVoltage(vt, isat)
        vcritBd = criticalVoltage(vt, ibv)

        def model(c: FloatArray) -> tuple[FloatArray, FloatArray]:
            v = float(c[0])
            e, de = exp(v / vt)
            i = isat * (e - 1.0) + gmin * v
            g = isat / vt * de + gmin
            if bv is not None:
                eb, deb = exp(-(v + bv) / vt)
                i -= ibv * eb
                g += ibv / vt * deb
            return np.array([i, -i]), np.array([[g], [-g]])

        def limit(new: FloatArray, old: FloatArray) -> FloatArray:
            v = pnjlim(float(new[0]), float(old[0]), vt, vcrit)
            if bv is not None:  # same limiter, mirrored, for the reverse-breakdown exponential
                arg = -(v + bv)
                mirrored = pnjlim(arg, -(float(old[0]) + bv), vt, vcritBd)
                if mirrored != arg:   # only touch v if limiting happened: rewriting it as -(-(v + bv)) - bv
                    v = -mirrored - bv    # can differ by an ulp, which Newton would then read as "still limiting" forever
            return np.array([v])

        return Nonlinear(edge, self.terminals, nodes, 1.0, np.array([[1.0, -1.0]]), model, limit)

@dataclass(frozen=True, slots=True)
class Bjt(Component):
    """Transport (Ebers-Moll) bipolar transistor. Terminals: collector, base, emitter.

    `earlyVoltage` (positive) adds output conductance; None means an ideal current source.
    """
    polarity: Literal["npn", "pnp"] = "npn"
    saturationCurrent: float = 1e-14  # Is, amps
    betaForward: float = 100.0
    betaReverse: float = 1.0
    earlyVoltage: float | None = None
    thermalVoltage: float = 0.02585

    letter = "Q"
    terminals = BjtTerminals

    _MODEL: ClassVar[frozenset[str]] = frozenset({"is", "bf", "br", "vaf", "va"})
    _IGNORED: ClassVar[frozenset[str]] = frozenset({"level"})

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> Bjt:
        nodes, rest = _splitNodes(name, line, positional, 3)
        _only(name, line, parameters, set())
        if not rest:
            raise SpiceError(f"{name}: a transistor needs a model name", line)
        if len(rest) > 1:
            raise SpiceError(f"{name}: unexpected {rest[1]!r} (substrate nodes and the area factor are not supported)", line)
        card = context.model(name, line, rest[0], ("npn", "pnp"))
        context.checkModel(card, cls._MODEL, cls._IGNORED)
        p = card.params
        return cls(
            name, line, nodes, polarity="npn" if card.kind == "npn" else "pnp",
            saturationCurrent=p.get("is", 1e-16), betaForward=p.get("bf", 100.0), betaReverse=p.get("br", 1.0),
            earlyVoltage=p.get("vaf", p.get("va")) or None,   # 0 (SPICE's default) means "no Early effect"
        )

    def validate(self, edge: object) -> None:
        requireValues(edge, "polarity", self.polarity, "npn", "pnp")
        requirePositive(edge, "saturation current", self.saturationCurrent)
        requirePositive(edge, "beta forward", self.betaForward)
        requirePositive(edge, "beta reverse", self.betaReverse)
        requirePositive(edge, "thermal voltage", self.thermalVoltage)
        if self.earlyVoltage is not None:
            requirePositive(edge, "early voltage", self.earlyVoltage)

    def build[E: Hashable](self, edge: E, nodes: tuple[int | None, ...], row: int | None, gmin: float) -> Nonlinear[E]:
        vt, isat, bf, br, va = self.thermalVoltage, self.saturationCurrent, self.betaForward, self.betaReverse, self.earlyVoltage
        vcrit = criticalVoltage(vt, isat)

        def model(c: FloatArray) -> tuple[FloatArray, FloatArray]:
            vbe, vbc = float(c[0]), float(c[1])
            ef, def_ = exp(vbe / vt)
            er, der = exp(vbc / vt)
            iF, iR = isat * (ef - 1.0), isat * (er - 1.0)
            gF, gR = isat / vt * def_, isat / vt * der

            qb, dqb = 1.0, 0.0  # Early-effect factor and its derivative with respect to vbc
            if va is not None:
                raw = 1.0 - vbc / va
                qb, dqb = (raw, -1.0 / va) if raw > 1e-3 else (1e-3, 0.0)

            ict = (iF - iR) * qb  # transport current, collector -> emitter
            ic = ict - iR / br
            ib = iF / bf + iR / br
            dic = (gF * qb, -gR * qb + (iF - iR) * dqb - gR / br)
            dib = (gF / bf, gR / br)

            # gmin shunts across base-emitter (b -> e) and base-collector (b -> c)
            ic -= gmin * vbc
            ib += gmin * (vbe + vbc)
            dic = (dic[0], dic[1] - gmin)
            dib = (dib[0] + gmin, dib[1] + gmin)

            currents = np.array([ic, ib, -(ic + ib)])
            jacobian = np.array([
                [dic[0], dic[1]],
                [dib[0], dib[1]],
                [-(dic[0] + dib[0]), -(dic[1] + dib[1])],
            ])
            return currents, jacobian

        def limit(new: FloatArray, old: FloatArray) -> FloatArray:
            return np.array([
                pnjlim(float(new[0]), float(old[0]), vt, vcrit),
                pnjlim(float(new[1]), float(old[1]), vt, vcrit),
            ])

        # controls: vbe = Vb - Ve, vbc = Vb - Vc; terminals ordered (c, b, e)
        T = np.array([[0.0, 1.0, -1.0], [-1.0, 1.0, 0.0]])
        return Nonlinear(edge, self.terminals, nodes, 1.0 if self.polarity == "npn" else -1.0, T, model, limit)

@dataclass(frozen=True, slots=True)
class Mosfet(Component):
    """Level-1 (square law) MOSFET. Terminals: drain, gate, source (the bulk node of an ``M`` line is read and ignored).

    threshold                    normalised threshold: conduction needs Vgs > threshold for
                                 an NMOS and Vsg > threshold for a PMOS (so an enhancement
                                 device has a positive value for both)
    k                            Kp * W / L, in A/V^2: Id = k/2 (Vgs - Vth)^2 in saturation
    channelLengthModulation      lambda, 1/V (0 for an ideal current source in saturation)
    The device is symmetric: drain and source swap automatically if Vds reverses.
    """
    polarity: Literal["nmos", "pmos"] = "nmos"
    threshold: float = 1.0
    k: float = 2e-3
    channelLengthModulation: float = 0.0

    letter = "M"
    terminals = MosfetTerminals

    _MODEL: ClassVar[frozenset[str]] = frozenset({"vto", "kp", "lambda", "level"})
    _INSTANCE: ClassVar[frozenset[str]] = frozenset({"w", "l", "m", "ad", "as", "pd", "ps", "nrd", "nrs"})
    _DEFAULT_SIZE: ClassVar[float] = 100e-6   # SPICE's default W and L, metres

    @classmethod
    def parse(cls, name: str, line: int, positional: list[str], parameters: dict[str, str], context: ParseContext) -> Mosfet:
        allNodes, rest = _splitNodes(name, line, positional, 4)   # d g s b
        _only(name, line, parameters, set(cls._INSTANCE))
        modelName = _single(name, line, rest, "model name")
        card = context.model(name, line, modelName, ("nmos", "pmos"))
        context.checkModel(card, cls._MODEL)
        if card.params.get("level", 1.0) != 1.0:
            context.warn(card.line, f".model {card.name}: only level 1 is implemented, using level 1")

        def size(key: str) -> float:
            return _number(parameters[key], line, key) if key in parameters else cls._DEFAULT_SIZE
        multiplier = _number(parameters["m"], line, "m") if "m" in parameters else 1.0

        vto = card.params.get("vto", 0.0)
        # SPICE writes a PMOS threshold as a negative number; this class normalises it to be positive
        threshold = vto if card.kind == "nmos" else -vto
        length = size("l")
        k = card.params.get("kp", 2e-5) * multiplier * size("w") / length if length > 0.0 else 0.0
        return cls(
            name, line, allNodes[:3], polarity="nmos" if card.kind == "nmos" else "pmos",
            threshold=threshold, k=k, channelLengthModulation=card.params.get("lambda", 0.0),
        )

    def validate(self, edge: object) -> None:
        requireValues(edge, "polarity", self.polarity, "nmos", "pmos")
        requireFinite(edge, "threshold", self.threshold)
        requirePositive(edge, "k", self.k)
        requireNonNegative(edge, "channel length modulation", self.channelLengthModulation)

    def build[E: Hashable](self, edge: E, nodes: tuple[int | None, ...], row: int | None, gmin: float) -> Nonlinear[E]:
        vth, k, lam = self.threshold, self.k, self.channelLengthModulation

        def model(c: FloatArray) -> tuple[FloatArray, FloatArray]:
            vgs, vds = float(c[0]), float(c[1])
            if vds >= 0.0:
                i, gm, gds = squareLaw(vgs, vds, vth, k, lam)
            else:  # symmetric device: the channel runs the other way, source and drain swap
                i2, gm2, gds2 = squareLaw(vgs - vds, -vds, vth, k, lam)
                i, gm, gds = -i2, -gm2, gm2 + gds2
            i += gmin * vds
            gds += gmin
            currents = np.array([i, 0.0, -i])
            jacobian = np.array([[gm, gds], [0.0, 0.0], [-gm, -gds]])
            return currents, jacobian

        def limit(new: FloatArray, old: FloatArray) -> FloatArray:
            delta = new - old
            return np.where(np.abs(delta) <= _FET_STEP, new, old + np.sign(delta) * _FET_STEP)

        # controls: vgs = Vg - Vs, vds = Vd - Vs; terminals ordered (d, g, s)
        T = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]])
        return Nonlinear(edge, self.terminals, nodes, 1.0 if self.polarity == "nmos" else -1.0, T, model, limit)


# Which class reads which netlist letter. An explicit table rather than `Component.__subclasses__()`, because
# `@dataclass(slots=True)` creates a second class object and the stale one can show up in `__subclasses__()`.
COMPONENT_TYPES: dict[str, type[Component]] = {
    cls.letter.lower(): cls
    for cls in (Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, Diode, Bjt, Mosfet)
}
