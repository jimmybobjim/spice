"""Read a SPICE netlist into a `Netlist`, and flatten its subcircuits.

    netlist = parseNetlist("amp.cir")          # or parseText(text)
    components = flatten(netlist)              # one flat list of primitive devices

Supported: R C L V I D Q M devices, ``X`` subcircuit instances, ``.model`` (d npn pnp nmos pmos),
``.subckt/.ends``, ``.op .dc .ac .tran``, ``.print/.plot/.probe``, ``.options``. Directives that would
change the answer if ignored are rejected; directives that only matter for output are accepted.
"""
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal

from circuit.components import COMPONENT_TYPES, Component, ParseContext
from circuit.modelcard import ModelCard
from errors import CircuitError, SpiceError
from util.spicetext import canonicalNode, isValue, parseValue

_INLINE_COMMENT_REGEX = re.compile(r";.*$|(?:^|\s)\$.*$")
_PROBE_REGEX = re.compile(r"\w+\([^()]*\)")

# Directives that would change the circuit if ignored, so they are rejected instead.
_UNSUPPORTED_DIRECTIVES = (
    ".param", ".include", ".inc", ".lib", ".func", ".global", ".measure", ".meas", ".step",
    ".noise", ".tf", ".sens", ".four", ".disto", ".pz", ".data", ".if"
)
# Directives that do not affect results (or are only hints), accepted and ignored.
_HARMLESS_DIRECTIVES = (".save", ".width")
# Directives that do affect results but cannot be honoured here, so they are ignored *with a warning*.
_WARN_DIRECTIVES = (".ic", ".nodeset", ".temp")

def _looksLikeDevice(title: str) -> bool:
    """Whether a title line looks like it was meant to be a component, e.g. ``r1 a b 1k`` or ``d1 a k dmod``."""
    tokens = title.lower().split()
    if len(tokens) < 4 or not re.fullmatch(r"[\w.]+", tokens[0]):
        return False
    if tokens[0][0] in "rclvi":   # name node node value: the fourth token is a number or a source keyword
        return isValue(tokens[3]) or tokens[3] in ("dc", "ac", "sin", "pulse", "pwl", "exp")
    return tokens[0][0] in "dqmx" and any(ch.isdigit() for ch in tokens[0])

def _value(text: str, line: int, quantity: str) -> float:
    """`parseValue` that reports failures as a `SpiceError` pointing at the line."""
    try:
        return parseValue(text)
    except ValueError:
        raise SpiceError(f"{quantity} {text!r} is not a number", line) from None


# ---- parsed pieces -------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """An ``X`` line: a use of a subcircuit, not yet expanded."""
    name: str
    line: int
    nodes: tuple[str, ...]
    subcircuit: str

type Element = Component | Instance

@dataclass(frozen=True)
class Subcircuit:
    name: str
    terminals: list[str]
    components: list[Element] = field(default_factory=list[Element])
    line: int = 0

@dataclass(frozen=True)
class OpAnalysis:
    """``.op``: DC operating point."""
    kind: ClassVar[str] = "op"
    line: int = 0

@dataclass(frozen=True)
class DcSweep:
    """``.dc source start stop step``: repeat the operating point while sweeping one source."""
    source: str
    start: float
    stop: float
    step: float
    line: int = 0
    kind: ClassVar[str] = "dc"

@dataclass(frozen=True)
class AcAnalysis:
    """``.ac dec|oct|lin points start stop``: small-signal frequency sweep."""
    variation: Literal["dec", "oct", "lin"]
    points: int
    start: float
    stop: float
    line: int = 0
    kind: ClassVar[str] = "ac"

@dataclass(frozen=True)
class TranAnalysis:
    """``.tran tstep tstop [tstart [tmax]] [uic]``: time-domain simulation."""
    step: float
    stop: float
    start: float = 0.0
    max_step: float | None = None
    use_initial_conditions: bool = False
    line: int = 0
    kind: ClassVar[str] = "tran"

type Analysis = OpAnalysis | DcSweep | AcAnalysis | TranAnalysis

@dataclass(frozen=True)
class PrintRequest:
    """A ``.print``, ``.plot`` or ``.probe`` line: which quantities the user wants to see."""
    analysis: str | None
    probes: tuple[str, ...]

@dataclass
class Netlist:
    """A parsed netlist."""
    title: str = ""
    components: list[Element] = field(default_factory=list[Element])
    models: dict[str, ModelCard] = field(default_factory=dict[str, ModelCard])
    subcircuits: dict[str, Subcircuit] = field(default_factory=dict[str, Subcircuit])
    analyses: list[Analysis] = field(default_factory=list[Analysis])
    prints: list[PrintRequest] = field(default_factory=list[PrintRequest])
    options: dict[str, float] = field(default_factory=dict[str, float])
    warnings: list[str] = field(default_factory=list[str])


# ---- reading lines ---------------------------------------------------------------

@dataclass(frozen=True)
class LogicalLine:
    """One statement after comments are removed and continuation lines are joined."""
    number: int   # 1-based number of the physical line it started on
    text: str

def readLines(text: str) -> Iterator[LogicalLine]:
    """Yield the title, then each statement with comments stripped and ``+`` continuations joined."""
    rawLines = text.splitlines()
    if not rawLines:
        return

    # first line of a SPICE file is always the title
    yield LogicalLine(1, rawLines[0].strip())

    pending: LogicalLine | None = None
    for number, raw in enumerate(rawLines[1:], start=2):
        line = _INLINE_COMMENT_REGEX.sub("", raw).strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            if pending is None:
                raise SpiceError("continuation line (+) has nothing to continue", number)
            pending = LogicalLine(pending.number, f"{pending.text} {line[1:].strip()}")
        else:
            if pending is not None:
                yield pending
            pending = LogicalLine(number, line)
    if pending is not None:
        yield pending

def _tokens(text: str) -> list[str]:
    """Lower-case tokens, with ``(``, ``)`` and ``,`` as whitespace and ``a = b`` tightened to ``a=b``."""
    return re.sub(r"[(),]", " ", re.sub(r"\s*=\s*", "=", text.strip().lower())).split()


# ---- elements ----------------------------------------------------------------------

def parseElement(line: LogicalLine, context: ParseContext) -> Element:
    """Parse one device or subcircuit-instance line."""
    name, *tokens = _tokens(line.text)

    positional: list[str] = []
    parameters: dict[str, str] = {}
    for token in tokens:
        if len(pair := token.split("=", 1)) == 2:
            parameters[pair[0]] = pair[1]
        else:
            positional.append(token)

    if name[0] == "x":
        if parameters or len(positional) < 1:
            raise SpiceError(f"{name}: expected 'node... subcircuit-name' (subcircuit parameters are not supported)", line.number)
        return Instance(name, line.number, tuple(canonicalNode(node) for node in positional[:-1]), positional[-1])

    kind = COMPONENT_TYPES.get(name[0])
    if kind is None:
        supported = ", ".join(letter.upper() for letter in (*COMPONENT_TYPES, "x"))
        raise SpiceError(f"Unknown or unsupported component: {name} (supported device letters: {supported})", line.number)

    component = kind.parse(name, line.number, positional, parameters, context)
    try:
        component.validate(name)
    except CircuitError as error:
        raise SpiceError(str(error), line.number) from None
    if name[0] == "m":
        context.warn(line.number, f"{name}: the bulk terminal is ignored (this MOSFET model has no body effect)")
    return component


# ---- directives --------------------------------------------------------------------

def parseDirective(netlist: Netlist, line: LogicalLine, word: str) -> None:
    """Handle one dot-directive other than ``.subckt``/``.ends``/``.end``, recording it on `netlist`."""
    tokens = _tokens(line.text)
    match word:
        case ".title":
            netlist.title = line.text.split(None, 1)[1] if len(line.text.split(None, 1)) > 1 else ""

        case ".model":
            if len(tokens) < 3:
                raise SpiceError(".model needs a name and a type", line.number)
            _, name, kind, *rest = tokens

            if kind not in ("d", "npn", "pnp", "nmos", "pmos"):
                raise SpiceError(f".model type {kind!r} is not supported (use d, npn, pnp, nmos or pmos)", line.number)

            parameters: dict[str, float] = {}
            for token in rest:
                if len(pair := token.split("=", 1)) != 2:
                    raise SpiceError(f".model {name}: expected name=value, got {token!r}", line.number)
                parameters[pair[0]] = _value(pair[1], line.number, f"model parameter {pair[0]}")
            netlist.models[name] = ModelCard(name, kind, parameters, line.number)  # type: ignore[arg-type]

        case ".op":
            netlist.analyses.append(OpAnalysis(line=line.number))

        case ".dc":
            if len(tokens) == 9:
                raise SpiceError("nested .dc sweeps (a second source) are not supported", line.number)
            if len(tokens) != 5:
                raise SpiceError(".dc source needs: source start stop step", line.number)
            start, stop, step = (_value(token, line.number, ".dc value") for token in tokens[2:])
            if step == 0 or (stop - start) / step < 0:
                raise SpiceError(".dc step must be non-zero and point from start towards stop", line.number)
            netlist.analyses.append(DcSweep(tokens[1], start, stop, step, line.number))

        case ".ac":
            if len(tokens) != 5 or tokens[1] not in ("dec", "oct", "lin"):
                raise SpiceError(".ac needs: dec|oct|lin points fstart fstop", line.number)
            variation = tokens[1]
            points = int(_value(tokens[2], line.number, ".ac point count"))
            fStart = _value(tokens[3], line.number, ".ac start")
            fStop = _value(tokens[4], line.number, ".ac stop")
            if points < 1 or fStart < 0 or fStop <= fStart or (variation != "lin" and fStart <= 0):
                raise SpiceError(".ac needs points >= 1 and 0 < fstart < fstop (fstart may be 0 only for lin)", line.number)
            netlist.analyses.append(AcAnalysis(variation, points, fStart, fStop, line.number))  # type: ignore[arg-type]

        case ".tran":
            uic = "uic" in tokens
            numbers = [_value(token, line.number, ".tran argument") for token in tokens[1:] if token != "uic"]
            if not 2 <= len(numbers) <= 4:
                raise SpiceError(".tran needs: tstep tstop [tstart [tmax]] [uic]", line.number)
            tStep, tStop = numbers[0:2]
            tStart = numbers[2] if len(numbers) > 2 else 0.0
            maxStep = numbers[3] if len(numbers) > 3 else None
            if tStep <= 0 or tStop <= 0 or tStart < 0 or (maxStep is not None and maxStep <= 0):
                raise SpiceError(".tran times must be positive", line.number)
            if tStart >= tStop:
                raise SpiceError(".tran tstart must be less than tstop", line.number)
            netlist.analyses.append(TranAnalysis(tStep, tStop, tStart, maxStep, uic, line.number))

        case ".print" | ".plot" | ".probe":
            analysis = tokens[1] if len(tokens) > 1 and tokens[1] in ("op", "dc", "ac", "tran") else None
            probes = tuple(re.sub(r"\s+", "", p) for p in _PROBE_REGEX.findall(line.text.lower()))
            if not probes:
                raise SpiceError(f"{word} needs at least one expression such as v(out)", line.number)
            netlist.prints.append(PrintRequest(analysis, probes))

        case ".options" | ".option":
            for token in tokens[1:]:
                if "=" not in token:
                    continue
                key, text = token.split("=", 1)
                if isValue(text):
                    netlist.options[key] = parseValue(text)
                else:
                    netlist.warnings.append(f"line {line.number}: option {key}={text} is not numeric and was ignored")

        case directive if directive in _WARN_DIRECTIVES:
            netlist.warnings.append(f"line {line.number}: {word} is not supported and was ignored, so results may differ from SPICE")

        case directive if directive in _HARMLESS_DIRECTIVES:
            pass

        case directive if directive in _UNSUPPORTED_DIRECTIVES:
            raise SpiceError(f"{word} is not supported", line.number)

        case _:
            raise SpiceError(f"unknown directive {word}", line.number)


# ---- whole netlists -------------------------------------------------------------------

def parseText(text: str) -> Netlist:
    """Parse netlist text. Device lines are read after every directive, so a ``.model`` may come after its users."""
    netlist = Netlist()
    lines = readLines(text)

    first = next(lines, None)
    if first is None:
        raise SpiceError("the netlist is empty")
    netlist.title = first.text
    if _looksLikeDevice(first.text):
        netlist.warnings.append(f"line 1: the first line of a netlist is its title, but {first.text!r} looks like a component, which was not read")

    deferred: list[tuple[LogicalLine, list[Element]]] = []   # device lines, and where their result goes
    current: Subcircuit | None = None
    for line in lines:
        word = line.text.split()[0].lower()

        match word:
            case ".end":
                if current is not None:
                    raise SpiceError(f"subcircuit {current.name!r} is missing .ends", current.line)
                break

            case ".subckt":
                if current is not None:
                    raise SpiceError("nested .subckt definitions are not supported", line.number)
                tokens = _tokens(line.text)
                if len(tokens) < 2 or any(token.endswith(":") or "=" in token for token in tokens[2:]):
                    raise SpiceError(".subckt needs a name and a list of nodes (parameters are not supported)", line.number)
                name, ports = tokens[1], tokens[2:]
                if name in netlist.subcircuits:
                    raise SpiceError(f"subcircuit {name!r} is defined twice", line.number)
                if "0" in ports or "gnd" in ports or len(set(ports)) != len(ports):
                    raise SpiceError(f".subckt {name}: ports must be distinct and cannot include ground", line.number)
                current = netlist.subcircuits[name] = Subcircuit(name, ports, line=line.number)

            case ".ends":
                if current is None:
                    raise SpiceError(".ends without a matching .subckt", line.number)
                current = None

            case directive if not directive.startswith("."):
                deferred.append((line, current.components if current else netlist.components))

            case directive if directive != ".model" and current is not None:
                raise SpiceError(f"{word} is not allowed inside a .subckt (only devices and .model)", line.number)
            
            case _:
                parseDirective(netlist, line, word)

    if current is not None:
        raise SpiceError(f"subcircuit {current.name!r} is missing .ends", current.line)

    context = ParseContext(netlist.models, netlist.warnings)
    for line, target in deferred:
        target.append(parseElement(line, context))
    return netlist

def parseNetlist(filepath: str) -> Netlist:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return parseText(f.read())

def flatten(netlist: Netlist, maxDepth: int = 16) -> list[Component]:
    """Expand every subcircuit instance, giving one flat list of primitive components. maxDepth guards
    against a subcircuit that instantiates itself.

    Inside an instance named ``x1``, a component ``r2`` becomes ``x1.r2``, the subcircuit's port nodes
    are replaced by the nodes the caller connected, its other nodes become private (``x1.mid``), and
    node ``0`` stays the global ground. Instances may be nested.
    """
    flat: list[Component] = []

    def expand(elements: list[Element], prefix: str, ports: dict[str, str], depth: int) -> None:
        def local(node: str) -> str:
            if node == "0":
                return node
            return ports.get(node, prefix + node)

        for element in elements:
            if isinstance(element, Component):
                flat.append(replace(element, name=prefix + element.name, nodes=tuple(local(node) for node in element.nodes)))
                continue

            if depth >= maxDepth:
                raise SpiceError(f"subcircuits nested more than {maxDepth} deep (does {element.subcircuit!r} instantiate itself?)", element.line)
            subcircuit = netlist.subcircuits.get(element.subcircuit)
            if subcircuit is None:
                raise SpiceError(f"{element.name}: subcircuit {element.subcircuit!r} is not defined", element.line)
            if len(element.nodes) != len(subcircuit.terminals):
                raise SpiceError(
                    f"{element.name}: subcircuit {subcircuit.name!r} has {len(subcircuit.terminals)} ports "
                    f"but {len(element.nodes)} nodes were given", element.line)
            connections = {port: local(node) for port, node in zip(subcircuit.terminals, element.nodes)}
            expand(subcircuit.components, f"{prefix}{element.name}.", connections, depth + 1)

    expand(netlist.components, "", {}, 0)
    return flat
