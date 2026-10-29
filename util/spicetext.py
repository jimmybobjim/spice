"""Small pieces of SPICE text syntax shared by the parser and the component classes."""
import re
from decimal import Decimal

# Source: https://en.wikipedia.org/wiki/Metric_prefix, this is an incomplete list here but if larger/smaller
# values are needed then this is probably the wrong program to be doing your simulation
_METRIC_EXPONENTS: dict[str, int] = {"t": 12, "g": 9, "k": 3, "m": -3, "u": -6, "n": -9, "p": -12, "f": -15}

_NUMBER_REGEX = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)([a-z]*)$")

def parseValue(text: str) -> float:
    """Convert a SPICE number such as ``4.7u``, ``1meg``, ``10mil`` or ``2.2nF`` to a float."""
    match = _NUMBER_REGEX.match(text.strip().lower())
    if match is None:
        raise ValueError(f"not a number: {text!r}")
    number, suffix = match.groups()
    # Scale the decimal digits exactly and round to a float once: 10 * 1e-6 would give
    # 9.999999999999999e-06, whereas "10e-6" is exactly the float nearest to 1e-05.
    digits = Decimal(number)
    if suffix.startswith("meg"):
        return float(digits.scaleb(6))
    if suffix.startswith("mil"):
        return float(digits * Decimal("25.4e-6"))
    if suffix and suffix[0] in _METRIC_EXPONENTS:
        return float(digits.scaleb(_METRIC_EXPONENTS[suffix[0]]))
    return float(digits)

def isValue(text: str) -> bool:
    """Report whether `text` looks like a SPICE number (see `parseValue`)."""
    return _NUMBER_REGEX.match(text.strip().lower()) is not None

def canonicalNode(name: str) -> str:
    """Canonicalise a node name: lower case, with ``gnd`` meaning ground (``0``)."""
    name = name.lower()
    return "0" if name == "gnd" else name
