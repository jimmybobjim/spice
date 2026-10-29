import math

from errors import CircuitError

def requirePositive(edge: object, what: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise CircuitError(f"{edge!r}: {what} must be positive and finite, got {value!r}")

def requireNonNegative(edge: object, what: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise CircuitError(f"{edge!r}: {what} must be positive or zero and finite, got {value!r}")

def requireFinite(edge: object, what: str, value: float) -> None:
    if not math.isfinite(value):
        raise CircuitError(f"{edge!r}: {what} must be finite, got {value!r}")

def requireValues(edge: object, what: str, value: object, *values: object) -> None:
    if value not in values:
        raise CircuitError(f"{edge!r}: {what} must be in {values!r}, got {value!r}")
