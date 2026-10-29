class CircuitError(ValueError):
    """The circuit is malformed, or its equations have no unique solution."""

class ConvergenceError(CircuitError):
    """Newton iteration failed to converge (or hit a singular matrix on the way)."""

class SpiceError(ValueError):
    """The netlist is malformed or uses something this reader does not support."""
 
    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line or None
        self.reason = message

class SimulationError(ValueError):
    """The request cannot be carried out for a reason that is not a netlist syntax error.

    Examples: a ``.dc`` sweep names a source that does not exist, a transient would need an absurd
    number of points, or the circuit has no ground node.
    """


class ProbeError(ValueError):
    """A probe expression is malformed or refers to something that does not exist."""