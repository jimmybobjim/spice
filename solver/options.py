from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class NewtonOptions:
    """Controls for circuits that contain nonlinear devices."""
    maxIterations: int = 100
    reltol: float = 1e-6         # relative tolerance on every unknown
    vntol: float = 1e-9          # absolute tolerance on node voltages, volts
    abstol: float = 1e-12        # absolute tolerance on branch currents, amps
    maxStepHalvings: int = 12    # transient: how many times a failing step may be halved
