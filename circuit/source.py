from dataclasses import dataclass

from solver.waveformHelpers import Waveform

@dataclass(frozen=True, slots=True)
class Source:
    """The value of an independent source.

    dc        used by solveDc (and by solveAc to find the operating point)
    ac        complex amplitude (magnitude and phase) used by solveAc
    waveform  if given, used by solveTransient instead of `dc`; maps time to value
    """
    dc: float = 0.0
    ac: complex = 0.0
    waveform: Waveform | None = None

    def at(self, time: float) -> float:
        return self.dc if self.waveform is None else self.waveform(time)
