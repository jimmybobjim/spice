import bisect
import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]
type Waveform = Callable[[float], float]

def sine(amplitude: float, frequency: float, *, offset: float = 0.0,
         phase: float = 0.0, delay: float = 0.0, damping: float = 0.0) -> Waveform:
    """offset + amplitude * exp(-damping (t - delay)) * sin(2 pi f (t - delay) + phase); phase in radians."""
    def waveform(time: float) -> float:
        if time < delay:
            return offset
        t = time - delay
        return offset + amplitude * math.exp(-damping * t) * math.sin(2.0 * math.pi * frequency * t + phase)
    return waveform

def exponential(low: float, high: float, *, riseDelay: float, riseTau: float,
                fallDelay: float, fallTau: float) -> Waveform:
    """SPICE EXP: `low` until `riseDelay`, then an exponential approach to `high` (time constant
    `riseTau`); from `fallDelay` an exponential return towards `low` (time constant `fallTau`)."""
    if riseTau <= 0.0 or fallTau <= 0.0:
        raise ValueError("exp time constants must be positive")

    def waveform(time: float) -> float:
        value = low
        if time >= riseDelay:
            value += (high - low) * (1.0 - math.exp(-(time - riseDelay) / riseTau))
        if time >= fallDelay:
            value += (low - high) * (1.0 - math.exp(-(time - fallDelay) / fallTau))
        return value
    return waveform

def step(low: float, high: float, at: float = 0.0) -> Waveform:
    """`low` before time `at`, `high` from then on."""
    def waveform(time: float) -> float:
        return high if time >= at else low
    return waveform

def pulse(low: float, high: float, *, width: float, delay: float = 0.0,
          rise: float = 0.0, fall: float = 0.0, period: float | None = None) -> Waveform:
    """Trapezoidal pulse (SPICE PULSE): delay, rise, `width` at `high`, fall, repeat every `period`."""
    if period is not None and period < rise + width + fall:
        raise ValueError("period must be at least rise + width + fall")

    def waveform(time: float) -> float:
        if time < delay:
            return low
        t = time - delay
        if period is not None:
            t %= period
        if t < rise:
            return low + (high - low) * t / rise
        t -= rise
        if t < width:
            return high
        t -= width
        if t < fall:
            return high + (low - high) * t / fall
        return low
    return waveform

def logSweep(start: float, stop: float, pointsPerDecade: int = 10) -> FloatArray:
    """Logarithmically spaced frequencies from `start` to `stop` inclusive."""
    if not (0 < start < stop) or pointsPerDecade < 1:
        raise ValueError("need 0 < start < stop and pointsPerDecade >= 1")
    count = math.ceil(math.log10(stop / start) * pointsPerDecade) + 1
    return np.logspace(math.log10(start), math.log10(stop), count)

def octaveSweep(start: float, stop: float, pointsPerOctave: int = 10) -> FloatArray:
    """Logarithmically spaced frequencies, `pointsPerOctave` per doubling, from `start` to `stop` inclusive."""
    if not (0 < start < stop) or pointsPerOctave < 1:
        raise ValueError("need 0 < start < stop and pointsPerOctave >= 1")
    count = math.ceil(math.log2(stop / start) * pointsPerOctave) + 1
    return np.logspace(math.log10(start), math.log10(stop), count)

def linearSweep(start: float, stop: float, points: int) -> FloatArray:
    """`points` evenly spaced frequencies from `start` to `stop` inclusive."""
    if points < 1 or start < 0 or stop < start:
        raise ValueError("need points >= 1 and 0 <= start <= stop")
    return np.linspace(start, stop, points)

def pwl(points: Sequence[tuple[float, float]]) -> Waveform:
    """Build a piecewise-linear waveform, matching SPICE's PWL source.
 
    The value is found by straight-line interpolation between the given ``(time, value)`` points. Before
    the first point it holds the first value, and after the last point it holds the last value, so the
    waveform is defined for all time (the solver may ask for any t >= 0).
 
    Args:
        points: ``(time, value)`` pairs with strictly increasing times, in seconds. At least one is
            required. A vertical edge cannot be written with equal times; use a very short ramp instead.
 
    Returns:
        A function suitable for ``Source(waveform=...)``.
 
    Raises:
        ValueError: If `points` is empty or its times are not strictly increasing.
    """
    if not points:
        raise ValueError("pwl needs at least one (time, value) point")
    times = [float(t) for t, _ in points]
    values = [float(v) for _, v in points]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("pwl times must be strictly increasing")
 
    def waveform(time: float) -> float:
        """Evaluate the piecewise-linear waveform at `time` (seconds)."""
        if time <= times[0]:
            return values[0]
        if time >= times[-1]:
            return values[-1]
        k = bisect.bisect_right(times, time)  # times[k-1] <= time < times[k]
        fraction = (time - times[k - 1]) / (times[k] - times[k - 1])
        return values[k - 1] + fraction * (values[k] - values[k - 1])
    return waveform
