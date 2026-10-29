"""Turn solutions into text tables and CSV files."""
import csv
from pathlib import Path

import numpy as np

from probes import evaluate
from simulate import Solution
from spiceParser import AcAnalysis, DcSweep, OpAnalysis, TranAnalysis

def describe(solution: Solution) -> str:
    match solution.analysis:
        case OpAnalysis():
            return "DC operating point"
        case DcSweep(source=name, start=start, stop=stop, step=step):
            return f"DC sweep of {name} from {start:g} to {stop:g} in steps of {step:g}"
        case AcAnalysis(variation=variation, points=points, start=start, stop=stop):
            return f"AC sweep, {points} points ({variation}) from {start:g} Hz to {stop:g} Hz"
        case TranAnalysis(stop=stop):
            return f"Transient analysis to {stop:g} s"

def formatTable(solution: Solution, probes: tuple[str, ...], maxRows: int | None = None) -> str:
    """A text rendering: name = value lines for .op, one row per point otherwise.

    With `maxRows`, a longer table shows that many evenly spaced points (always including the first and last)."""
    columns = {probe: evaluate(probe, solution) for probe in probes}
    lines = [describe(solution), ""]

    if isinstance(solution.analysis, OpAnalysis):
        width = max((len(probe) for probe in probes), default=0)
        lines += [f"{probe:<{width}} = {values[0]:.6g}" for probe, values in columns.items()]
        return "\n".join(lines)

    headers = [solution.axisName, *probes]
    widths = [max(14, len(header)) for header in headers]
    lines.append("  ".join(f"{header:>{width}}" for header, width in zip(headers, widths)))
    lines.append("  ".join("-" * width for width in widths))
    count = solution.axis.size
    rows = range(count) if maxRows is None or count <= maxRows else sorted({int(k) for k in np.linspace(0, count - 1, maxRows)})
    for k in rows:
        cells = [solution.axis[k], *(values[k] for values in columns.values())]
        lines.append("  ".join(f"{cell:>{width}.6e}" for cell, width in zip(cells, widths)))
    if len(rows) < count:
        lines += ["", f"showing {len(rows)} of {count} points; export to CSV to get all of them"]
    return "\n".join(lines)

def writeCsv(path: Path, solution: Solution, probes: tuple[str, ...]) -> None:
    columns = [evaluate(probe, solution) for probe in probes]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([solution.axisName or "point", *probes])
        for k in range(solution.axis.size):
            writer.writerow([f"{solution.axis[k]:.9g}", *(f"{float(np.real(column[k])):.9g}" for column in columns)])
