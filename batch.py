"""Run every netlist in a folder and save an image of each analysis's graph.

    python batch.py                        examples/ -> results/
    python batch.py my_netlists out_dir    any folders (relative paths are relative to where you run it)

Images are named <netlist>_<n>_<analysis>.png, e.g. rc_1_ac.png and rc_2_tran.png. No window is needed.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

from errors import CircuitError, ProbeError, SimulationError, SpiceError
from gui.plotting import makeFigure
from probes import probesFor
from simulate import simulate
from spiceParser import parseNetlist

_HERE = Path(__file__).resolve().parent
NETLIST_SUFFIXES = (".cir", ".sp", ".spice", ".net", ".ckt")

@dataclass
class BatchReport:
    images: list[Path] = field(default_factory=list[Path])
    failures: dict[Path, str] = field(default_factory=dict[Path, str])   # netlist -> why it could not be run

def runExamples(examplesDir: Path = _HERE / "examples", resultsDir: Path = _HERE / "results",
                *, dark: bool = False, dpi: int = 150) -> BatchReport:
    """Simulate every netlist in `examplesDir` and save one PNG per analysis into `resultsDir` (created if needed).

    A netlist that fails to parse or simulate is recorded in `failures` and the rest still run."""
    examplesDir, resultsDir = Path(examplesDir), Path(resultsDir)
    netlists = sorted(p for p in examplesDir.iterdir() if p.is_file() and p.suffix.lower() in NETLIST_SUFFIXES)
    resultsDir.mkdir(parents=True, exist_ok=True)

    report = BatchReport()
    for path in netlists:
        try:
            netlist = parseNetlist(str(path))
            solutions, _warnings = simulate(netlist)
            for number, solution in enumerate(solutions, start=1):
                figure = makeFigure(solution, probesFor(solution, netlist.prints), dark=dark)
                image = resultsDir / f"{path.stem}_{number}_{solution.analysis.kind}.png"
                figure.savefig(image, dpi=dpi)
                report.images.append(image)
        except (SpiceError, SimulationError, CircuitError, ProbeError, OSError) as error:
            report.failures[path] = str(error)
    return report

def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 2:
        print("usage: python batch.py [examples_dir [results_dir]]", file=sys.stderr)
        return 2
    kwargs = {}
    if args:
        kwargs["examplesDir"] = Path(args[0])
    if len(args) > 1:
        kwargs["resultsDir"] = Path(args[1])
    try:
        report = runExamples(**kwargs)
    except OSError as error:
        print(f"error: {error.strerror or error}: {error.filename}", file=sys.stderr)
        return 2
    for image in report.images:
        print(f"saved {image}")
    for path, reason in report.failures.items():
        print(f"FAILED {path.name}: {reason}", file=sys.stderr)
    print(f"{len(report.images)} image(s) saved, {len(report.failures)} netlist(s) failed")
    return 1 if report.failures else 0

if __name__ == "__main__":
    sys.exit(main())