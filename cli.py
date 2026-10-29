"""Command-line SPICE simulator (the graphical version is main.py).

    python cli.py circuit.cir                  print the results of every analysis in the netlist
    python cli.py circuit.cir --csv out/       also write one CSV file per analysis into out/
    python cli.py circuit.cir --method backward_euler
"""
import argparse
import sys
from pathlib import Path

from errors import CircuitError, ProbeError, SimulationError, SpiceError
from probes import probesFor
from report import formatTable, writeCsv
from simulate import simulate
from spiceParser import parseNetlist

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate a SPICE netlist.")
    parser.add_argument("netlist", help="path to the netlist file")
    parser.add_argument("--csv", metavar="DIR", help="also write each analysis's table to a CSV file in DIR")
    parser.add_argument("--method", choices=("trapezoidal", "backward_euler"), default="trapezoidal",
                        help="integration method for .tran (default: trapezoidal; backward_euler is safer for stiff or discontinuous circuits)")
    args = parser.parse_args(argv)

    try:
        try:
            netlist = parseNetlist(args.netlist)
        except OSError as error:
            print(f"error: cannot read {args.netlist}: {error.strerror or error}", file=sys.stderr)
            return 2
        solutions, warnings = simulate(netlist, method=args.method)

        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)

        print(netlist.title or Path(args.netlist).name)
        for number, solution in enumerate(solutions, start=1):
            probes = probesFor(solution, netlist.prints)
            print()
            print(formatTable(solution, probes))
            if args.csv:
                directory = Path(args.csv)
                directory.mkdir(parents=True, exist_ok=True)
                writeCsv(directory / f"{Path(args.netlist).stem}_{number}_{solution.analysis.kind}.csv", solution, probes)
    except (SpiceError, SimulationError, CircuitError, ProbeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
