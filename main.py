"""SPICE circuit simulator, graphical front end.

    python main.py                  open the window, then pick a netlist with the file dialog (Ctrl+O)
    python main.py circuit.cir      open the window with that netlist already loaded

The command-line version is cli.py.
"""
import argparse
import sys

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate SPICE netlists in a window.")
    parser.add_argument("netlist", nargs="?", help="optional netlist to load at startup")
    args = parser.parse_args(argv)

    try:
        from gui.gui import App
    except ModuleNotFoundError as error:
        if error.name not in ("tkinter", "customtkinter", "matplotlib"):
            raise
        hint = ("tkinter ships with Python; on Debian/Ubuntu install it with: sudo apt install python3-tk"
                if error.name == "tkinter" else "install the GUI dependencies with: pip install -r requirements.txt")
        print(f"error: the {error.name!r} package is missing. {hint}", file=sys.stderr)
        return 1

    App(args.netlist).mainloop()
    return 0

if __name__ == "__main__":
    sys.exit(main())