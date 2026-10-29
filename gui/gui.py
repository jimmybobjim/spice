"""Graphical front end: pick a netlist in the system file dialog, run it, browse tables and plots.

The simulation itself runs on a worker thread (`runNetlist`) so the window never freezes; its result comes
back through a queue that the Tk thread polls. `runNetlist` touches no widgets, so it is testable on its own.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from tkinter import font as tkfont
from typing import Any, Literal

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from errors import CircuitError, ProbeError, SimulationError, SpiceError
from gui.plotting import makeFigure
from probes import probesFor
from report import formatTable, writeCsv
from simulate import Solution, simulate
from spiceParser import parseNetlist

type Method = Literal["trapezoidal", "backward_euler"]

ctk.set_appearance_mode("system")   # follow the operating system's light/dark setting
ctk.set_default_color_theme("blue")

_FILETYPES = [("SPICE netlists", "*.cir *.sp *.spice *.net *.ckt *.txt"), ("All files", "*.*")]
_TEXT_ROWS = 2000   # the Results tab shows at most this many rows; CSV export has them all
_MONO_CANDIDATES = ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono", "Courier New")


# ---- work that needs no widgets -----------------------------------------------

@dataclass(frozen=True)
class AnalysisView:
    """One analysis ready to display: its solution, the probes to show, and the rendered table."""
    label: str
    solution: Solution
    probes: tuple[str, ...]
    text: str

@dataclass(frozen=True)
class Loaded:
    title: str
    source: str
    views: list[AnalysisView]
    warnings: list[str]

def runNetlist(path: Path, method: Method) -> Loaded:
    """Parse and simulate `path`, rendering every table. Raises the simulator's own errors unchanged."""
    netlist = parseNetlist(str(path))
    solutions, warnings = simulate(netlist, method=method)
    views = []
    for number, solution in enumerate(solutions, start=1):
        probes = probesFor(solution, netlist.prints)
        views.append(AnalysisView(f"{number}: .{solution.analysis.kind}", solution, probes,
                                  formatTable(solution, probes, maxRows=_TEXT_ROWS)))
    source = path.read_text(encoding="utf-8", errors="replace")
    return Loaded(netlist.title, source, views, warnings)

def numberedSource(source: str) -> str:
    """The netlist with line numbers, so 'line 7' in an error message can be found at a glance."""
    return "\n".join(f"{number:>4}  {line}" for number, line in enumerate(source.splitlines(), start=1))

def _monoFamily(root: Any) -> str:
    try:
        available = set(tkfont.families(root))
        for family in _MONO_CANDIDATES:
            if family in available:
                return family
        return str(tkfont.nametofont("TkFixedFont").actual("family"))
    except Exception:   # a missing font must never stop the window opening
        return "Courier"


# ---- the window -------------------------------------------------------------------

class App(ctk.CTk):
    def __init__(self, initialPath: str | None = None) -> None:
        super().__init__()
        self.title("SPICE Solver")
        self.geometry("1180x740")
        self.minsize(940, 580)

        self.path: Path | None = None
        self.method: Method = "trapezoidal"
        self.views: list[AnalysisView] = []
        self.current = 0
        self._busy = False
        self._outcomes: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._canvas: FigureCanvasTkAgg | None = None
        self._toolbar: NavigationToolbar2Tk | None = None
        self._mono = ctk.CTkFont(family=_monoFamily(self), size=13)

        self._build()
        self.bind("<Control-o>", lambda _event: self.openFile())
        self.bind("<F5>", lambda _event: self.run())
        if initialPath:
            self.after(100, lambda: self.load(Path(initialPath)))

    # ---- layout

    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, width=250, corner_radius=0)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(side, text="SPICE Solver", font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, padx=20, pady=(24, 0), sticky="w")
        ctk.CTkLabel(side, text="Circuit simulator", text_color=("gray40", "gray65")).grid(row=1, column=0, padx=20, pady=(0, 18), sticky="w")

        self.openButton = ctk.CTkButton(side, text="Open netlist…", height=38, command=self.openFile)
        self.openButton.grid(row=2, column=0, padx=20, pady=(0, 6), sticky="ew")
        self.fileLabel = ctk.CTkLabel(side, text="No file selected", wraplength=210, justify="left", anchor="w", text_color=("gray40", "gray65"))
        self.fileLabel.grid(row=3, column=0, padx=20, pady=(0, 14), sticky="ew")

        self.runButton = ctk.CTkButton(side, text="Run again  (F5)", height=34, state="disabled", command=self.run)
        self.runButton.grid(row=4, column=0, padx=20, pady=(0, 6), sticky="ew")
        self.exportButton = ctk.CTkButton(
            side, text="Export CSV…", height=34, state="disabled", command=self.exportCsv,
            fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),
        )
        self.exportButton.grid(row=5, column=0, padx=20, pady=(0, 18), sticky="ew")

        ctk.CTkLabel(side, text="Transient method", anchor="w").grid(row=6, column=0, padx=20, sticky="ew")
        ctk.CTkOptionMenu(side, values=["trapezoidal", "backward_euler"], command=self._setMethod).grid(row=7, column=0, padx=20, pady=(4, 0), sticky="ew")

        ctk.CTkLabel(side, text="Appearance", anchor="w").grid(row=9, column=0, padx=20, sticky="ew")
        appearance = ctk.CTkOptionMenu(side, values=["System", "Light", "Dark"], command=self._setAppearance)
        appearance.grid(row=10, column=0, padx=20, pady=(4, 20), sticky="ew")

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=18, pady=(14, 8))
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        self.heading = ctk.CTkLabel(main, text="Open a netlist to begin", font=ctk.CTkFont(size=20, weight="bold"), anchor="w")
        self.heading.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.selector = ctk.CTkSegmentedButton(main, values=[" "], command=self._select)
        self.selector.grid(row=1, column=0, sticky="w", pady=(0, 8))
        self.selector.grid_remove()

        self.tabs = ctk.CTkTabview(main)
        self.tabs.grid(row=2, column=0, sticky="nsew")
        for name in ("Results", "Plot", "Netlist", "Messages"):
            self.tabs.add(name)
            self.tabs.tab(name).grid_columnconfigure(0, weight=1)
            self.tabs.tab(name).grid_rowconfigure(0, weight=1)

        self.results = self._textbox("Results")
        self.netlistBox = self._textbox("Netlist")
        self.messages = self._textbox("Messages")
        self.plotHolder = ctk.CTkFrame(self.tabs.tab("Plot"), fg_color="transparent")
        self.plotHolder.grid(row=0, column=0, sticky="nsew")
        self.plotNote = ctk.CTkLabel(self.plotHolder, text="Run a simulation to see its plot.", text_color=("gray40", "gray65"))
        self.plotNote.pack(expand=True)

        self.progress = ctk.CTkProgressBar(main, mode="indeterminate", height=6)
        self.progress.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.progress.grid_remove()
        self.status = ctk.CTkLabel(main, text="Ready.  Ctrl+O opens a netlist.", anchor="w", text_color=("gray40", "gray65"))
        self.status.grid(row=4, column=0, sticky="ew", pady=(6, 0))

        self._setText(self.results, "Choose a netlist with “Open netlist…” (Ctrl+O) and it runs straight away.\n\n"
                                    "Results appear here as tables, in the Plot tab as graphs, and any\n"
                                    "warnings or errors in the Messages tab.")

    def _textbox(self, tab: str) -> ctk.CTkTextbox:
        box = ctk.CTkTextbox(self.tabs.tab(tab), font=self._mono, wrap="none", state="disabled")
        box.grid(row=0, column=0, sticky="nsew")
        return box

    # ---- helpers

    @staticmethod
    def _setText(box: ctk.CTkTextbox, text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _setStatus(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, text_color=("#c0392b", "#ff6b5e") if error else ("gray40", "gray65"))

    def _setBusy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self.progress.grid()
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.grid_remove()
        state = "disabled" if busy else "normal"
        self.openButton.configure(state=state)
        self.runButton.configure(state="disabled" if busy or self.path is None else "normal")

    def _setMethod(self, method: str) -> None:
        self.method = "backward_euler" if method == "backward_euler" else "trapezoidal"

    def _setAppearance(self, mode: str) -> None:
        ctk.set_appearance_mode(mode.lower())
        self._renderPlot()   # the figure draws its own colours, so it has to be rebuilt

    # ---- opening and running

    def openFile(self) -> None:
        """Show the system's file explorer and load whatever the user picks."""
        chosen = filedialog.askopenfilename(
            title="Open SPICE netlist", filetypes=_FILETYPES,
            initialdir=str(self.path.parent) if self.path else None,
        )
        if chosen:
            self.load(Path(chosen))

    def load(self, path: Path) -> None:
        self.path = path
        self.fileLabel.configure(text=path.name)
        self.run()

    def run(self) -> None:
        if self.path is None or self._busy:
            return
        self._setBusy(True)
        self._setStatus(f"Simulating {self.path.name}…")
        threading.Thread(target=self._work, args=(self.path, self.method), daemon=True).start()
        self.after(80, self._poll)

    def _work(self, path: Path, method: Method) -> None:
        """Worker thread: no widgets here, only the queue."""
        try:
            self._outcomes.put(("ok", runNetlist(path, method)))
        except (SpiceError, SimulationError, CircuitError, ProbeError) as error:
            self._outcomes.put(("error", str(error)))
        except OSError as error:
            self._outcomes.put(("error", f"cannot read {path}: {error.strerror or error}"))
        except Exception as error:   # keep the window usable whatever goes wrong inside the simulator
            self._outcomes.put(("error", f"unexpected {type(error).__name__}: {error}"))

    def _poll(self) -> None:
        try:
            kind, payload = self._outcomes.get_nowait()
        except queue.Empty:
            self.after(80, self._poll)
            return
        self._setBusy(False)
        if kind == "error":
            self._showError(payload)
        else:
            self._showLoaded(payload)

    # ---- showing results

    def _showError(self, message: str) -> None:
        self._setText(self.messages, f"Error\n\n{message}\n")
        self.tabs.set("Messages")
        self._setStatus(message.splitlines()[0], error=True)

    def _showLoaded(self, loaded: Loaded) -> None:
        self.views = loaded.views
        self.current = 0
        self.heading.configure(text=loaded.title or (self.path.name if self.path else ""))
        self._setText(self.netlistBox, numberedSource(loaded.source))
        self._setText(self.messages, "\n".join(f"warning: {w}" for w in loaded.warnings) or "No warnings.")

        if len(self.views) > 1:
            self.selector.configure(values=[v.label for v in self.views])
            self.selector.set(self.views[0].label)
            self.selector.grid()
        else:
            self.selector.grid_remove()

        self.exportButton.configure(state="normal")
        self._showCurrent()
        self.tabs.set("Results")
        count = len(loaded.warnings)
        self._setStatus(f"Done: {len(self.views)} analysis(es)" + (f", {count} warning(s) in the Messages tab." if count else "."))

    def _select(self, label: str) -> None:
        self.current = next((k for k, v in enumerate(self.views) if v.label == label), 0)
        self._showCurrent()

    def _showCurrent(self) -> None:
        self._setText(self.results, self.views[self.current].text)
        self._renderPlot()

    def _renderPlot(self) -> None:
        if not self.views:
            return
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        self.plotNote.pack_forget()

        view = self.views[self.current]
        try:
            figure = makeFigure(view.solution, view.probes, dark=ctk.get_appearance_mode() == "Dark")
        except Exception as error:   # a plotting problem should not hide the numeric results
            self.plotNote.configure(text=f"Could not draw this plot: {error}")
            self.plotNote.pack(expand=True)
            return
        self._canvas = FigureCanvasTkAgg(figure, master=self.plotHolder)
        self._toolbar = NavigationToolbar2Tk(self._canvas, self.plotHolder, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side="bottom", fill="x")
        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self._canvas.draw()

    def exportCsv(self) -> None:
        if not self.views or self.path is None:
            return
        view = self.views[self.current]
        chosen = filedialog.asksaveasfilename(
            title="Export CSV", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.path.stem}_{view.solution.analysis.kind}.csv", initialdir=str(self.path.parent),
        )
        if not chosen:
            return
        try:
            writeCsv(Path(chosen), view.solution, view.probes)
        except OSError as error:
            self._setStatus(f"Could not save {chosen}: {error.strerror or error}", error=True)
        else:
            self._setStatus(f"Saved {chosen}")