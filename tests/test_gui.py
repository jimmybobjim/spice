"""GUI tests that need no display: `runNetlist` is tested directly, and `App` is driven against stand-in
toolkit modules. This checks our own logic (file dialog -> worker thread -> tables, plot, CSV, errors);
it cannot check how the real widgets look. Run: python -m unittest discover -s tests -t ."""
import csv
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

class _Anything:
    """Accepts any construction, call or attribute, like a widget that does nothing."""
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return _Anything()
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()

class _Window(_Anything):
    pass

def _fakeModule(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__getattr__ = lambda attribute: _Anything()   # type: ignore[attr-defined]
    for key, value in attributes.items():
        setattr(module, key, value)
    return module

class GuiLogic(unittest.TestCase):
    FAKES = ("tkinter", "tkinter.filedialog", "tkinter.font", "customtkinter", "matplotlib.backends.backend_tkagg", "gui")

    def setUp(self):
        self.saved = {name: sys.modules.get(name) for name in self.FAKES}
        self.dialogs = types.SimpleNamespace(open=str(EXAMPLES / "rc.cir"), save="")
        filedialog = _fakeModule("tkinter.filedialog",
                                 askopenfilename=lambda **kw: self.dialogs.open,
                                 asksaveasfilename=lambda **kw: self.dialogs.save)
        tk = _fakeModule("tkinter", filedialog=filedialog, font=_fakeModule("tkinter.font"))
        sys.modules.update({
            "tkinter": tk, "tkinter.filedialog": filedialog, "tkinter.font": tk.font,
            "customtkinter": _fakeModule("customtkinter", CTk=_Window),
            "matplotlib.backends.backend_tkagg": _fakeModule("matplotlib.backends.backend_tkagg"),
        })
        sys.modules.pop("gui", None)
        from gui import gui
        self.gui = gui
        self.status: list[tuple[str, bool]] = []
        self.texts: dict[str, str] = {}
        self.app = gui.App()
        self.app._setStatus = lambda text, error=False: self.status.append((text, error))   # type: ignore[method-assign]
        self.app._setText = lambda box, text: self.texts.__setitem__(id(box), text)         # type: ignore[method-assign]

    def tearDown(self):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def waitForResult(self):
        deadline = time.time() + 20
        while time.time() < deadline:
            before = len(self.status)
            self.app._poll()
            if not self.app._busy and len(self.status) > before:
                return
            time.sleep(0.02)
        self.fail("the worker thread never reported back")

    def test_run_netlist_renders_every_analysis(self):
        loaded = self.gui.runNetlist(EXAMPLES / "rc.cir", "trapezoidal")
        self.assertEqual([v.label for v in loaded.views], ["1: .ac", "2: .tran"])
        self.assertIn("vdb(out)", loaded.views[0].text)
        self.assertEqual(loaded.title, "RC low-pass")

    def test_numbered_source(self):
        self.assertEqual(self.gui.numberedSource("a\nb"), "   1  a\n   2  b")

    def test_open_dialog_loads_runs_and_shows_results(self):
        self.app.openFile()                       # the (faked) file dialog returns examples/rc.cir
        self.assertEqual(self.app.path, EXAMPLES / "rc.cir")
        self.waitForResult()
        self.assertEqual(len(self.app.views), 2)
        self.assertFalse(self.status[-1][1], self.status[-1][0])
        self.app._select("2: .tran")
        self.assertEqual(self.app.current, 1)
        self.app._renderPlot()                    # builds the real matplotlib figure for the current analysis

    def test_cancelling_the_dialog_does_nothing(self):
        self.dialogs.open = ""
        self.app.openFile()
        self.assertIsNone(self.app.path)
        self.assertFalse(self.app._busy)

    def test_bad_netlist_reports_the_error_and_stays_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.cir"
            bad.write_text("t\nR1 a 0 oops\n.op\n")
            self.app.load(bad)
            self.waitForResult()
        self.assertTrue(self.status[-1][1])
        self.assertIn("line 2", self.status[-1][0])
        self.assertFalse(self.app._busy)

    def test_missing_file_is_reported(self):
        self.app.load(Path("/definitely/not/here.cir"))
        self.waitForResult()
        self.assertTrue(self.status[-1][1])

    def test_export_csv_writes_the_selected_analysis(self):
        self.app.load(EXAMPLES / "rc.cir")
        self.waitForResult()
        self.app._select("2: .tran")
        with tempfile.TemporaryDirectory() as tmp:
            self.dialogs.save = str(Path(tmp) / "out.csv")
            self.app.exportCsv()
            with open(self.dialogs.save, newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["time", "v(in)", "v(out)"])
        self.assertEqual(len(rows), 1 + 31)

if __name__ == "__main__":
    unittest.main()