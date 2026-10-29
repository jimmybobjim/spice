# SPICE solver

A small SPICE-style circuit simulator: modified nodal analysis, Newton iteration for nonlinear devices.

pip install -r requirements.txt
python main.py # opens the window; pick a netlist with the file dialog
python main.py examples/rc.cir # opens the window with a netlist already loaded
python -m unittest discover -s tests -t .

Needs Python 3.12+ with tkinter (included with Python; on Debian/Ubuntu: `sudo apt install python3-tk`),
plus numpy, scipy, matplotlib and customtkinter (see `requirements.txt`).

## Using the window

- **Open netlist…** (or Ctrl+O) shows your system's file explorer. The chosen file is simulated straight away;
  **Run again** (or F5) re-reads it from disk, so you can edit in any editor and re-run.
- **Results** shows each analysis as a table (long transients show 2000 evenly spaced rows). If the netlist has
  several analyses, a selector above the tabs switches between them.
- **Plot** draws the selected analysis: lines for `.dc/.ac/.tran`, bars for `.op`. Probes of different kinds
  (`v`, `vdb`, `vp`, `i` ...) get separate axes, so a Bode plot has magnitude over phase. The toolbar zooms,
  pans and saves the image.
- **Netlist** shows the file with line numbers, so "line 7" in an error can be found at once.
  **Messages** lists warnings and errors.
- **Export CSV…** asks where to save the selected analysis, with every point.
- **Transient method** and **Appearance** (system / light / dark) are in the sidebar.

The command line is also available: `python cli.py circuit.cir [--csv DIR] [--method backward_euler]`.

## Supported netlist features

| Devices  | `R C L V I D Q M`, and `X` subcircuit instances (nesting allowed)                                                                                                                             |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sources  | `[dc] v`, `dc v`, `ac mag [phase]`, `SIN PULSE PWL EXP`                                                                                                                                       |
| Models   | `.model` of type `d npn pnp nmos pmos` (D: `is n bv ibv`; BJT: `is bf br vaf`; MOS level 1: `vto kp lambda`, instance `w l m`)                                                                |
| Analyses | `.op`, `.dc` (one source), `.ac dec/oct/lin`, `.tran [tstart [tmax]] [uic]`                                                                                                                   |
| Output   | `.print/.plot/.probe`: `v(n)` `v(a,b)` `i(dev)` `vdb vp vm vr vi` (and `i*`), `ic ib ie` (BJT), `id ig is` (MOSFET). With none given, all node voltages and voltage-source currents are shown |
| Other    | `.subckt/.ends`, `.options` (`gmin reltol vntol abstol itl1`), `.title`, `.end`                                                                                                               |

Anything that would silently change the answer is an error (`.param`, `.include`, area factors, unknown device
letters ...). Things that cannot be honoured but only mildly affect results (`.ic`, `.nodeset`, `.temp`, model
parameters that are not modelled, the MOSFET bulk node) produce a warning on stderr.

## Layout

    main.py              graphical front end (starts gui.py)
    gui.py               the window: file dialog, worker thread, tabs
    plotting.py          matplotlib figures for a Solution (no GUI toolkit needed)
    cli.py               command-line front end
    spiceParser.py       netlist text -> Netlist; subcircuit flattening
    simulate.py          runs each analysis, returns uniform `Solution`s
    probes.py, report.py probe expressions; text/CSV output
    circuit/             devices (parse + stamp), terminals, sources, model cards, graph builder
    solver/              MNA system, Newton, DC/AC/transient analyses, waveforms
    graph.py             generic hypergraph the circuit is stored in
    util/                number syntax, validation helpers
    tests/, examples/

## Notes

- Currents follow the solver's convention: a device's current flows from its first terminal through it to its
  second, so a voltage source that is delivering power reports a _negative_ current (as in SPICE).
- A PMOS `vto` is negative for an enhancement device, as in SPICE (it is normalised internally).
- In `.tran`, `tstep` is also the integration step (reduced to `tmax` if smaller), the circuit is solved on a
  fixed grid, and `tstart` only hides the early part of the output. Trapezoidal integration treats a source that
  jumps _between_ grid points as a ramp across the step; use `--method backward_euler` or a smaller step for
  sharp edges.
