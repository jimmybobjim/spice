"""Build matplotlib figures for a `Solution`. Needs no GUI toolkit, so it can be tested and reused anywhere.

Probes are grouped by function (all ``v(..)`` together, ``vdb(..)`` and ``vp(..)`` each on their own axes,
and so on) so that quantities with different units never share a y axis.
"""
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from circuit.components import VoltageSource
from probes import evaluate
from simulate import Solution
from spiceParser import AcAnalysis, OpAnalysis

_PALETTE = ("#4c9be8", "#f08a4b", "#58b368", "#d9534f", "#a77bd8", "#c9a227", "#3fb5b5", "#e377c2")
_MAX_PLOT_POINTS = 20_000   # longer traces are thinned for display (a CSV export keeps every point)

def groupProbes(probes: tuple[str, ...]) -> dict[str, list[str]]:
    """Probes by function name, in order of first appearance: ``{"v": ["v(a)", "v(b)"], "i": ["i(v1)"]}``."""
    groups: dict[str, list[str]] = {}
    for probe in probes:
        groups.setdefault(probe.split("(", 1)[0], []).append(probe)
    return groups

def yLabel(function: str) -> str:
    if function.endswith("db"):
        return "Magnitude (dB)"
    if function in ("vp", "ip"):
        return "Phase (°)"
    return "Voltage (V)" if function.startswith("v") else "Current (A)"

def xLabel(solution: Solution) -> str:
    match solution.axisName:
        case "time":
            return "Time (s)"
        case "frequency":
            return "Frequency (Hz)"
        case name:
            device = solution.devices.get(name)
            return f"{name} ({'V' if isinstance(device, VoltageSource) else 'A'})"

def _style(ax: Axes, dark: bool) -> None:
    text, grid, background = ("#e6e6e6", "#5a5a5a", "#2b2b2b") if dark else ("#1a1a1a", "#b0b0b0", "#f4f4f4")
    ax.set_facecolor(background)
    ax.tick_params(colors=text)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)
    for side, spine in ax.spines.items():
        spine.set_color(grid)
        spine.set_visible(side in ("left", "bottom"))
    ax.grid(True, color=grid, alpha=0.4, linewidth=0.7)

def makeFigure(solution: Solution, probes: tuple[str, ...], dark: bool = False) -> Figure:
    """One axes per probe function: lines against the sweep axis, or bars for an operating point."""
    figure = Figure(figsize=(8.0, 5.6), dpi=100, facecolor="#2b2b2b" if dark else "#ebebeb", layout="constrained")
    groups = groupProbes(probes)
    isOp = isinstance(solution.analysis, OpAnalysis)
    axes = figure.subplots(len(groups), 1, sharex=not isOp, squeeze=False)[:, 0]
    text = "#e6e6e6" if dark else "#1a1a1a"

    stride = max(1, solution.axis.size // _MAX_PLOT_POINTS)
    for ax, (function, names) in zip(axes, groups.items()):
        _style(ax, dark)
        if isOp:
            values = [float(evaluate(name, solution)[0]) for name in names]
            ax.bar(names, values, color=[_PALETTE[k % len(_PALETTE)] for k in range(len(names))], width=0.6)
            ax.axhline(0.0, color=text, linewidth=0.8)
            ax.set_xlim(-0.6, max(len(names), 3) - 0.4)   # a lone bar should not fill the whole axes
            if len(names) > 6:
                ax.tick_params(axis="x", labelrotation=45)
        else:
            for k, name in enumerate(names):
                ax.plot(solution.axis[::stride], evaluate(name, solution)[::stride], label=name,
                        color=_PALETTE[k % len(_PALETTE)], linewidth=1.7)
            legend = ax.legend(loc="best", frameon=False)
            for entry in legend.get_texts():
                entry.set_color(text)
            if isinstance(solution.analysis, AcAnalysis) and solution.analysis.variation != "lin":
                ax.set_xscale("log")
        ax.set_ylabel(yLabel(function))

    if not isOp:
        axes[-1].set_xlabel(xLabel(solution))
    return figure