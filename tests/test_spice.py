"""Regression tests. Run from the project root:  python -m unittest discover -s tests -t ."""
import math
import unittest

import numpy as np

from errors import ProbeError, SimulationError, SpiceError
from probes import evaluate
from simulate import simulate
from spiceParser import flatten, parseText, readLines
from util.spicetext import parseValue

def run(text: str, kind: str | None = None):
    solutions, _ = simulate(parseText(text))
    return solutions[0] if kind is None else next(s for s in solutions if s.analysis.kind == kind)

def v(solution, probe: str, index: int = 0) -> float:
    return float(evaluate(probe, solution)[index])

class Parsing(unittest.TestCase):
    def test_values(self):
        for text, expected in [("1k", 1e3), ("4.7u", 4.7e-6), ("1meg", 1e6), ("2.2nF", 2.2e-9), ("10mil", 254e-6), ("1e-3", 1e-3)]:
            self.assertEqual(parseValue(text), expected)

    def test_logical_lines_keep_their_own_numbers_and_the_last_line(self):
        lines = list(readLines("title\n* comment\nR1 a b\n+ 1k ; trailing\n\nC1 a 0 1u"))
        self.assertEqual([(l.number, l.text) for l in lines], [(1, "title"), (3, "R1 a b 1k"), (6, "C1 a 0 1u")])

    def test_model_may_follow_its_user(self):
        netlist = parseText("t\nV1 a 0 5\nD1 a 0 dm\n.model dm D(is=2e-14)\n")
        self.assertEqual(flatten(netlist)[1].saturationCurrent, 2e-14)

    def test_title_that_merely_starts_like_a_device_is_not_warned_about(self):
        self.assertEqual(parseText("Low-pass filter with a long title\nR1 a 0 1k\n").warnings, [])

    def test_nested_subcircuit_names_and_nodes(self):
        netlist = parseText("t\n.subckt d top bot mid\nR1 top mid 1k\nR2 mid bot 1k\n.ends\n"
                            ".subckt two in out\nX1 in out m d\nR3 m 0 1k\n.ends\nV1 a 0 1\nXA a b two\n")
        flat = {c.name: c.nodes for c in flatten(netlist)}
        self.assertEqual(flat["xa.x1.r1"], ("a", "xa.m"))
        self.assertEqual(flat["xa.r3"], ("xa.m", "0"))

    def test_errors_carry_line_numbers(self):
        for text, line in [("t\nR1 a 0 abc\n", 2), ("t\nV1 a 0 1\nR1 a 0 -5\n", 3), ("t\nJ1 a b c m\n", 2), ("t\nQ1 c b e\n", 2)]:
            with self.assertRaises(SpiceError) as caught:
                parseText(text)
            self.assertEqual(caught.exception.line, line)

class Solving(unittest.TestCase):
    def test_divider(self):
        s = run("t\nV1 in 0 10\nR1 in out 1k\nR2 out 0 3k\n.op\n")
        self.assertAlmostEqual(v(s, "v(out)"), 7.5, places=6)
        self.assertAlmostEqual(v(s, "i(v1)"), -2.5e-3, places=9)

    def test_current_source_pushes_from_p_to_n(self):   # the p terminal was stamped with the wrong sign
        into = run("t\nI1 0 a 1m\nR1 a 0 2k\n.op\n")      # p is ground: pushes 1 mA into a
        self.assertAlmostEqual(v(into, "v(a)"), 2.0, places=6)
        out = run("t\nI1 a 0 1m\nR1 a 0 2k\n.op\n")       # p is a real node: draws 1 mA out of a
        self.assertAlmostEqual(v(out, "v(a)"), -2.0, places=6)
        both = run("t\nI1 a b 1m\nR1 a 0 1k\nR2 b 0 1k\n.op\n")   # moves current between two nodes
        self.assertAlmostEqual(v(both, "v(a)"), -1.0, places=6)
        self.assertAlmostEqual(v(both, "v(b)"), 1.0, places=6)

    def test_rc_step_response(self):
        s = run("t\nV1 in 0 PWL(0 1 1 1)\nR1 in out 1k\nC1 out 0 1u\n.tran 10u 5m uic\n")
        k = int(np.searchsorted(s.axis, 1e-3))
        self.assertAlmostEqual(v(s, "v(out)", k), 1 - math.exp(-1), delta=5e-3)

    def test_rc_low_pass_corner(self):
        s = run("t\nV1 in 0 ac 1\nR1 in out 1k\nC1 out 0 1u\n.ac dec 10 10 100k\n")
        k = int(np.abs(s.axis - 159.1549).argmin())
        self.assertAlmostEqual(v(s, "vdb(out)", k), -3.01, delta=0.1)
        self.assertAlmostEqual(v(s, "vp(out)", k), -45.0, delta=1.5)

    def test_bjt_current_gain(self):
        s = run("t\nVCC vcc 0 12\nRB vcc b 1meg\nRC vcc c 4.7k\nQ1 c b 0 q\n.model q NPN(is=1e-14 bf=100)\n.op\n")
        self.assertAlmostEqual(v(s, "ic(q1)") / v(s, "ib(q1)"), 100.0, delta=1.0)

    def test_nmos_triode_point(self):   # solved by hand: Vds = 0.2721 V
        s = run("t\nVDD vdd 0 5\nVG g 0 2\nRD vdd d 10k\nM1 d g 0 0 nm w=20u l=1u\n.model nm NMOS(vto=1 kp=100u lambda=0.02)\n.op\n")
        self.assertAlmostEqual(v(s, "v(d)"), 0.27214, places=4)

    def test_pmos_uses_spice_negative_vto(self):   # 5 Vsd^2 - 11 Vsd + 5 = 0 -> Vd = 4.358 V
        s = run("t\nVDD vdd 0 5\nVG g 0 3\nRL d 0 10k\nM1 d g vdd vdd pm w=20u l=1u\n.model pm PMOS(vto=-1 kp=50u)\n.op\n")
        self.assertAlmostEqual(v(s, "v(d)"), 4.35826, places=4)

    def test_zener_transient_converges_and_clamps(self):   # roundoff in Diode.limit used to make Newton never converge
        s = run("t\nV1 in 0 PWL(0 0 1m 10)\nR1 in out 1k\nD1 0 out zd\n.model zd D(is=1e-14 bv=5.1 ibv=5m)\n.tran 100u 1m\n")
        self.assertAlmostEqual(v(s, "v(out)", -1), 5.1, delta=0.1)

    def test_dc_sweep_does_not_modify_the_circuit(self):
        s = run("t\nV1 a 0 0\nR1 a b 1k\nR2 b 0 1k\n.dc V1 0 4 1\n")
        np.testing.assert_allclose(evaluate("v(b)", s), [0, 0.5, 1, 1.5, 2], atol=1e-6)

    def test_sources_follow_spice_definitions(self):
        s = run("t\nV1 a 0 SIN(1 2 1k 0 0 90)\nV2 b 0 EXP(0 5 0 1m 5m 1m)\nR1 a 0 1k\nR2 b 0 1k\n.tran 0.1m 2m\n")
        k = int(np.searchsorted(s.axis, 1e-3))
        self.assertAlmostEqual(v(s, "v(a)", k), 1 + 2 * math.sin(math.pi / 2 + 2 * math.pi), places=6)
        self.assertAlmostEqual(v(s, "v(b)", k), 5 * (1 - math.exp(-1)), places=6)

class Failures(unittest.TestCase):
    def test_no_ground(self):
        with self.assertRaises(SimulationError):
            run("t\nV1 a b 5\nR1 a b 1k\n.op\n")

    def test_sweeping_something_that_is_not_a_source(self):
        with self.assertRaises(SimulationError):
            run("t\nV1 a 0 5\nR1 a 0 1k\n.dc R1 0 1 0.1\n")

    def test_bad_probe(self):
        s = run("t\nV1 a 0 5\nR1 a 0 1k\n.op\n")
        for probe in ("v(nope)", "i(r9)", "bogus(a)", "ic(r1)"):
            with self.assertRaises(ProbeError):
                evaluate(probe, s)

if __name__ == "__main__":
    unittest.main()
