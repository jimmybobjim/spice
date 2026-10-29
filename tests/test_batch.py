import tempfile
import unittest
from pathlib import Path

from batch import runExamples

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

class Batch(unittest.TestCase):
    def test_every_example_gets_an_image_per_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = runExamples(EXAMPLES, Path(tmp) / "results" / "nested")   # the folder is created
            names = sorted(p.name for p in report.images)
            self.assertEqual(report.failures, {})
            self.assertEqual(len(names), 10)
            self.assertIn("rc_1_ac.png", names)
            self.assertIn("rc_2_tran.png", names)
            for image in report.images:
                self.assertTrue(image.read_bytes().startswith(b"\x89PNG"), image)

    def test_a_bad_netlist_is_reported_and_the_rest_still_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in", Path(tmp) / "out"
            src.mkdir()
            (src / "bad.cir").write_text("t\nR1 a 0 oops\n.op\n")
            (src / "good.cir").write_text("t\nV1 a 0 5\nR1 a 0 1k\n.op\n")
            (src / "notes.md").write_text("ignored: not a netlist")
            report = runExamples(src, out)
            self.assertEqual([p.name for p in report.images], ["good_1_op.png"])
            self.assertEqual([p.name for p in report.failures], ["bad.cir"])
            self.assertIn("line 2", next(iter(report.failures.values())))

if __name__ == "__main__":
    unittest.main()