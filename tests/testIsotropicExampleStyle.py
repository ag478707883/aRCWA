from pathlib import Path
import unittest


class IsotropicExampleStyleTests(unittest.TestCase):
    def testExamplesUseSimulationCudaInterface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        exampleDir = root / "examples" / "isotropic_example"
        disallowed = (
            "argparse",
            "add_argument",
            "parser",
            "args",
            "solveStackBatch",
            "if __name__",
            "EPS_AIR_TENSOR",
            "EPS_SI_TENSOR",
            "EPS_AU_TENSOR",
            "EPS_COVER_TENSOR",
            "EPS_SUBSTRATE_TENSOR",
            "EPS_GRATING_TENSOR",
            "EPS_GROOVE_TENSOR",
            'backend="cpu"',
            "backend='cpu'",
            'backend = "cpu"',
            "backend = 'cpu'",
            'backend="torch-cpu"',
            "backend='torch-cpu'",
            'backend = "torch-cpu"',
            "backend = 'torch-cpu'",
        )

        for path in sorted(exampleDir.glob("*.py")):
            if path.name == "run_examples.py":
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(example=path.name):
                for token in disallowed:
                    self.assertNotIn(token, text)
                self.assertIn('BACKEND = "cuda"', text)
                self.assertIn("PRECOMPILE = ", text)
                self.assertIn("CACHE_MODES = ", text)
                self.assertIn("rcwa.RCWASimulation(", text)
                self.assertIn("precompile=PRECOMPILE", text)
                self.assertIn("cacheModes=CACHE_MODES", text)
                self.assertTrue(".solve(" in text or ".spectrum(" in text or ".solveExcitations(" in text)
                if ".spectrum(" in text:
                    self.assertIn("WORKERS = ", text)
                    self.assertGreaterEqual(text.count("workers=WORKERS"), 2)


if __name__ == "__main__":
    unittest.main()
