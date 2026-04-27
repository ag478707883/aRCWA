from pathlib import Path
import unittest


class IsotropicExampleStyleTests(unittest.TestCase):
    def testExamplesUseOneReusableCudaSolverInterface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        exampleDir = root / "examples" / "isotropic_example"
        disallowed = (
            "argparse",
            "add_argument",
            "parser",
            "args",
            "RCWASimulation",
            "solveStackBatch",
            "def ",
            "if __name__",
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
                self.assertIn("rcwa.compileLayers(", text)
                self.assertIn("rcwa.solveStack(", text)


if __name__ == "__main__":
    unittest.main()
