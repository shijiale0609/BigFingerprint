import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigfingerprint.cli import main


class BigFingerprintCliTest(unittest.TestCase):
    def test_supported_radii_generate_vectors(self):
        bigsmiles = "CO{[>][<]CCO[>],[<]CC(C)O[>][<]}C"
        for radius, on_bits in ((0, 6), (1, 16), (2, 36)):
            with self.subTest(radius=radius):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = main(
                        ["fingerprint", bigsmiles, "--radius", str(radius)]
                    )
                payload = json.loads(output.getvalue())
                self.assertEqual(status, 0)
                self.assertEqual(payload["size"], 2088)
                self.assertEqual(len(payload["on_bits"]), on_bits)

    def test_unsupported_radius_is_rejected_before_computation_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fingerprint.json"
            for radius in (-1, 3, 100):
                with self.subTest(radius=radius):
                    error = io.StringIO()
                    with (
                        patch(
                            "bigfingerprint.cli.generate_bigfingerprint_vector"
                        ) as generate,
                        contextlib.redirect_stderr(error),
                        self.assertRaises(SystemExit) as exit_context,
                    ):
                        main(
                            [
                                "fingerprint",
                                "{[][<]CCO[>][]}",
                                "--radius",
                                str(radius),
                                "--output",
                                str(output),
                            ]
                        )
                    self.assertEqual(exit_context.exception.code, 2)
                    self.assertIn("invalid choice", error.getvalue())
                    generate.assert_not_called()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
