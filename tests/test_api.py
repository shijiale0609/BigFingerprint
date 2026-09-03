import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from bigfingerprint import (
    FingerprintResult,
    generate_bigfingerprint_vector,
    generate_bigfingerprint_vectors,
)


class BigFingerprintApiTest(unittest.TestCase):
    def test_low_level_fingerprint_is_silent_and_has_expected_shape(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            vector = generate_bigfingerprint_vector(
                "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}",
                pecfp_iterations=0,
            )
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(vector.shape, (2088,))
        self.assertTrue(np.isin(vector, [0, 1]).all())

    def test_high_level_api_preserves_candidate_vector_mapping(self):
        fake_vector = np.array([0.0, 1.0])
        with patch(
            "bigfingerprint.api.generate_bigfingerprint_vector",
            return_value=fake_vector,
        ) as generate:
            result = generate_bigfingerprint_vectors(
                "{[>][<]CCO[>][<]}CC",
                preprocess_h=True,
            )

        self.assertIsInstance(result, FingerprintResult)
        self.assertEqual(result.candidates, ("CCO{[>][<]CCO[>][<]}CC",))
        np.testing.assert_array_equal(result.vector, fake_vector)
        generate.assert_called_once()

    def test_result_vector_rejects_ambiguous_candidates(self):
        result = FingerprintResult(
            "input",
            ("candidate-a", "candidate-b"),
            np.zeros((2, 3)),
        )
        with self.assertRaisesRegex(ValueError, "produced 2"):
            _ = result.vector


if __name__ == "__main__":
    unittest.main()
