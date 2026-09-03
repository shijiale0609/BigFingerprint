import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from bigfingerprint import (
    generate_bigfingerprint_vector,
)


class BigFingerprintApiTest(unittest.TestCase):
    def test_standard_fingerprint_is_silent_and_has_expected_format(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            vector = generate_bigfingerprint_vector(
                "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}",
                pecfp_iterations=0,
            )
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(vector.shape, (2088,))
        self.assertEqual(vector.dtype, np.uint8)
        self.assertTrue(np.isin(vector, [0, 1]).all())

    def test_multiple_h_candidates_are_combined_by_bitwise_or(self):
        bigsmiles = "{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"
        candidate_vectors = [
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([0, 1, 1], dtype=np.uint8),
        ]
        with patch(
            "bigfingerprint.bigfingerprint._generate_pecfp_vector",
            side_effect=candidate_vectors,
        ) as generate, patch(
            "bigfingerprint.bigfingerprint.polymer_topfp.bigsmiles_topology_vector",
            return_value=np.array([1, 0], dtype=np.uint8),
        ) as topology:
            vector = generate_bigfingerprint_vector(
                bigsmiles,
                pecfp_iterations=2,
                fp_size=3,
            )

        np.testing.assert_array_equal(
            vector,
            np.array([1, 1, 1, 1, 0], dtype=np.uint8),
        )
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [candidate_call.args[0] for candidate_call in generate.call_args_list],
            [
                "CCO{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
                "CC(C)O{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
            ],
        )
        topology.assert_called_once_with(bigsmiles)


if __name__ == "__main__":
    unittest.main()
