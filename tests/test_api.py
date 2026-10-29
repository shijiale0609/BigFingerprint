import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from bigfingerprint import (
    generate_bigfingerprint_vector,
    polymer_ecfp,
    search_tools,
)


class BigFingerprintApiTest(unittest.TestCase):
    def test_standard_fingerprint_is_silent_and_has_expected_format(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            vector = generate_bigfingerprint_vector(
                "{[][<]C(=O)CCC(=O)[<],[>]OCCCO[>][]}",
                pecfp_radius=0,
            )
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(vector.shape, (2088,))
        self.assertEqual(vector.dtype, np.uint8)
        self.assertTrue(np.isin(vector, [0, 1]).all())

    def test_radii_greater_than_two_are_rejected_before_preprocessing(self):
        with patch("bigfingerprint.bigfingerprint.preprocess_bigsmiles") as preprocess:
            for radius in (3, 4, 100):
                with self.subTest(radius=radius):
                    with self.assertRaisesRegex(ValueError, "at most 2"):
                        generate_bigfingerprint_vector(
                            "{[][<]CCO[>][]}", pecfp_radius=radius
                        )
            preprocess.assert_not_called()

    def test_core_rejects_radii_greater_than_two_before_reading_graph(self):
        for radius in (3, 4, 100):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "at most 2"):
                    polymer_ecfp.ECFP(
                        {}, [], radius=radius, print_all=False
                    )

    def test_figure_one_supported_radii_keep_reference_fingerprints(self):
        bigsmiles = "CO{[>][<]CCO[>],[<]CC(C)O[>][<]}C"
        # Full-vector on-bit indices captured from unmodified GitHub commit 4195169.
        expected_bits = {
            0: [274, 347, 1242, 1751, 2048, 2069],
            1: [
                155,
                184,
                274,
                281,
                347,
                528,
                782,
                1041,
                1155,
                1242,
                1317,
                1751,
                1850,
                2024,
                2048,
                2069,
            ],
            2: [
                87,
                136,
                138,
                155,
                184,
                228,
                274,
                281,
                305,
                347,
                376,
                385,
                426,
                484,
                528,
                702,
                771,
                782,
                835,
                862,
                906,
                923,
                1041,
                1155,
                1242,
                1268,
                1317,
                1400,
                1448,
                1561,
                1751,
                1850,
                1951,
                2024,
                2048,
                2069,
            ],
        }
        graphs = search_tools.generate_NetworkX_graphs(bigsmiles)
        _, clusters, _ = search_tools.identify_cycles(graphs)
        for radius, bits in expected_bits.items():
            with self.subTest(radius=radius):
                expected = np.zeros(2088, dtype=np.uint8)
                expected[bits] = 1
                vector = generate_bigfingerprint_vector(
                    bigsmiles, pecfp_radius=radius
                )
                np.testing.assert_array_equal(vector, expected)
                core = polymer_ecfp.ECFP(
                    graphs, clusters, radius=radius, print_all=False
                )
                np.testing.assert_array_equal(core, expected[:2048])
        np.testing.assert_array_equal(
            generate_bigfingerprint_vector(bigsmiles), expected
        )

    def test_negative_radius_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            generate_bigfingerprint_vector(
                "{[][<]CCO[>][]}",
                pecfp_radius=-1,
            )

    def test_multiple_h_candidates_are_combined_by_bitwise_or(self):
        bigsmiles = "{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"
        candidate_vectors = [
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([0, 1, 1], dtype=np.uint8),
        ]
        with (
            patch(
                "bigfingerprint.bigfingerprint._generate_pecfp_vector",
                side_effect=candidate_vectors,
            ) as generate,
            patch(
                "bigfingerprint.bigfingerprint.polymer_topfp.bigsmiles_topology_vector",
                return_value=np.array([1, 0], dtype=np.uint8),
            ) as topology,
        ):
            vector = generate_bigfingerprint_vector(
                bigsmiles,
                pecfp_radius=2,
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
