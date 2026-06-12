import unittest

from bigfingerprint_preprocess import add_missing_h_endgroups, preprocess_bigsmiles


class BigFingerprintPreprocessTest(unittest.TestCase):
    def test_right_h_directional_expansion(self):
        self.assertEqual(
            preprocess_bigsmiles("OCCO{[>][<]CCO[>], [<]CC(C)O[>][<]}[H]"),
            [
                "OCCO{[>][<]CCO[>], [<]CC(C)O[>][<]}CCO",
                "OCCO{[>][<]CCO[>], [<]CC(C)O[>][<]}CC(C)O",
            ],
        )

    def test_left_h_directional_expansion(self):
        self.assertEqual(
            preprocess_bigsmiles("[H]{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"),
            [
                "CCO{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
                "CC(C)O{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
            ],
        )

    def test_missing_left_h_is_added_then_expanded(self):
        self.assertEqual(
            add_missing_h_endgroups("{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"),
            "[H]{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
        )
        self.assertEqual(
            preprocess_bigsmiles("{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"),
            [
                "CCO{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
                "CC(C)O{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
            ],
        )

    def test_symmetric_dollar_single_repeat_two_h_sites(self):
        self.assertEqual(
            preprocess_bigsmiles("[H]{[$][$]CC(C)[$][$]}[H]"),
            [
                "CCC{[$][$]CC(C)[$][$]}CCC",
                "CCC{[$][$]CC(C)[$][$]}C(C)C",
                "CC(C){[$][$]CC(C)[$][$]}CCC",
                "CC(C){[$][$]CC(C)[$][$]}C(C)C",
            ],
        )

    def test_symmetric_dollar_two_repeats_two_h_sites(self):
        self.assertEqual(
            preprocess_bigsmiles("[H]{[$][$]CC(C)[$],[$]CC(CC)[$][$]}[H]"),
            [
                "CCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCC",
                "CCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)C",
                "CCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCCC",
                "CCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)CC",
                "CC(C){[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCC",
                "CC(C){[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)C",
                "CC(C){[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCCC",
                "CC(C){[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)CC",
                "CCCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCC",
                "CCCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)C",
                "CCCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCCC",
                "CCCC{[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)CC",
                "CC(CC){[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCC",
                "CC(CC){[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)C",
                "CC(CC){[$][$]CC(C)[$],[$]CC(CC)[$][$]}CCCC",
                "CC(CC){[$][$]CC(C)[$],[$]CC(CC)[$][$]}C(C)CC",
            ],
        )


if __name__ == "__main__":
    unittest.main()
