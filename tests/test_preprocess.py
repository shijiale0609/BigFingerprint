import unittest

from bigfingerprint import add_missing_h_endgroups, preprocess_bigsmiles


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
        bigsmiles = "{[>][<]CCO[>], [<]CC(C)O[>][<]}CC"
        self.assertEqual(add_missing_h_endgroups(bigsmiles), "[H]" + bigsmiles)
        self.assertEqual(
            preprocess_bigsmiles(bigsmiles),
            [
                "CCO{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
                "CC(C)O{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
            ],
        )

    def test_closed_object_terminal_does_not_get_h(self):
        bigsmiles = "CCC(C){[$][$]CC(c1ccccc1)[$][$]}{[$][$]CC(C)[$][]}"  # closed
        self.assertEqual(add_missing_h_endgroups(bigsmiles), bigsmiles)
        self.assertEqual(preprocess_bigsmiles(bigsmiles), [bigsmiles])

    def test_open_object_terminal_gets_h(self):
        bigsmiles = "CCC(C){[$][$]CC(c1ccccc1)[$][$]}{[$][$]CC(C)[$][$]}"
        self.assertEqual(add_missing_h_endgroups(bigsmiles), bigsmiles + "[H]")

    def test_directional_repeat_with_nested_object_is_preserved(self):
        bigsmiles = (
            "COCCO{[>][<]C(O{[<][>]CCO[<][>]}C)CCCCCO[>],"
            "[<]C(=O)CCCCCO[>][<]}"
        )
        self.assertEqual(
            preprocess_bigsmiles(bigsmiles),
            [
                bigsmiles + "C(O{[<][>]CCO[<][>]}C)CCCCCO",
                bigsmiles + "C(=O)CCCCCO",
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

    def test_unbalanced_braces_raise(self):
        with self.assertRaisesRegex(ValueError, "Unmatched opening brace"):
            preprocess_bigsmiles("{[>][<]CCO[>][<]")


if __name__ == "__main__":
    unittest.main()
