from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from helpers import load_stage
from statistical_contracts import benjamini_hochberg
from cbioportal_client import _assert_complete_page


class MultipleTestingTests(unittest.TestCase):
    def test_bh_values_and_missing_positions(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002, np.nan])
        np.testing.assert_allclose(adjusted[:4], [0.02, 0.04, 0.04, 0.008])
        self.assertTrue(np.isnan(adjusted[4]))

    def test_invalid_p_value_is_rejected(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.2, 1.1])


class CmhTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = load_stage("17_curated_interactions.py", "stage17_for_tests")

    @staticmethod
    def stratum(a: int, b: int, c: int, d: int) -> np.ndarray:
        rows = [[1, 1]] * a + [[1, 0]] * b + [[0, 1]] * c + [[0, 0]] * d
        return np.asarray(rows, dtype=np.int8)

    def test_mantel_haenszel_common_odds_ratio(self):
        accumulator = self.stage._new_accumulator(2)
        covered = np.ones(2, dtype=bool)
        self.stage._accumulate_stratum(accumulator, self.stratum(4, 1, 2, 3), covered)
        self.stage._accumulate_stratum(accumulator, self.stratum(2, 3, 1, 4), covered)
        odds, low, high, p_value, n_strata, n_informative = (
            self.stage._finish_accumulator(accumulator)
        )
        self.assertAlmostEqual(float(odds[0, 1]), 4.0, places=12)
        self.assertLess(float(low[0, 1]), 4.0)
        self.assertGreater(float(high[0, 1]), 4.0)
        self.assertTrue(np.isfinite(p_value[0, 1]))
        self.assertEqual(int(n_strata[0, 1]), 2)
        self.assertEqual(int(n_informative[0, 1]), 2)

    def test_leave_two_out_burden_removes_both_tested_genes(self):
        info = pd.DataFrame(
            {
                "analysisCancerCode": "LUAD",
                "studyId": "study_a",
                "panelStratum": "WES/WGS",
                "mutationBurden": [2, 2, 2, 2],
            }
        )
        design = self.stage._leave_two_out_design(info)
        strata, stratum_is_wes = self.stage._leave_two_out_pair_strata(
            np.array([0, 1, 0, 1], dtype=np.int8),
            np.array([0, 0, 1, 1], dtype=np.int8),
            design,
        )
        # Background burdens are 2, 1, 1 and 0. Their tied average ranks place
        # them in quintile indices 4, 3, 3 and 1, respectively.
        np.testing.assert_array_equal(strata, [4, 3, 3, 1])
        self.assertTrue(stratum_is_wes.all())


class ApiResponseContractTests(unittest.TestCase):
    def test_get_page_below_boundary_is_accepted(self):
        _assert_complete_page("GET", {"pageSize": 10}, [{"id": 1}] * 9)

    def test_get_page_at_boundary_is_rejected(self):
        with self.assertRaises(RuntimeError):
            _assert_complete_page("GET", {"pageSize": 10}, [{"id": 1}] * 10)


if __name__ == "__main__":
    unittest.main()
