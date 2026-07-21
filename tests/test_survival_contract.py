from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from helpers import load_stage
from statistical_contracts import genotype_group, replace_zero_follow_up


class SurvivalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = load_stage("22_curated_survival.py", "stage22_for_tests")

    def test_four_genotype_encoding(self):
        groups = genotype_group([0, 1, 0, 1], [0, 0, 1, 1])
        self.assertEqual(groups.tolist(), ["A−/B−", "A only", "B only", "A+B"])

    def test_zero_time_replacement_is_limited_to_exact_zero(self):
        epsilon = self.stage.ZERO_TIME_EPSILON_MONTHS
        months, zero = replace_zero_follow_up([0.0, 1.0, np.nan, -1.0], epsilon)
        self.assertTrue(zero.tolist() == [True, False, False, False])
        self.assertAlmostEqual(months[0], epsilon)
        self.assertEqual(months[1], 1.0)
        self.assertEqual(months[3], -1.0)

    def test_primary_context_excludes_zero_time_but_sensitivity_reincludes_it(self):
        n_positive = 12
        sample_ids = [f"S{i:02d}" for i in range(n_positive)] + ["S-ZERO"]
        months = list(np.arange(1, n_positive + 1, dtype=float)) + [0.0]
        events = [1.0, 1.0, 1.0, 1.0] + [0.0] * (n_positive - 4) + [1.0]
        mut_a = [0, 1, 0, 1] * 3 + [1]
        mut_b = [0, 0, 1, 1] * 3 + [1]
        clinical = pd.DataFrame(
            {
                "studyId": "study_a",
                "sampleId": sample_ids,
                "patientId": [f"P{i:02d}" for i in range(len(sample_ids))],
                "patientKey": [f"study_a::P{i:02d}" for i in range(len(sample_ids))],
                "broadCancerCode": "LUAD",
                "analysisCancerCode": "LUAD",
                "months": months,
                "event": events,
                "validOsNonnegative": True,
                "validPositiveOs": [True] * n_positive + [False],
                "zeroOsTime": [False] * n_positive + [True],
                "negativeOsTime": False,
            }
        )
        flags = pd.DataFrame(
            {
                "studyId": "study_a",
                "sampleId": sample_ids,
                "callable_GENE1": True,
                "callable_GENE2": True,
                "mut_GENE1": mut_a,
                "mut_GENE2": mut_b,
            }
        )
        primary, sensitivity, audit, strata = self.stage.context_data(
            clinical, flags, "LUAD", "GENE1", "GENE2"
        )
        self.assertEqual(len(primary), n_positive)
        self.assertFalse(primary.zeroOsTime.any())
        self.assertEqual(len(sensitivity), n_positive + 1)
        zero_row = sensitivity.loc[sensitivity.zeroTimeOriginal].iloc[0]
        self.assertAlmostEqual(zero_row.months, self.stage.ZERO_TIME_EPSILON_MONTHS)
        self.assertEqual(zero_row.group, "A+B")
        self.assertEqual(audit["nZeroTimesReincludedSensitivity"], 1)
        self.assertTrue(bool(strata.retainedPrimary.iloc[0]))


if __name__ == "__main__":
    unittest.main()
