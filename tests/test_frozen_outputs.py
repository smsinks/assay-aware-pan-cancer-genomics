from __future__ import annotations

import unittest

from helpers import SRC
from result_summary import headline_results, preflight
from verify_results_manifest import verify_manifest


class FrozenOutputTests(unittest.TestCase):
    def test_inventory_and_integrity(self):
        checks = preflight()
        self.assertTrue(checks["ready"], checks["missing"])
        self.assertEqual(checks["tables"], 43)
        self.assertEqual(checks["main_figures"], 9)
        self.assertEqual(checks["supplementary_figures"], 7)
        manifest = verify_manifest()
        self.assertTrue(manifest["ok"], manifest["failures"])

    def test_headline_sentinels(self):
        results = headline_results()
        self.assertEqual(results["cohort"]["selected_tissue_tumours"], 132_181)
        self.assertEqual(results["cohort"]["contributing_studies"], 367)
        self.assertAlmostEqual(results["gene_prevalence_pct"]["TP53"], 37.42)
        self.assertAlmostEqual(
            results["luad_conditioned_odds_ratios"]["primary_no_burden"][
                "EGFR-KRAS"
            ],
            0.02773275699,
        )
        self.assertAlmostEqual(
            results["luad_conditioned_odds_ratios"]["primary_no_burden"][
                "KEAP1-STK11"
            ],
            10.8291866,
        )
        self.assertEqual(results["pairwise_screen"]["Jointly estimable"], 71_360)
        self.assertEqual(
            results["pairwise_screen"]["+ sensitivity assay concordance"], 1_540
        )
        self.assertEqual(
            results["survival"]["complete_screen_joint_state_models"], 2_612
        )
        self.assertEqual(
            results["survival"]["expanded_diagnostic_joint_state_models"], 29
        )
        self.assertEqual(
            results["survival"]["primary_joint_state_ph_p_below_0_05"], 10
        )
        self.assertEqual(results["assay_scope_audit"]["off_panel_mutation_records"], 2_508)


if __name__ == "__main__":
    unittest.main()
