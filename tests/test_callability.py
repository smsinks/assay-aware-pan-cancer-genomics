from __future__ import annotations

import unittest

import pandas as pd

from helpers import load_stage


class AssayCallabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.callability = load_stage("callability.py", "callability_for_tests")

    def test_wes_is_genome_wide_and_panel_calls_require_membership(self):
        mutations = pd.DataFrame(
            {
                "sampleId": ["WES-1", "PANEL-1", "PANEL-1"],
                "studyId": ["study", "study", "study"],
                "entrezGeneId": [20, 10, 20],
                "hugoSymbol": ["GENE20", "GENE10", "GENE20"],
            }
        )
        assay = pd.DataFrame(
            {
                "sampleId": ["WES-1", "PANEL-1"],
                "studyId": ["study", "study"],
                "genePanelId": ["WES", "PANEL-A"],
                "assayType": ["WES/WGS documented", "Targeted panel"],
                "panelMetadataAvailable": [True, True],
            }
        )
        membership = pd.DataFrame(
            {"genePanelId": ["PANEL-A"], "entrezGeneId": [10]}
        )
        kept, conflicts = self.callability.partition_callable_mutations(
            mutations, assay=assay, membership=membership
        )
        self.assertEqual(set(zip(kept.sampleId, kept.entrezGeneId)), {("WES-1", 20), ("PANEL-1", 10)})
        self.assertEqual(set(zip(conflicts.sampleId, conflicts.entrezGeneId)), {("PANEL-1", 20)})
        self.assertIn("absent from documented panel", conflicts.callabilityConflictReason.iloc[0])


if __name__ == "__main__":
    unittest.main()
