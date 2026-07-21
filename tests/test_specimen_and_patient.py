from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from helpers import load_stage


class SpecimenIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = load_stage("01b_dedupe_samples.py", "stage01b_for_tests")

    def test_global_and_reviewed_identifiers_collapse_but_generic_collisions_do_not(self):
        samples = pd.DataFrame(
            [
                ("TCGA-AB-0001", "luad_tcga"),
                ("TCGA-AB-0001", "luad_tcga_pan_can_atlas_2018"),
                ("GENERIC-1", "study_a"),
                ("GENERIC-1", "study_c"),
                ("GENERIC-2", "study_a"),
                ("GENERIC-2", "study_b"),
            ],
            columns=["sourceSampleId", "studyId"],
        )
        whitelist = {frozenset(("study_a", "study_b"))}
        identity = self.stage.assign_specimen_identities(samples, whitelist)
        tcga = identity.loc[identity.sourceSampleId.eq("TCGA-AB-0001")]
        unresolved = identity.loc[identity.sourceSampleId.eq("GENERIC-1")]
        reviewed = identity.loc[identity.sourceSampleId.eq("GENERIC-2")]
        self.assertEqual(tcga.specimenIdentityKey.nunique(), 1)
        self.assertEqual(unresolved.specimenIdentityKey.nunique(), 2)
        self.assertEqual(reviewed.specimenIdentityKey.nunique(), 1)
        self.assertTrue(reviewed.specimenIdentityRule.eq("reviewed cross-study overlap").all())
        self.assertTrue(
            unresolved.specimenIdentityRule.eq(
                "unresolved generic collision preserved study-qualified"
            ).all()
        )

    def test_duplicate_bare_identifiers_are_qualified_in_analysis_ids(self):
        winners = pd.DataFrame(
            {
                "sourceSampleId": ["SAMPLE-1", "SAMPLE-1", "SAMPLE-2"],
                "studyId": ["study_a", "study_b", "study_a"],
            }
        )
        ids = self.stage.analysis_ids_for_winners(winners)
        self.assertEqual(ids.tolist(), ["study_a::SAMPLE-1", "study_b::SAMPLE-1", "SAMPLE-2"])
        self.assertFalse(ids.duplicated().any())


class PatientSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = load_stage("01d_curate_cohort.py", "stage01d_for_tests")

    def test_primary_precedes_metastasis_and_liquid_biopsy_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp)
            original = self.stage.PROCESSED
            self.stage.PROCESSED = processed
            try:
                pd.DataFrame(
                    {"studyId": ["study_a"], "name": ["Adult solid tumour cohort"]}
                ).to_csv(processed / "cohort_studies.csv", index=False)
                meta = pd.DataFrame(
                    [
                        {
                            "sampleId": "P1-MET",
                            "studyId": "study_a",
                            "personKey": "study_a::P1",
                            "patientId": "P1",
                            "cancerFamilyCode": "LUAD",
                            "sampleType": "Metastatic tumour",
                            "sampleClass": "Tumour",
                            "specimenPreservationType": "FFPE",
                        },
                        {
                            "sampleId": "P1-PRI",
                            "studyId": "study_a",
                            "personKey": "study_a::P1",
                            "patientId": "P1",
                            "cancerFamilyCode": "LUAD",
                            "sampleType": "Primary tumour",
                            "sampleClass": "Tumour",
                            "specimenPreservationType": "FFPE",
                        },
                        {
                            "sampleId": "P2-PLASMA",
                            "studyId": "study_a",
                            "personKey": "study_a::P2",
                            "patientId": "P2",
                            "cancerFamilyCode": "LUAD",
                            "sampleType": "ctDNA",
                            "sampleClass": "Liquid biopsy",
                            "specimenPreservationType": "Plasma",
                        },
                    ]
                )
                selected = self.stage.build_curated_samples(meta).set_index("sampleId")
                self.assertTrue(bool(selected.loc["P1-PRI", "analysisEligible"]))
                self.assertFalse(bool(selected.loc["P1-MET", "analysisEligible"]))
                self.assertFalse(bool(selected.loc["P2-PLASMA", "tissueEligible"]))
                self.assertEqual(int(selected.analysisEligible.sum()), 1)
            finally:
                self.stage.PROCESSED = original


if __name__ == "__main__":
    unittest.main()
