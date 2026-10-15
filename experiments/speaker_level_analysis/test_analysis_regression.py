"""Regression coverage for the analysis-only provenance validation correction."""

import copy
import json
import unittest

import analyze_speakers as analysis


class CanonicalFoldFingerprintRegression(unittest.TestCase):
    def test_native_outer_evidence_accepts_original_fold_fingerprint(self):
        before = analysis.read(analysis.OUT / "preservation_before.json")
        rows, provenance, _, _ = analysis.validate_sources(before["original_files"])
        self.assertEqual(len(rows), 17346)
        self.assertEqual(len(provenance), 6)
        self.assertTrue(all(p["development_test_speakers_disjoint"] for p in provenance))

    def test_fold_fingerprint_ignores_serialization_whitespace(self):
        folds = analysis.read(analysis.ORIGINAL / "data/splits/speaker_folds.json")
        differently_serialized = json.loads(json.dumps(folds, indent=7, sort_keys=False))
        self.assertEqual(analysis.folds_fingerprint(differently_serialized),
                         "5ab4e1af6c54f97af940c3e5ba5d447c77d827102b88656bf19cd5227b5fee27")
        self.assertNotEqual(analysis.folds_fingerprint(folds),
                            analysis.sha(analysis.ORIGINAL / "data/splits/speaker_folds.json"))

    def test_fold_fingerprint_detects_changed_speaker_assignment(self):
        folds = analysis.read(analysis.ORIGINAL / "data/splits/speaker_folds.json")
        changed = copy.deepcopy(folds)
        changed["speaker_to_evaluation_fold"]["F03"] = 2
        self.assertNotEqual(analysis.folds_fingerprint(folds), analysis.folds_fingerprint(changed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
