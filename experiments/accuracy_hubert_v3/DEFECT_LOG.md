# V3 implementation defect log — 2026-10-13

All changes are confined to V3. No inherited source was patched.

| ID | Finding | Correction and regression evidence | Status |
|---|---|---|---|
| V3-D001 | First synthetic end-to-end fixture relocated the output root without its required lineage file; 1/50 tests errored before study execution. | Fixture copies the lineage into its synthetic root; end-to-end selection/freeze/report/resume test now passes. | CLOSED |
| V3-D002 | A changed completed-fit JSON could have supplied altered metrics even though its model checkpoint checksum was valid. | Compare all completion metadata against the saved checkpoint result; test_completed_fit_metrics_tampering_rejected_on_resume. | CLOSED |
| V3-D003 | Augmented payload hashes alone did not bind crop/waveform/diagnostic metadata. | Completion metadata self-checksum plus payload/per-layer checksums; test_augmented_metadata_tampering_is_rejected. | CLOSED |
| V3-D004 | An empty extractor output could leave an OOM-helper loop without progress. | Enforce exact batch count/shape/dtype before advancing; test_empty_extractor_output_stops_instead_of_looping. | CLOSED |
| V3-D005 | Reusing development-report wording in a completed outer report could incorrectly say accuracy was still unknown. | Use phase-neutral development-evidence wording and explicit separate outer report; end-to-end report assertion. | CLOSED |
| V3-D006 | Decisions needed a byte-level seal in addition to model/config checks; interruption between manifest and seal publication needed recovery. | Validate decision seal before outer access; recover missing publication only when no outer claim exists; end-to-end tamper and interrupted-publication regressions. | CLOSED |
| V3-D007 | A completed development resume could regenerate timing receipts and alter frozen metadata. | Reuse and verify the sealed decision set before repeating any selection; end-to-end test forbids select_fold on terminal resume. | CLOSED |
| V3-D008 | The new terminal-resume regression exposed CSV column-order drift after sorted-key JSON reload. | Canonical sorted CSV columns; test_csv_serialization_stable_across_frozen_json_key_sorting and full synthetic resume. The failing log is retained as validation/tests_final.log. | CLOSED |

Numerical corruption, OOM and interrupted outer claims in unit tests are deliberate
synthetic fault injection. No real numerical corruption or real outer claim occurred.
No unresolved implementation defect is known. Full-size runtime and scientific
performance remain unvalidated, not silently counted as completed tests.
