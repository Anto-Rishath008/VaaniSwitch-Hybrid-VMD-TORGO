# Defect Log

| ID | Module | Severity | Status | Description | Reproduction | Root cause | Fix | Regression test |
|---|---:|---|---|---|---|---|---|---|
| M6-CONFIG-001 | 6 | Low | CLOSED 2026-10-08 | Malformed execution mapping raised AttributeError instead of explicit config rejection. | Three failing None/list/string regression cases in `artifacts/module6/setup/regression_malformed_before.log`. | Nested mapping was used before type validation. | Validate execution as a mapping before policy comparison. | `test_malformed_execution_fails_cleanly`; final M6 suite PASS. |
| M6-TTS-001 | 6 | High | CLOSED 2026-10-08 | First real Zira save wrote WAV bytes but the following event loop timed out after 60 seconds. | Preserved failed Phase-1 run and failing `test_sapi_save_is_queued_before_run_loop`. | pyttsx3 2.99 leaves its proxy idle after settings-only runAndWait; synchronous save then completes before the next loop starts. | Flush settings through public external-loop APIs, end that loop to restore the queue barrier, verify actual voice/settings, then save_to_file plus runAndWait. Installed package unchanged. | Synthetic lifecycle regression PASS; real TTS-only retry PASS, 80,675 PCM frames. Original successful ASR reused without another forward. |
| M6-IO-001 | 6 | Medium | CLOSED 2026-10-08 | TTS recovery could not reload UTF-8 JSON under Windows CP1252. | Actual UnicodeDecodeError before worker execution; failing forced-CP1252 synthetic retained-ASR test. | JSON read_text omitted encoding despite Unicode punctuation in frozen options. | Parse persisted JSON bytes so JSON applies UTF-8 decoding independently of Windows locale. | `test_resume_unicode_and_hash_validation`; TTS recovery and explicit runtime evidence test PASS. |
| M6-REF-001 | 6 | High | CLOSED 2026-10-08 | CORE membership alone is insufficient for ASR references. | Phase-0 audit found image filenames and sense annotations. | Classification and spoken-reference eligibility differ. | M6-only exclusions, exact supported sense rule, source-byte retention and signed CORE consumption; 8562 eligible targets verified. | Reference contract/composite-join/exclusion tests and recovered 9494-record inventory equality PASS. |
| M0-DATA-001 | 0 | High | ACCEPTED_LIMITATION | Four prompt records have neither microphone recording. | Build Module 0 and inspect logical exclusions for F03/S2/0117, M04/S1/0066, MC04/S1/0655, MC04/S1/0656. | Source TORGO files are absent. | Preserve records, set `excluded`, and record `missing_primary_audio`; do not synthesize audio. | `test_neither_microphone_excludes_logical_utterance` |
| M0-DATA-002 | 0 | High | ACCEPTED_LIMITATION | F01/S1 heads 0067 and 0068 are zero-byte while valid array recordings exist. | Inspect the two head rows in the audio-quality CSV. | The source head recordings are empty. | Preserve both invalid heads as audited alternates and use the structurally valid arrays as primary audio; retain both logical records. | `test_zero_byte_head_uses_valid_array_fallback_and_is_retained`; `test_invalid_head_remains_in_full_audio_quality_audit` |
| M0-DATA-003 | 0 | Medium | ACCEPTED_LIMITATION | 381 direct microphone paths do not exactly map to prompt metadata; four filenames/IDs are malformed. | Inspect orphan rows in `module0_exclusions.csv`. | TORGO contains orphan and malformed direct microphone files. | Discover from prompt files only; audit and log orphans without inventing logical utterances. | `test_discovery_is_prompt_driven_and_logs_orphan_audio`; `test_literal_dot_wav_filename_is_audited_as_malformed_orphan` |
| M0-DATA-004 | 0 | Medium | ACCEPTED_LIMITATION | Three byte-identical WAV hash groups exist. | Inspect `raw_audio_duplicate_audit.duplicate_hash_groups`. | Two same-speaker alternate pairs duplicate content; the third is the zero-byte F01 pair. | Report all groups and hard-fail cross-speaker/cross-fold duplicates; all observed groups remain group-safe. | `test_same_speaker_duplicate_hash_is_reported_but_group_safe`; `test_distinct_cross_speaker_paths_with_same_hash_fail` |
| M0-CODE-001 | 0 | High | CLOSED | A literal filename `.wav` was initially skipped because `Path.suffix` is empty. | Create a direct mic file named `.wav`. | Suffix-based discovery did not cover this legal literal name. | Match case-insensitive filename endings and classify its empty stem as malformed. | `test_literal_dot_wav_filename_is_audited_as_malformed_orphan` |
| M0-CODE-002 | 0 | High | CLOSED | Early output validation trusted stale reports and did not validate every artifact. | Remove an output or tamper the fold map/leakage status/manifest CSV. | Cross-artifact presence, parity, and source fingerprints were incomplete. | Validate all required outputs, JSONL/CSV parity, raw/source fingerprints, fold maps, leakage internals, CSV contexts, and totals. | `test_tiny_pipeline_writes_all_outputs_without_mutating_raw`; `test_tampered_speaker_to_fold_map_fails_loudly`; `test_internally_inconsistent_pass_report_fails_loudly` |
| M0-CODE-003 | 0 | High | CLOSED | Loaded manifest JSON could coerce types or accept tampered TORGO labels and incomplete usable audio. | Set `normal_speech_eligible` to `"false"`, relabel F as control, or clear usable header fields. | Schema loader used permissive coercion and lacked scientific cross-field invariants. | Require exact JSON types/fields, canonical group labels, transcript exclusions, readable metadata, duration consistency, and SHA-256 for non-excluded audio. | `test_manifest_json_schema_does_not_coerce_boolean_strings`; `test_manifest_schema_rejects_tampered_canonical_labels`; `test_manifest_schema_rejects_usable_record_without_readable_metadata` |
| M0-CODE-004 | 0 | Medium | CLOSED | Speaker-group summary and orphan-only session coverage could be miscounted. | Build a session containing only an orphan WAV or compare `speakers.by_group` with unique speakers. | Aggregation counted utterances and seeded sessions only from prompt records. | Deduplicate speaker labels and seed coverage from prompt, direct-audio, and orphan contexts. | `test_orphan_only_session_is_included_in_microphone_coverage`; `test_tiny_pipeline_writes_all_outputs_without_mutating_raw` |
| M0-CODE-005 | 0 | Medium | CLOSED | Hash-read failure or a truncated WAV payload could terminate or be misreported. | Inject a hash `OSError` or truncate a valid WAV payload. | Error handling did not cover hashing and header frame counts were not checked against decoded samples. | Return explicit unreadable audit records and validate expected sample counts. | `test_hash_read_failure_is_reported_not_raised`; `test_truncated_wav_payload_is_reported_as_unreadable` |
| M0-CODE-006 | 0 | High | CLOSED | Path-only head priority excluded recoverable prompts when the head existed but was structurally unusable. | Provide a zero-byte, corrupt, or empty-frame head beside a valid array recording. | Primary selection occurred before both microphone recordings had structural audit evidence, and the schema required any existing head path to be primary. | Audit both microphones first; require a valid head when available; allow array fallback only with recorded head failure status, issues, selection reason, and quality flags; cross-check that evidence during output validation. | `test_zero_byte_head_uses_valid_array_fallback_and_is_retained`; `test_corrupt_head_uses_valid_array_fallback`; `test_empty_frame_head_uses_valid_array_fallback`; `test_schema_rejects_arbitrary_array_primary_when_head_is_valid`; `test_schema_requires_unusable_head_evidence_for_array_fallback` |
| M1-SCI-001 | 1 | High | ACCEPTED_LIMITATION | Initial tau=0 all-mode reconstruction leaves a normalized residual of 0.0813–0.1771 in the six-record smoke run, including speech-band energy. | Run the signed six-record smoke build and inspect reconstruction metrics/residual spectra. | With tau=0, the VMD dual reconstruction constraint is inactive; update convergence does not imply exact signal reconstruction. | Preserve all modes, report metrics descriptively, avoid denoising/losslessness claims, and defer parameter comparison to the authorized optimization stage. | `test_all_mode_reconstruction_is_close_for_locked_multitone`; real smoke cache revalidation |
| M1-OPS-001 | 1 | Medium | ACCEPTED_LIMITATION | Full CORE float64 arrays require a nominal 18.57 GiB before metadata and filesystem overhead. | Sum CORE sample counts and multiply by six arrays and eight bytes. | Caching five lossless float64 modes plus one float64 reconstruction is storage-intensive. | Report the exact estimate and require a manual free-space check before the full build. | `test_pipeline_build_validate_cache_skip_and_raw_immutability` |
| M1-CODE-001 | 1 | High | CLOSED | Misconfigured Module 1 output paths could write under raw TORGO or overwrite signed Module 0 inputs. | Point cache/log/summary output paths into the raw root or at a Module 0 input. | Early configuration validation only required repository containment. | Require cache under `data/processed`, reports under `artifacts`, raw disjointness, and explicit Module 0 input protection before any write. | `test_config_rejects_outputs_inside_raw_root`; `test_config_rejects_module0_input_overwrite` |
| M1-CODE-002 | 1 | High | CLOSED | A structurally complete cache with missing or stale scientific metadata could be treated as a resume hit. | Tamper nested metadata while keeping generic artifact hashes and completion marker valid. | Generic cache validation did not enforce pipeline scientific semantics. | Apply semantic identity/provenance/parameter/metric validation before cache skip and during final validation; recompute invalid entries. | `test_semantically_invalid_cache_is_recomputed` |
| M1-CODE-003 | 1 | Medium | CLOSED | Failed builds and selected-mode ablations could receive misleading all-mode/success labels. | Force a record failure or enable the explicit selected-mode test configuration. | Summary labels were static and failure status was assigned after summary construction. | Derive status and scientific labels from actual completion and reconstruction policy. | `test_failed_build_summary_is_not_marked_validated`; `test_selected_mode_ablation_has_explicit_scientific_label` |
| M1-CODE-004 | 1 | Medium | CLOSED | Diagnostic overlays could obscure reconstruction attenuation if traces were independently normalized. | Plot an input and a half-amplitude reconstruction. | Per-trace plotting scale can erase relative-amplitude differences. | Pass a shared input-derived scale to both overlay traces and lock the behavior with an amplitude-excursion regression. | `test_overlay_traces_preserve_relative_amplitude_with_shared_scale` |
| M1-SCI-002 | 1 | High | ACCEPTED_LIMITATION | Nonzero tau improves aggregate all-mode fidelity but is unstable at the fixed 500-iteration budget. | Run the fixed 12-record tau sweep at 0, 0.1, 0.25, and 0.5. | Dual-ascent enforcement improves reconstruction but most real utterances do not reach the 1e-7 update tolerance within 500 iterations; runtime rises 2.67–3.13×. | Retain tau=0 for the full build, preserve the complete numerical report, and do not select from classification accuracy. | `artifacts/metrics/module1_tau_fidelity_sweep.json`; `artifacts/logs/module1_tau_fidelity_per_record.csv` |
| M1-SCI-003 | 1 | Medium | ACCEPTED_LIMITATION | Forty-eight of 8,673 production CORE records reached the 500-iteration ceiling before satisfying the 1e-7 VMD update tolerance. | Inspect `module1_nonconverged_records.csv` and run the full Module 1 validator. | The configured tau=0 bandwidth-regularized solver reached its iteration budget on 48 utterances; solver-update convergence is distinct from reconstruction validity. | Retain all 48 records without recomputation or exclusion: each cache is finite, correctly shaped, frequency-bounded, source-bound, an exact all-mode sum, and reconstruction-metric-valid. | `artifacts/metrics/module1_full_signoff_audit.json`; `artifacts/logs/module1_nonconverged_records.csv`; full `vaaniswitch-m1 validate` PASS |
| M2-001 | 2 | Medium | CLOSED | The proposed 128-band HTK Mel bank contained one empty low-frequency triangle at the production 512-point FFT resolution. | Construct the 128-band 20–8,000 Hz filter bank and check that every row has positive support. | The requested band density exceeded the usable low-frequency resolution of the 257-bin positive spectrum. | Use 80 native Mel bands and explicitly reject any configuration that creates an empty triangle; retain deterministic 80-to-224 frequency interpolation. | `test_mel_filterbank_rejects_empty_native_band_configuration`; `test_mel_filterbank_is_nonnegative_and_bounded_to_rfft_bins` |
| M2-SCI-001 | 2 | Medium | ACCEPTED_LIMITATION | The fixed 8-second Mel canvas center-crops 235/8,673 CORE records, while recurrence and visibility use each complete reconstruction. | Audit all CORE durations and regenerate every >8-second cache from the signed full reconstruction; inspect `F03/Session1/0118` at 193.828625 seconds. | A fixed 224-pixel, 8-second physical Mel axis trades long-record coverage against temporal resolution; six records exceed 30 seconds. | Retain the locked policy for initial production, exclude no records, state that cropping can affect retained content, and require Module 5 duration/crop robustness analysis. | `test_mel_explicit_center_crop_uses_128000_samples_and_798_frames`; `test_realistic_80_to_224_resize_preserves_constant_rows`; full 235-record sign-off regeneration audit |

| M4-CODE-001 | 4 | Medium | CLOSED | A checksum-consistent malformed checkpoint escaped controlled rejection; extra optimizer entries could hide unrelated tensors. | Both failing tests reproduced; `artifacts/logs/module4_defect_reproduction.log`. | Restricted unpickler errors and exact optimizer index coverage were not validated. | Catch malformed restricted-load errors as ValueError; require exact optimizer state indices and tensors. | `test_matching_checksum_invalid_pickle_is_controlled_rejection`; `test_optimizer_state_cannot_hide_extra_dataset_arrays`; 96 module / 328 full tests and 84.38% coverage PASS. |
| M4-CODE-002 | 4 | High | CLOSED | A redirected artifact category could resolve under raw data while passing repository containment. | Failing `test_resolved_output_directory_cannot_redirect_to_raw`; reproduction log retained. | Output category resolution checked repository containment but not its canonical artifact location. | Require canonical artifact category location and unchanged output basename before writes. | `test_resolved_output_directory_cannot_redirect_to_raw`; 96 module / 328 full tests and protected-input audit PASS. |
| M4-SCI-001 | 4 | Medium | ACCEPTED_LIMITATION | Deterministic Fold 1 inner validation selects M03 and MC02, both male. | Inspect `module4_split_audit.json` and signed manifest genders. | Two class-stratified validation speakers cannot cover all four condition-by-gender strata. | Retain the metadata-only seed-42 development split and state its limits; no female-generalization or final performance claim; later frozen grouped-fold evaluation belongs to Module 5. | `test_all_outer_folds_deterministic_disjoint_both_classes`; live split audit. |

## Module 4 final audit disposition — 2026-09-13

No new production defect was found. M4-CODE-001 and M4-CODE-002 remain CLOSED with
their regression evidence preserved; M4-SCI-001 remains ACCEPTED_LIMITATION. The completed
baseline and independent validation agree, frozen ImageNet state and protected inputs
pass, and no production failure or reserved-fold forward appears in the retained evidence.
The final manual regression gates were confirmed by the user; the fresh targeted log
records 96 passes and retained implementation full-suite/coverage evidence agrees.
See the current Module 4 TEST_REPORT and `module4_full_signoff_audit.json` for provenance.

## Module 5 implementation findings — 2026-09-14

| ID | Module | Severity | Status | Description | Reproduction | Root cause | Fix | Regression test |
|---|---:|---|---|---|---|---|---|---|
| M5-CODE-001 | 5 | High | CLOSED | A0 accepted an altered record carrying another primary waveform under an existing logical ID. | Initial Module 5 suite: 1 failed / 68 passed; `artifacts/module5_test_initial.log`. | Logical-ID membership alone did not bind the supplied record to the manifest entry. | Require exact signed-record equality before primary loading and cache identity construction. | `test_a0_forged_selected_primary_rejected` |
| M5-CODE-002 | 5 | High | CLOSED | Editing a C2 selection file could change matched ablations or promote a losing candidate during selection recovery. | Workflow suite: 1 failed / 6 passed; `artifacts/module5_test_workflow.log`; later recovery regression reproduced in `artifacts/module5_selection_recovery_reproduction.log`. | Selection-file recovery trusted the file, and ablation preparation omitted equality with its ledger selection. | Recompute the winner from all retained inner candidate results during recovery; require exact ledger selection equality before ablation training. | `test_tampered_selection_cannot_change_ablation_candidate`; `test_selection_recovery_cannot_promote_a_losing_candidate` |
| M5-CODE-003 | 5 | Medium | CLOSED | Ledger transactions left SQLite connections open until garbage collection. | Targeted lifecycle test failed; `artifacts/module5_test_targeted.log`. | SQLite connection context managers commit/rollback but do not close the connection. | Explicitly close each connection in a context-manager finally block. | `test_ledger_connection_closes_after_transaction` |

Module 5 closure: all three fixes pass targeted regression, all 92 Module 5 tests,
all 420 project tests and 84.63% configured branch-aware coverage. Bounded real
smoke and protected-input audits pass. No full experiments or real outer evaluation
ran. Accepted upstream scientific limitations remain documented in Module 5 MODULE.md.

Status: OPEN / FIXING / RETEST / CLOSED / ACCEPTED_LIMITATION

## M5-RECOVERY-001 — interrupted native Fold 1/A1 claim (2026-10-05)

Disposition: AUTHORIZED PROTOCOL EXCEPTION; IMPLEMENTATION READY; NOT EXECUTED.
Native event 135 consumed claim cb66ac287b054a8b8cd3e68d3d8ee385 without a retained
result. A nearby host restart is documented; zero prior forwards cannot be proven.
This is not permission to clear the native claim or rewrite frozen evaluator code.
The external script reserves one attempt and journals durable intent/commit pairs;
ambiguous batches require review. All native evidence is preserved. Forty-eight
mocked tests, Ruff and instrumented metadata preflight pass; no real model forward.
See docs/modules/5_accuracy_robustness/A1_OUTER_RECOVERY_EXCEPTION.md and the retained
forensics/fold1_A1_20261005T060808Z evidence. M5-CODE-001/002/003 remain CLOSED.

## M5-RECOVERY-001 continuation disposition — 2026-10-06

External recovery is now COMPLETE / VERIFIED, provenance RECOVERED_PROTOCOL_EXCEPTION.
The original native claim/state/result absence are preserved; prior native forwards
remain UNKNOWN. This disposition does not relabel A1 as a normal native completion.
The original external preflight intentionally rejects unfinished claims, while
the unchanged native evaluator checks only the requested target's claim.

A separate read-only continuation guard pins this exact verified exception and
allows only ordered native target completion; a second unfinished claim fails.
54 focused synthetic/mocked tests and Ruff pass. The real B1 metadata preflight
passes with actual model calls blocked; zero scientific forwards. Frozen source,
protocol, original external helpers and all native/recovery evidence are unchanged.
Only B1 is handed off. No native evaluator defect was bypassed or fixed.
M5-CODE-001/002/003 remain CLOSED. See POST_RECOVERY_B1_CONTINUATION.md.

## M5-RECOVERY-001 Fold-1 closure disposition — 2026-10-06

Fold 1 panel closure PASS with six NATIVE cells and one
RECOVERED_PROTOCOL_EXCEPTION. A1 remains native FINAL_TRAIN_COMPLETE with its
original consumed claim/event and no native result. Prior native forwards UNKNOWN;
the recovered cell does not satisfy uninterrupted native once-only evaluation.
All original native and external recovery evidence remains preserved.

The separate Fold-2/A0 guard requires the pinned closure and rejects any second
unfinished claim or unauthorized activity. Actual frozen native fold/claim guards
are verified by source inspection and mocked model/data with real temporary SQL.
117 focused tests and Ruff pass; real closure and A0 preflight perform zero model
forwards. Fold 2 has not started; no tuning/aggregation/robustness occurred.
No scientific code defect was bypassed or changed. M5-CODE-001/002/003 remain CLOSED.
See docs/modules/5_accuracy_robustness/FOLD1_CLOSURE_FOLD2_HANDOFF.md.

## M5-RECOVERY-001 preservation during Fold2/A1 handoff — 2026-10-06

Fold2/A0 independently verified and completion evidence frozen. A separate A1-only
guard preserves A0 and the original Fold1/A1 exception; it accepts exactly A1's
legitimate two-event native completion and rejects another unfinished claim.
64 focused tests, Ruff and instrumented A1 preflight pass with zero scientific
forwards/native writes. No frozen source or previously used helper changed.
Fold1/A1 prior native forwards remain UNKNOWN; no strict once-only claim.
No new scientific-code defect was bypassed. M5-CODE-001/002/003 remain CLOSED.
See docs/modules/5_accuracy_robustness/FOLD2_A0_CLOSURE_A1_HANDOFF.md.

## M5-RECOVERY-001 preservation during Fold2/B2 handoff — 2026-10-06

Fold2/A0/A1/B1 independently verified; B1 completion frozen. The new B2-only guard
preserves all earlier evidence and accepts only the legitimate B2 two-event delta.
92 focused tests, Ruff and instrumented real B2 preflight pass. Zero scientific
forwards/native writes. Used A0/A1/B1 helpers and frozen source/protocol unchanged.
Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION with original prior forwards UNKNOWN.
No native/scientific defect bypassed; M5-CODE-001/002/003 remain CLOSED.
See docs/modules/5_accuracy_robustness/FOLD2_B1_CLOSURE_B2_HANDOFF.md.

## M5-RECOVERY-001 preservation during Fold2/B3 handoff — 2026-10-06

Fold2/A0/A1/B1/B2 independently verified; B2 completion frozen. The separate B3
guard permits only its exact two-event native completion. 99 focused tests, Ruff
and instrumented real B3 preflight pass. Zero scientific forwards/native writes.
Used helpers and frozen source/protocol unchanged. Fold1/A1 exceptional provenance
and original prior forwards UNKNOWN preserved. No scientific defect bypassed;
M5-CODE-001/002/003 remain CLOSED.
See docs/modules/5_accuracy_robustness/FOLD2_B2_CLOSURE_B3_HANDOFF.md.

## M5-RECOVERY-001 preservation during Fold2/C1 handoff — 2026-10-07

Fold2/A0/A1/B1/B2/B3 independently verified; B3 completion frozen. The separate C1
guard permits only its exact two-event native completion. 106 focused tests, Ruff
and instrumented real C1 preflight pass. Zero scientific forwards/native writes.
Used helpers and frozen source/protocol unchanged. Fold1/A1 exceptional provenance
and original prior forwards UNKNOWN preserved. No scientific defect bypassed;
M5-CODE-001/002/003 remain CLOSED.
See docs/modules/5_accuracy_robustness/FOLD2_B3_CLOSURE_C1_HANDOFF.md.

## M5-RECOVERY-001 preservation in generic remaining sequence — 2026-10-07

C1 independently verified and frozen. One new external sequence guard preserves
all used helpers, frozen code/protocol/runtime and Fold1/A1 exceptional provenance.
Its original native claim and missing result remain unchanged; prior native
forwards UNKNOWN. No second exception or native defect bypass is authorized.
88 focused tests, Ruff, status and C2 preflight PASS. Append-only external
verification never creates/clears native claims. Actual agent scientific
forwards/native writes ZERO. M5-CODE-001/002/003 remain CLOSED.
See docs/modules/5_accuracy_robustness/FOLD2_C1_CLOSURE_REMAINING_HANDOFF.md.

## Module5 final closure — 2026-10-07

Final outer integrity audit PASS. M5-RECOVERY-001 remains an explicit
RECOVERED_PROTOCOL_EXCEPTION, not a retroactively native-compliant result;
Fold1/A1 original prior native forwards remain UNKNOWN. No claims reset or results replaced.
M5-CODE-001/002/003 remain CLOSED; the 830-test Module5 suite and 27 focused closure
tests pass. The external closure helper leaves frozen scientific code and previously
used helpers unchanged. No unresolved integrity blocker; Module5 signed off with
the retained exception and descriptive limitations. Module6 NOT STARTED.
See docs/modules/5_accuracy_robustness/FINAL_CLOSURE_MODULE6_HANDOFF.md.

## Module 6 recovery findings — 2026-10-08

| ID | Module | Severity | Status | Description / evidence | Disposition / regression |
|---|---:|---|---|---|---|
| M6-RIFF-001 | 6 | Medium | CLOSED | Strict equality conflated source trailing bytes with generated-file validity; failing trailing-data regression retained. | Validate declared source chunk region and complete PCM; record trailing count; keep generated strictness. Six RIFF tests pass. |
| M6-RIFF-002 | 6 | High | ACCEPTED DATASET EXCEPTION 2026-10-08 | Five frozen M01/Session1 outer RIFF headers overdeclare by eight bytes; PCM payload is complete. | User explicitly superseded blanket rejection for these verified source identities only. Dedicated wrapper verifies canonical path, Phase-0 SHA, exact +8, fmt/PCM/chunks/frame count and independent wave decoding. No byte repair; 125 strict + 5 exception PASS. Generic/generated overdeclarations remain rejected. |
| M6-PANEL-001 | 6 | High | CLOSED | Integration compared a stored panel digest string without hashing ordered IDs; reversed-ID regression failed. | Recompute canonical ordered-ID SHA and require the frozen identity; test passes. |
| M6-CORE-001 | 6 | Medium | CLOSED | Integration reconstructed CORE from manifest flags. | Consume SHA-verified retained Module 2 selection; test shows flag changes cannot redefine membership. Live inventory unchanged. |
| M6-GATE-001 | 6 | Medium | CLOSED | Added integration failed Ruff/strict mypy. | Formatting, explicit compatibility exports and concrete JSON/path annotations; no scientific-policy changes. Ruff and strict mypy PASS. |

Final evaluation/resume implementation remains unexecuted and unvalidated. No automatic
recovery of interrupted final inference is claimed. A BLOCKED freeze cannot authorize
execution (regression PASS). Human correction workflow disabled and unassessed.

## Module 6 final runner and typing closure — 2026-10-08

| ID | Module | Severity | Status | Finding | Repair and evidence |
|---|---:|---|---|---|---|
| M6-FINAL-001 | 6 | High | CLOSED | Earlier integration lacked accepted once-only/interrupted-run protections. | Single immutable claim, Windows OS lock, complete run identity, immutable raw seals, failed in-flight claims retained, no score-driven retries, all-terminal scoring. Mocked interruption/concurrency/tamper/second-run tests PASS; actual 130-record run needed no resume. |
| M6-FINAL-002 | 6 | High | CLOSED | Initial state-machine prototype did not compare the supplied full ordered job list before inference. | Reproduced in job_binding_before.log; bind ordered jobs and their digest before any work. Regression PASS. |
| M6-FINAL-003 | 6 | High | CLOSED | Initial retained-seal validation did not independently check device/dtype and decoded input hash. | Reproduced in retained_binding_before.log; bind retained scientific metadata to run identity and source provenance. Regressions PASS. |
| M6-TYPE-001 | 6 | Low | CLOSED | Plain strict mypy rejected untyped Whisper/pyttsx3 imports; root 3.11 stubs omit runtime 3.12 Path.is_junction. | Narrow import-specific and version-specific annotations. Plain and explicit-3.12 strict mypy PASS; unchanged AST proof and runtime regression suite PASS. Protected root config unchanged. |

The previous final-runner unvalidated note describes the blocked recovery stage. Current
computational runner validation is PASS. Human correction remains disabled/unassessed;
human TTS listening remains pending. No claim of standards-conformant source headers.
