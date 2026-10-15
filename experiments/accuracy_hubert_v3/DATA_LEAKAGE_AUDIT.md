# V3 data leakage audit

PASS for implemented contracts and bounded validation; not a claim that the full
unexecuted experiment has run successfully. Dataset and CV audit JSON files are
under validation/audit-*.json. Tests include live manifest/split checks and synthetic
adversarial role/hash/variant tests. Bounded real access is in validation/latest_cuda.json.

The original manifest byte hash and fold byte hash are unchanged. CORE remains
8,673 logical speaker/session/prompt records, 15 speakers, 8 dysarthric and 7 control;
outer counts remain 2,819 / 2,873 / 2,981. Validity-selected head/array primaries are
inherited; alternate microphones never add logical records. The distinct canonical
fold fingerprint is retained rather than confused with the file-byte SHA-256.

The inherited split audit and new grouped CV check disjoint speaker IDs, logical
IDs, primary paths and waveform hashes. Three deterministic development folds
cover every development speaker exactly once for validation and contain both
classes. No recording-random split is available. No historical outer predictions
or speaker error rankings are accepted by the candidate-selection functions.

Weights, class totals, medians, standard/robust scales and all fitted classifier
parameters use only current training originals. Optional calibration uses separate
base-training, calibration and assessment speakers with a fixed recipe; it cannot
consume selected-model results or any reserved outer predictions. Threshold is 0.5.

Augmentation requires a fitting role and checks parent membership and unchanged
speaker/label/record identity. Original and variant remain in the same active
partition. Reuse across different outer-fold training contexts is allowed only
when that parent is training in that context. Available augmented caches never
authorize validation/outer consumption. Scalers fit originals only and variants
share the parent's objective mass. Access receipts list exact role and record IDs.

Bounded real panel: 16 records from the eight Fold 1 inner-training speakers,
four unaugmented records from its two inner-validation speakers, eight augmented
training waveforms. Excluded outer speakers: F03, FC01, M04, MC03, MC04. No outer
speaker entered bounded model fitting, scaling, weighting, augmentation or selection.
All 8,673 original embeddings were checksum-read as immutable integrity inputs;
this is not corpus-statistic fitting or outer model inference.

All production choices must freeze across all three folds before outer inference.
Authorization is checked before any outer inference operation; a durable claim
precedes the forward call. Synthetic tests verify claim reuse, ambiguous interruption
rejection, decision/checkpoint tampering rejection and complete prediction coverage.
No real V3 outer claim or result exists in this implementation handoff.

Post-hoc research-direction exposure remains: original/V1/V2 results motivated V3.
The controls prevent within-run leakage; they do not restore an independent test set.
