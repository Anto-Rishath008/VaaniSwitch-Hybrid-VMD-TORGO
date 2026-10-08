# Module 6 Phase 2 contract validation

**2026-10-08: PASS.** [PHASE2_EVIDENCE.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/PHASE2_EVIDENCE.json)

Live validation covers 9,494 manifest records and 15 speakers. CORE membership is read
from the SHA-verified retained Module 2 selection (8,673 IDs), without recomputing
upstream eligibility. The recovered inventory is exactly equal to the partial-run
inventory: 111 CORE image references excluded; 8,562 provisional scoring references,
comprising 6,458 words and 2,104 sentences.

References join by speaker/session/prompt and the corresponding TORGO prompts path.
Original prompt bytes, hashes, decoded text and derived targets remain separate.
Only the supported `word [as in ...]` pattern reduces to the word. Other brackets
require review and remain unscored. Missing/blank/xxx/image references are unscored
with explicit reasons. Targets are never repaired from ASR.

**TORGO prompt-based reference; not guaranteed verbatim audio transcription.** Scores express agreement with these supplied prompts.

`m6_text_v1` applies NFKC, lowercase, curly-apostrophe conversion and apostrophe removal,
other punctuation to spaces, whitespace collapse and trim. Digits remain unchanged;
no lexical, spelling, number or grammar repair. CER removes normalized spaces.
JiWER 4.0.0 counts word/character substitutions, deletions and insertions. Empty valid
hypotheses score as deletions; runtime failures remain ASR_FAILURE with null metrics.
Corpus scores sum edits and reference lengths, and scores above 100% are retained.
Per-speaker, condition, prompt-type and inherited-group summaries are available;
three-group summaries use equal weighting and sample SD (ddof=1).

- Normalizer SHA: `764d2f745b1fefed7775f41674da19d865c9c882109cff693a6e2cdee883165d`
- Metric configuration SHA: `efa2bc4d3865bcc3721226c369b5875a0aca9680e973e938d965e994a38f381c`
- Retained inventory SHA: `54b9d49a7f7b2c5663da357a455c0deb6d89e4e043a17e046cea31779ad91db6`

Regression/contract coverage includes exact composite joins, reference contamination
rejection in audio jobs, signed CORE reuse, supported/unsupported references, missing
targets, normalization, insertion scores above 100%, failure versus empty hypothesis,
count-weighted aggregation, duplicate-record rejection and sample SD.

Correction is DISABLED / NOT EVALUATED. The optional correction writer has not been
accepted as a human-review workflow; no correction artifact was created.
