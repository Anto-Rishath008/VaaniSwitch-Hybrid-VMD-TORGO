# Module 6 final computational evaluation — 2026-10-08

**MODULE 6: COMPLETE / PASSED / SIGNED OFF. FINAL EVALUATION: COMPLETE / PASS.
HUMAN LISTENING: PASS — USER CONFIRMED.**

The authorized frozen 130-record panel completed once: **130 requested, 130 attempted,
130 successfully transcribed, 0 failed, 130 scored, 0 exclusions**. There were 130
record-level Whisper transcription calls, one immutable accepted raw output per ID,
no retries and no interruption. Development forwards and new TTS synthesis both **0**.

Raw corpus WER: **158/605 = 0.2611570247933884 (26.11570248%)**.
Raw corpus CER: **477/2554 = 0.18676585747846516 (18.67658575%)**.
Word S/D/I = 119/17/22; character S/D/I = 217/175/85. Values are not clamped.

These are agreement scores against **TORGO prompt-based references, not guaranteed
verbatim audio transcriptions**. Results describe this frozen panel, not every TORGO
record. No clinical accuracy claim is made. References never entered recognition,
and no ASR correction, lexical repair, model/decoding tuning or sample replacement occurred.

## Frozen source-container exception

Policy: **TORGO_RIFF_CHUNKSIZE_PLUS8_EXCEPTION**. It applies only through the explicit frozen TORGO reader
to the five approved IDs with their original Phase-0 hashes and signed manifest frame
counts. Generic overdeclared WAVs and generated TTS WAVs remain rejected.

Each accepted source satisfies: canonical raw-tree path and frozen complete-file SHA;
valid RIFF/WAVE signatures; outer overdeclaration exactly +8; complete 16-byte PCM fmt;
PCM tag 1, mono, 16 kHz, 16-bit, byte rate 32,000, block align 2; complete chunk walk;
data starting at byte 44 and ending at physical EOF; integral complete frames; exact
manifest frame count; Python wave reads the same complete payload. No source header
or sample byte is repaired, even in a temporary buffer. Original source hashes remain
over the complete unchanged files. Decoded float32 hashes are also bound to each result.

These containers are **not fully standards-conformant**. They are **accepted under a
narrow source-container exception after verifying complete PCM payload and frozen
source identity**. The PCM payload is complete while the outer ChunkSize is malformed.

| Logical ID | Physical bytes / ChunkSize | Declared total | PCM bytes | Frames | Source SHA-256 |
|---|---:|---:|---:|---:|---|
| `M01/Session1/0007` | 83884 | 83892 | 83840 | 41920 | `932952115252e32944534f68d9618febb7d260ee1e069df6d9f0999549f8b8cb` |
| `M01/Session1/0008` | 129964 | 129972 | 129920 | 64960 | `f26d9007142c0a3c2e6c1bacddf06d8e3a5f29a9705bf1b689d35ef70ee99458` |
| `M01/Session1/0014` | 88364 | 88372 | 88320 | 44160 | `7ba62962dfd075607496f93d4020e967f8f07ea3dc12b499c90e9216b829c6fc` |
| `M01/Session1/0017` | 94444 | 94452 | 94400 | 47200 | `dd48d341cfb8ee19bee0ca22ea08c59cc0f57531c2f9f37e5875da8961651d33` |
| `M01/Session1/0020` | 112364 | 112372 | 112320 | 56160 | `820b899f0e8e811bc38c3b114cb14e5ddbc331c7ca5f0af6d8547ad142333450` |


All five have overdeclaration 8, data offset 44, decoded frames equal to manifest frames,
and source_bytes_modified=false. Final preflight and post-run source audit independently
confirmed **125 strict sources + exactly these 5 exceptions; zero other exceptions**.
Detailed proof: [source exceptions](../../../artifacts/module6/exception_evaluation_20261008/preflight_20261008T173011117192Z/source_exceptions.json)
and [post-run verification](../../../artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f/POST_RUN_VERIFICATION.json).

## Quality and runtime safety

- Full M6 pytest: **93 passed, 1 skipped**, 7.03 s. Retained Phase-1 evidence separately
  **1 passed**, 0.35 s. The exception/runner/gate subset: **24 passed**, 1.41 s.
- Ruff PASS. Plain `mypy --strict` PASS (37 files); explicit Python-3.12 strict mypy
  also PASS. Compileall, whitespace and isolated dependency compatibility PASS.
- The user-supplied typing patch had stopped after artifacts.py. Import-specific
  `import-untyped` annotations and a version-specific Path typing annotation fixed
  the remaining errors. AST hashes for all five touched files are unchanged, so
  the typing repair changed no runtime/scientific behavior. Root pyproject/uv.lock unchanged.
- Source tests reject wrong SHA/frame count, +6/+10 declarations, incomplete data,
  wrong PCM, unapproved IDs and generic/generated misuse; metadata persistence and
  byte preservation pass. All five real sources were tested read-only.
- Windows OS lock and one immutable panel claim prevent concurrent/second panels.
  Run identity binds the PASS freeze SHA, exact ordered panel, code, config, environment
  lock/packages, model, normalizer, metric policy, CUDA device/dtype and frozen decoding.
- Each record gets an immutable attempt before inference and a success seal only after
  raw text, backend result and source-provenance hashes are stored. All retained seals
  and scientific bindings are checked before further work.
- Synthetic interruption/resume proves successful records are never silently rerun.
  A claimed unfinished attempt becomes ASR_FAILURE and is not retried; only never-claimed
  records execute on resume. Runtime failures stay failures, never empty hypotheses.
  Poor text/reference scores cannot cause retries. Any failures keep the run PARTIAL.
- Evaluation begins only after every requested record reaches an explicit terminal state.
  No real final-panel forward was used to test resume. The real run needed no resume.

## Freeze history and panel

Previous BLOCKED freeze remains byte-identical:
`artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/PRE_EVALUATION_FREEZE.json`
SHA `385c3be9ec4cdee0b5397d9cac9dc4d713952f945125592a7145940f5ee5d169`.

New **PASS** freeze:
`artifacts/module6/exception_evaluation_20261008/preflight_20261008T173011117192Z/PRE_EVALUATION_FREEZE.json`
SHA `79738cd85a8b251ddab654a217eba9e192ea054030fe362db8b352f4701d8fb8`.

Preflight PASS: 130 exact ordered IDs, 13 speakers, 70 dysarthric / 60 control,
65 word / 65 sentence, 545.6203125 seconds, M03/MC02 excluded, all source hashes match,
all references eligible, audio-only jobs, required model/config/environment/metric identities,
RTX 4070 CUDA float16 allocation and upstream preservation PASS. Zero existing final
completion and zero evaluation forwards at freeze creation. Both freeze histories are retained.

## Grouped raw results

### Per condition

| Group | Records | Raw WER (%) | Raw CER (%) |
|---|---:|---:|---:|
| control | 60 | 6.711409 | 6.174820 |
| dysarthric | 70 | 44.951140 | 30.604438 |

### Word / sentence

| Group | Records | Raw WER (%) | Raw CER (%) |
|---|---:|---:|---:|
| sentence | 65 | 22.222222 | 16.732543 |
| word | 65 | 58.461538 | 34.657040 |

### Inherited speaker groups

| Group | Records | Raw WER (%) | Raw CER (%) |
|---|---:|---:|---:|
| 1 | 50 | 22.388060 | 16.129032 |
| 2 | 40 | 27.810651 | 19.189189 |
| 3 | 40 | 30.357143 | 22.206304 |

Equal-weight three-group WER: **26.851951% +/- 4.070123
percentage points sample SD**. CER: **19.174842% +/-
3.038661 percentage points sample SD** (ddof=1).
These group means differ from the corpus micro scores because weighting differs.

### Per speaker

| Group | Records | Raw WER (%) | Raw CER (%) |
|---|---:|---:|---:|
| F01 | 10 | 63.829787 | 44.387755 |
| F03 | 10 | 14.754098 | 12.790698 |
| F04 | 10 | 9.523810 | 3.125000 |
| FC01 | 10 | 6.896552 | 6.437768 |
| FC02 | 10 | 3.921569 | 5.022831 |
| FC03 | 10 | 5.882353 | 5.633803 |
| M01 | 10 | 57.894737 | 36.363636 |
| M02 | 10 | 50.000000 | 39.634146 |
| M04 | 10 | 89.361702 | 55.882353 |
| M05 | 10 | 35.294118 | 27.343750 |
| MC01 | 10 | 16.666667 | 13.043478 |
| MC03 | 10 | 6.000000 | 6.190476 |
| MC04 | 10 | 3.846154 | 2.369668 |

## Runtime and frozen identities

ASR total excluding model loading: **40.091342700 s**.
Record min/median/mean/max: **0.096983400 / 0.261431550 /
0.308394944 / 0.856887900 s**.
Model loading: **2.748005900 s**, one load.
CUDA float16 on NVIDIA GeForce RTX 4070 Laptop GPU; torch 2.13.0+cu126/CUDA 12.6;
Whisper small.en/openai-whisper 20250625 with unchanged Phase-1 decoding.

| Binding | SHA-256 |
|---|---|
| Ordered panel | `15703d20490c83ac814664c48c3d8c8bf1197d2054d7d9d5ce495c68804a536b` |
| Model checkpoint | `f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872` |
| Config identity | `60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209` |
| Environment lock | `49cc7da351e802a277bd5435dc520c35e1ad15b36fc8d52b4c6f9cc0bc5d6af1` |
| m6_text_v1 normalizer | `764d2f745b1fefed7775f41674da19d865c9c882109cff693a6e2cdee883165d` |
| m6_metrics_v1 / JiWER 4.0.0 | `efa2bc4d3865bcc3721226c369b5875a0aca9680e973e938d965e994a38f381c` |
| M6 code/tests/launchers | `df56cd2d115a7d2d63961de0aa2a575cc2792847881a3e227ddac80aa807812a` |
| Final run identity | `165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f` |

## Preservation and artifacts

Preflight and post-run preservation **PASS**: 124 protected files, 20,366 original
environment files, 39 ledger files and 134 selected primary WAVs unchanged. Three
handoff checkpoint hashes match. Frozen M0-M5 scientific code remains
`7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8`.
No Module 5 execution or Module 7 work occurred. Fold1/A1 remains
RECOVERED_PROTOCOL_EXCEPTION; original native-forward count UNKNOWN.
Post-run read-only verification checked all 130 raw seals, decoded source hashes,
references and metrics and reproduced every aggregate/group result without inference.

Final artifact root: `artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f`.

- [Completion evidence](../../../artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f/FINAL_EVALUATION_COMPLETE.json)
- [Full raw metrics and runtime](../../../artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f/metrics_raw.json)
- [Per-record CSV](../../../artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f/metrics_raw.csv)
- [Post-run verification](../../../artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f/POST_RUN_VERIFICATION.json)
- [Quality gates](../../../artifacts/module6/exception_evaluation_20261008/QUALITY_GATES.json)

Original blocked report is preserved under `artifacts/module6/exception_evaluation_20261008/prior_reports/FINAL_EVALUATION_REPORT.md`.
The earlier recovery report is historical; this report supersedes its computational status.

## Human listening and final sign-off

The user personally listened to both original development demos and explicitly
confirmed: "Both produce clear, intelligible English speech." No new synthesis occurred:

- [M03/Session2/0013 clear.wav](../../../artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/M03/Session2/0013/tts_raw/clear.wav): 70428 frames, 3.194013605 seconds.
- [MC02/Session1/0006 clear.wav](../../../artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/MC02/Session1/0006/tts_raw/clear.wav): 68442 frames, 3.103945578 seconds.

Both mechanically pass PCM16 mono 22,050 Hz, nonzero finite samples, pyttsx3 2.99,
SAPI5/Zira, rate 175, volume 1.0. They use raw ASR only. Human listening is
**PASS — USER CONFIRMED** for both IDs. The agent did not personally listen.
[Listening confirmation and exact WAV hashes](HUMAN_LISTENING_REPORT.md).

Read-only completion/hash, retained-record consistency and upstream preservation
checks pass. No ASR, TTS or metric recalculation occurred during final sign-off.
Original computational completion and synthesis records retain their historical
pending-listening fields; the [new immutable sign-off](FINAL_SIGNOFF_REPORT.md)
records the subsequent user confirmation. Module 6 is **COMPLETE / PASSED / SIGNED OFF**;
VaaniSwitch implementation and Modules 0–6 are **COMPLETE**. All scientific limitations
and source exceptions remain unchanged. Stop after Module 6; Module 7 not started.
