# Module 6 Phase-0 audit and implementation plan

Date: 2026-10-07, Asia/Calcutta. Submission deadline: 2026-10-09.

**Phase-0 audit: PASS. Module 6 implementation: NOT STARTED. Runtime acceptance: NOT TESTED. Module 6: NOT SIGNED OFF.**

PASS means the existing design, upstream identities, local reference availability, implementation gaps and acceptance plan have been audited. It does not mean ASR/TTS works locally or that WER/CER has been measured. This session authorizes documentation and read-only inspection only. Stop here; no model download, environment installation, inference, training, full-corpus benchmark or Module 7 work was performed.

## 1. Evidence and frozen upstream boundary

Read the root scope lock and AGENTS, README, status, tracker, architecture rationale, final workflow, accuracy strategy/matrix, TORGO audit, Module 5 final closure and JSON handoff, Module 6 AGENTS/MODULE/TEST_PLAN/TEST_REPORT/CODEX_PROMPT, relevant configs, and current source/test/script/artifact structure. The live final closure supersedes older historical notes that described Module 5 as unfinished.

The following current identities match the user's supplied values. Protocol identity is the canonical scientific config digest, not a raw YAML-file checksum. The code digest uses the existing sorted `src/vaaniswitch/m[012345]*/*.py` mapping and covers 92 files; no experiment harness was executed.

| Identity | Verified SHA-256 |
|---|---|
| Protocol/config | `677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9` |
| Modules 0–5 scientific code | `7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8` |
| `final_closure/module5_final_closure.json` | `449b1a8237efbb21e7912868d4a0586bdca30e85806248ca646b6d647f5683e2` |
| `final_closure/cross_fold_metrics.json` | `41d1d316841c7779a8d11e5ab70ed58b1fb7a44a473585cbc271e9b981369d9f` |
| `final_closure/module6_handoff.json` | `36f5a276137922fd585dfe052e2402218fe36a4b0569384e0c4bd4e241b2235f` |
| `final_closure/evidence_manifest.json` | `f9d4e56917b2bfc63abbf312e2b389457aa2ad29734424f01da5ee03109a8a6d` |
| `final_closure_validation/delivery_checks.json` | `7dbdb25053426621cb42f1fa61f501c1e078c9234c16170a772e87a601a8fd20` |

These paths are relative to `artifacts/module5/<protocol-hash>/`. All eight closure-package member hashes also match its evidence manifest. The closure and delivery status are retained PASS evidence; their scientific audits/tests were not rerun.

All three C2 checkpoint files were hashed without loading models:

| Fold | Preserved candidate | Epoch | Checkpoint SHA-256 |
|---|---|---:|---|
| 1 | stronger_dropout | 1 | `d32bc51dce71ab37afd02471c5cfe874ce03f737e8191acad7f31663c6fe4e24` |
| 2 | stronger_decay | 2 | `cf54f92b2e06c6c544e4b9efb91d3e248f9207f9ce773fc510fd4f23b3649d8d` |
| 3 | baseline | 2 | `a0c0b3520ef6f04a11e29c32dd11a65fda363b3bbd8793c82f75f1e53ac37508` |

Fold1/A1 remains **RECOVERED_PROTOCOL_EXCEPTION**, original prior native-forward count **UNKNOWN**. All Module 5 limitations travel with its evidence. No deployment checkpoint or ensemble is selected. Module 6 requires no classifier invocation or refit; Module 7 integration remains outside this plan.

`PHASE0_EVIDENCE.json` records the inspection, proposed sample IDs, primary-audio hashes and 124 protected file hashes. This is a targeted identity/reference audit, not a fresh validation of all TORGO audio, caches, ledger history or every upstream result. Root status/changelog documents are live navigation records; their later Phase-0 annotations do not rewrite the immutable Module 5 delivery snapshot or its historical document hashes.

## 2. Existing intent, files and missing features

| Existing item | What is present | What is absent |
|---|---|---|
| `docs/modules/6_english_asr_tts/` | Objective, output list, agent rules, basic test plan; initial test report NOT RUN | Implemented/validated workflow |
| `src/vaaniswitch/m6_reconstruction/` | Empty `__init__.py`, README, AGENTS, placeholder | ASR, references, metrics, review, TTS, persistence, CLI |
| `tests/m6_reconstruction/` | README and placeholder | Executable tests |
| `configs/asr/default.yaml` | `frozen_whisper_family`, **`small.en`**, `fine_tune: false`, `language: en` | Package/weights hash, decoding/device settings |
| `configs/tts/default.yaml` | **`pyttsx3`**, offline preferred, no cloning | Voice ID, rate, volume, WAV validation |
| Module 0 manifest and folds | Logical IDs, prompt paths/text, audited primary WAV paths/hashes, speaker groups | Module 6 reference eligibility and immutable evaluation panel |
| `scripts/`, root package metadata | M0–M5 commands and audit tools | M6 command/isolated dependency lock |
| `apps/api`, `apps/demo`, `artifacts/demo_outputs` | Placeholders only | End-to-end utility demo; application belongs to M7 |
| Local packages | Existing scientific environment | Whisper, pyttsx3, JiWER and M6 runtime |

The model **family and size were specified already**; they were not previously downloaded, hash-pinned or runtime-frozen for M6. Use those choices. The decoding policy, voice choice, dependency pins, reference rules and small evaluation panel below are **new proposals**, not historical frozen decisions. No ASR or TTS training is required by any inspected architecture document.

## 3. Proposed architecture

```text
original complete English primary WAV (read-only)
  -> validate/decode PCM -> float32 waveform, 16 kHz mono
  -> frozen Whisper small.en, English transcription, no reference context
  -> immutable RAW ASR TRANSCRIPT + exact backend JSON
       |-> evaluation-only join to REFERENCE TRANSCRIPT -> raw WER/CER
       |-> optional explicit human review -> OPTIONAL CORRECTED TRANSCRIPT
       |                                  |-> separate corrected WER/CER
       -> selected synthesis text (raw by default; explicit correction otherwise)
          -> pyttsx3 / Windows SAPI5 / fixed English voice
          -> validated clear synthesized WAV + manifest + CLI demo report
```

Use the original primary WAV, as specified by “same user audio” in FINAL_WORKFLOW, not the VMD reconstruction or eight-second Mel crop. The completed shared-VMD classifier remains a separate frozen upstream result. Do not infer that TTS restores a speaker's original voice or acoustic characteristics. The output is standard clear English synthesized speech; no clinical claim.

The minimum interface accepts PCM16 mono 16 kHz WAV, matching every checked TORGO input. Read the complete file with Python `wave`, scale int16 by 32768 into float32, and pass the array to Whisper. Reject unsupported input formats explicitly in this first version. This avoids adding an FFmpeg dependency for the bounded TORGO path; FFmpeg is absent from PATH. Broader uploads/resampling belong to a later explicit extension. Whisper supports waveform input and handles successive windows; do not call a whole-file 30-second truncation helper. [Whisper audio implementation](https://github.com/openai/whisper/blob/main/whisper/audio.py), [transcription implementation](https://github.com/openai/whisper/blob/main/whisper/transcribe.py).

## 4. Exact pretrained approach

**ASR:** proposed `openai-whisper==20250625`, official `small.en` checkpoint. Expected official checkpoint SHA-256: `f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872`. Verify the downloaded bytes before use; no weights are currently present in the default Whisper cache. Cache only under `artifacts/module6/models/whisper/`. Record source URL, package version and checkpoint digest. The official model is English-only, 244M parameters, with an approximate 2 GB VRAM requirement, making it a plausible fit for this GPU; actual local compatibility/latency is unmeasured. Code/weights are MIT-licensed. [Official Whisper repository](https://github.com/openai/whisper), [checkpoint mapping](https://github.com/openai/whisper/blob/main/whisper/__init__.py), [release metadata](https://pypi.org/pypi/openai-whisper/json).

Proposed deterministic decoding: `task=transcribe`, `language=en`, scalar `temperature=0.0` (no temperature fallback), greedy decoding (`beam_size=None`, `best_of=None`), `initial_prompt=None`, `prefix=None`, `condition_on_previous_text=False`, `carry_initial_prompt=False`, `word_timestamps=False`, complete audio (`clip_timestamps="0"`). Pin the remaining backend defaults in the resolved manifest, including suppression tokens and silence/log-probability thresholds. Use `eval()`, disabled parameter gradients and inference mode; no optimizer or classifier import. No dictionary built from TORGO targets and no per-speaker prompts. Seed 42 is recorded for reproducibility, without promising identical CPU/GPU numerics. [Whisper transcription options](https://github.com/openai/whisper/blob/main/whisper/transcribe.py).

**TTS:** proposed `pyttsx3==2.99`, `driverName="sapi5"`, voice token `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0`, rate 175, volume 1.0. Microsoft Zira Desktop is registered locally (English US, LCID 409); David (English US) and Hazel (English UK) are also registered. Zira/rate are new proposed settings. Validate enumeration in the new environment and fail visibly if that exact voice is unavailable; do not silently switch voices. This follows the existing offline backend and needs no neural TTS model download. [pyttsx3 project](https://github.com/nateshmbhat/pyttsx3), [Windows driver](https://github.com/nateshmbhat/pyttsx3/blob/master/pyttsx3/drivers/sapi5.py), [release metadata](https://pypi.org/pypi/pyttsx3/json).

Run TTS in a bounded worker process with a 60-second timeout, using `save_to_file` followed by `runAndWait`; write a temporary WAV in the permitted run directory, validate it, then publish atomically. Require RIFF/WAVE, PCM, positive frame count/duration and nonzero finite samples; record actual sample rate/channels/bit depth. Verify the exact text sent to TTS. Windows voice engine/version is part of provenance; portable byte-identical audio across Windows installations is not promised. A listening check remains necessary. Do not substitute a cloud service or another model silently on failure.

## 5. TORGO reference source and eligibility

The exact local source is `data/raw/torgo/extracted/<group>/<speaker>/<session>/prompts/<four-digit-id>.txt`, linked through `data/manifests/torgo_manifest.jsonl` by **speaker/session/prompt**, never by prompt number alone. Preserve the existing validity-aware primary microphone choice; the alternate microphone is not another sample.

All **9,494** manifest-linked prompt files were read successfully as UTF-8-SIG; their stripped text matches the existing manifest exactly. M0 also permits CP1252 fallback; retain the same decoding order and record the encoding. Store original bytes/hash, exact decoded source text and a separate derived scoring reference. Do not edit the raw files or rewrite the signed manifest.

| Current local observation | Count / implication |
|---|---|
| Manifest / speakers | 9,494 / 15 |
| Prompt types | 6,938 word; 2,287 sentence; 118 articulation; 92 instruction/nonstandard; 59 unknown |
| Existing CORE | 8,673: 6,569 word + 2,104 sentence |
| Image filenames | 118 overall; **111 inside CORE**; not spoken references |
| Word-sense annotations | 85 overall; **81 inside CORE** |
| Placeholder `xxx` | 58 overall; zero CORE |
| Provisional M6 text-reference pool | **8,562** after excluding 111 CORE image references; 6,458 word + 2,104 sentence |

The official TORGO documentation identifies `prompts/` as orthographic text, `xxx` as unusable content and `.jpg` entries as picture stimuli. Its terms permit academic nonprofit use and require citation of at least one listed TORGO paper. Use these already acquired prompts for this academic evaluation, cite Rudzicz, Namasivayam and Wolff (2012), and retain source attribution. This does not establish permission for unrelated commercial use or public redistribution of the corpus. No extra dataset or image-card transcription is needed. [TORGO documentation and terms](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html).

M6-only reference rules, applied before seeing hypotheses:

1. Start with signed CORE membership; never recompute M0–M5 eligibility.
2. Exclude image paths/stimulus filenames, missing/blank/placeholder text, task instructions, malformed controls and unresolved annotations from WER/CER. Emit an explicit reason and denominator; inference-only demonstrations may retain a `reference_unavailable` status.
3. For the exact supported pattern `single_word [as in ...]`, score the word only. For example, `tear [as in tear up that paper]` becomes scoring reference `tear`; retain the complete source text and rule ID. Do not generically delete arbitrary bracketed material. All 81 CORE bracket cases match the documented sense pattern; unknown cases must stop for review.
4. Retain spoken repetitions, sentence fragments and lexical errors in the supplied text. Do not repair targets using ASR. Numeric spellings are not silently expanded.
5. Treat these as **prompt-based references**, not newly audio-verified verbatim transcripts. No listening/transcription audit occurred. Known long outliers include `F03/Session1/0118` (193.828625 s, prompt `no`) and `M01/Session2_3/0180` (179.943125 s, prompt `hill`); they need separate reference review before broader ASR evaluation. They remain unchanged upstream.

The 8,562 count establishes provisional availability only. It is not a validated full benchmark and must not be reported as an ASR result. The proposed short panel bounds duration without truncating any selected recording.

## 6. Leakage barriers and three distinct transcript states

**RAW ASR TRANSCRIPT:** exact backend-returned string, including original whitespace; persist before any text normalization or review. Hash the backend JSON and UTF-8 text. Never replace this file with a correction or target text.

**OPTIONAL CORRECTED TRANSCRIPT:** separate versioned artifact created only after an explicit edit is submitted. Record raw-parent hash, correction text/hash, timestamp, actor/source, reason and `reference_visible=false` declaration. Review can expose input audio and raw text; it must not expose references, WER/CER, target word lists or other scored outputs until the edit is sealed. Evaluation is a separate command/process. An operator already familiar with targets cannot support a blind corrected-text benchmark; label that case reference-assisted and exclude it from independent corrected evaluation.

**REFERENCE TRANSCRIPT:** evaluator-only input, with prompt-path/hash, logical ID, raw source text, derived spoken target, eligibility reason and normalizer version. It never enters the ASR or correction API. The recognition job manifest contains opaque record IDs and audio bindings only, no prompt text or prompt paths. Recognition does not import the reference reader. TTS receives raw or explicitly approved corrected text only, never the reference.

References and scores may appear together in the final sealed evaluation report. Technical isolation prevents accidental software leakage, but human blindness also requires the stated workflow. No LLM, automatic grammar correction or reference-driven substitution is proposed for the deadline path.

Raw WER/CER is mandatory and independently reportable. Corrected WER/CER is optional, named **human-assisted corrected-text agreement**. Compare raw and corrected outputs on the exact same corrected subset with the same references/normalization; also retain the complete raw-panel report. Never interpret human edits as improvement of the frozen ASR model. With no explicit corrections, report `not_evaluated / correction_disabled`; do not invent corrected transcripts or zero scores.

## 7. WER/CER computation contract

Use proposed `jiwer==4.0.0` for Levenshtein alignment/counts with explicit project normalization, not opaque default transformations. The metric implementation supports WER/CER and minimum edit distance. [JiWER documentation](https://jitsi.github.io/jiwer/), [release metadata](https://pypi.org/pypi/jiwer/json).

Proposed `m6_text_v1`: Unicode NFKC; lowercase; map curly apostrophes to ASCII apostrophes; remove apostrophes; replace other punctuation with spaces; collapse whitespace and trim. No spelling repair, word substitution, number-to-word expansion or target-specific substitutions. Derive the spoken reference first; apply this same normalizer to all three transcript states. WER splits on whitespace. CER removes normalized spaces before counting characters. Preserve original and normalized values plus a normalization config hash. These conventions are explicit project choices, so comparisons with other papers require matching their rules.

`WER = (word substitutions + deletions + insertions) / reference words`.
`CER = (character substitutions + deletions + insertions) / reference characters`.

Store H/S/D/I, reference lengths, alignments, ratios and percentages per record. Do not clamp scores at 100%. Empty ASR with a valid reference is all deletions and is scored. Empty/unavailable normalized references yield null metrics and an exclusion reason, not zero or success, regardless of JiWER's special empty-reference convention. Runtime inference failures remain failures and count against completion; do not silently drop them or manufacture empty hypotheses.

Corpus rates are sums of error counts divided by sums of reference lengths, computed per utterance before aggregation to prevent cross-record alignment. Report micro rates, per-speaker rates, condition/prompt-type counts and rates, and the three inherited speaker-group rates with unweighted mean ± sample SD (`ddof=1`). Group results describe one frozen ASR system, not three ASR models trained by cross-validation. No accuracy target or improvement guarantee is imposed.

## 8. Small reproducible evaluation and demo

Proposed operational development speakers: **M03 and MC02**, matching the existing development context but without using any M5 predictions. Use the first eligible word and sentence from each, ordered by logical ID, for four development smoke records. Their total audio duration is 16.075125 seconds. Any implementation troubleshooting or settings adjustment stays with these speakers or synthetic fixtures. Freeze the final config/environment before evaluating other speakers.

Proposed evaluation panel: remaining **13 speakers**, first five eligible words and first five eligible sentences per speaker, lexicographically by logical ID, full duration **0.5–15 seconds inclusive**. This yields **130 records**, 70 dysarthric and 60 control, 65 words and 65 sentences, totaling **545.6203125 seconds** of audio. Original M0 group membership gives group 1 = 50, group 2 = 40, group 3 = 40 records after withholding the two development speakers. No speaker overlaps the development and evaluation panels. Ordered IDs and their digest are in `PHASE0_EVIDENCE.json`: `15703d20490c83ac814664c48c3d8c8bf1197d2054d7d9d5ce495c68804a536b`.

This is a proposed deterministic convenience panel, selected using source metadata before ASR exists. It is not full-corpus performance, a random population sample, or evidence of clinical/general speaker performance. Shared prompts across speakers are disclosed; neither prompt lists nor references are provided to ASR. Membership/duration limits cannot be changed in response to WER. If an eligibility discrepancy emerges, log it and version the plan before scoring, never silently substitute a favorable record.

Read-only verification covered all **134 proposed primary WAVs** (130 evaluation + 4 development): hashes and frame counts match the signed manifest; all are PCM16 mono 16 kHz, with 104 head and 30 array primaries. This is not ASR execution. Original raw/cache full-tree audits were not repeated.

Demonstrate at least one dysarthric development sentence and one control development sentence through raw ASR -> report -> raw-text TTS. Use `M03/Session2/0013` and `MC02/Session1/0006` as predetermined examples. An explicit correction may be demonstrated separately if an operator supplies one; label any known-reference demonstration accordingly. Empty raw text must produce a truthful `NO_TEXT_FOR_TTS` outcome rather than target-based speech. Preserve unsuccessful examples; a supplemental self-recorded English demo may establish the mechanics with source permission, clearly outside TORGO metrics. No favorable-example replacement of the measured panel.

The CLI demonstration ends at M6 audio/report outputs. Building the web application, exposing classification probability or invoking a retained classifier is Module 7 scope.

## 9. Hardware and isolated environment

| Observed on this workstation | Result |
|---|---|
| OS / CPU | Windows 11 Pro 10.0.26300 / i9-13980HX, 24 cores, 32 logical processors |
| RAM / C: free space | 31.63 GiB / 32.29 GiB at inspection |
| GPU / driver | RTX 4070 Laptop GPU / 617.14 |
| GPU memory | 8,188 MiB total; 6,759 MiB free at inspection |
| Existing `.venv` | Python 3.12.2; torch 2.14.0; torchvision 0.29.0; NumPy 1.26.4 |
| Existing CUDA status | `torch.version.cuda=None`; `torch.cuda.is_available()=False` |
| Missing in existing environment | openai-whisper, pyttsx3, JiWER, soundfile, SciPy; FFmpeg absent from PATH |

Prefer **GPU ASR and CPU SAPI5 TTS in a new `.venv-m6`**. Leave `.venv`, root `pyproject.toml`, root `uv.lock`, all M0–M5 code/configs and their runtime untouched. Use standalone `configs/module6/environment/requirements.in` plus a Windows/Python-specific hashed lock and direct `.venv-m6/Scripts/python.exe` calls; do not run project-wide `uv sync` or install the root package's dependencies. Execute M6 using `PYTHONPATH=src` or a dedicated launcher with only M6 imports.

Proposed initial resolution: Python 3.12.2, `torch==2.13.0` from the official CUDA 12.6 wheel index, NumPy 1.26.4, openai-whisper 20250625, pyttsx3 2.99 and JiWER 4.0.0, plus separately pinned test/type tools and all transitive dependencies. The official PyTorch matrix lists Windows CUDA 12.6 wheels for 2.13.0. This is a candidate environment, not a compatibility-tested lock; resolve wheel hashes and smoke-test imports/CUDA/voice before freezing it. No torchvision/torchaudio is needed for this PCM/Whisper/SAPI path. [Official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/).

ASR uses batch size 1 and FP16 on CUDA; CPU fallback uses the same model, FP32 and a recorded thread count (proposed 8). Spend at most about one hour diagnosing CUDA installation during the deadline path, then use an explicitly recorded CPU profile if necessary. Decide the device on development smoke before evaluation; do not silently mix devices or retry scored records with another model. Require free-space and import checks, CUDA tensor allocation, complete model checksum validation and a real SAPI5 WAV smoke. None of these new-environment gates ran in Phase 0. Keep model/dependency downloads outside any raw-data path. GPU speed estimates from other hardware are not local timings.

## 10. Implementation order and exact planned files

Every path below is **future work**, unless explicitly identified in section 13 as Phase-0 documentation.

| Order | Modules/files to create or modify | Purpose |
|---:|---|---|
| 1 | Create `configs/module6/environment/requirements.in`, `requirements-win-py312.lock`; create `configs/module6/pipeline.yaml`; extend existing `configs/asr/default.yaml`, `configs/tts/default.yaml`; modify `.gitignore` for `.venv-m6/` | Resolve isolated runtime, pin weight/voice/device, record new M6 settings without touching M5 |
| 2 | Create `src/vaaniswitch/m6_reconstruction/config.py`, `schemas.py`, `artifacts.py` | Typed states, canonical output protections, hashes, exclusive creation and atomic completion |
| 3 | Create `references.py`, `normalization.py`, `metrics.py`, `evaluation.py` | Read-only source matching, eligibility log, isolated reference join, explicit metric contract, frozen panel selection |
| 4 | Create `audio.py`, `asr.py` | Full PCM validation, fixed model inference, exact raw output persistence |
| 5 | Create `review.py`, `tts.py` | Explicit correction state, exact synthesis text selection, bounded SAPI worker, WAV checks |
| 6 | Create `pipeline.py`, `cli.py`, `__main__.py`; keep or minimally document existing `__init__.py`; create `scripts/module6_demo.ps1` | Separate prepare/transcribe/review/evaluate/synthesize/demo commands; ordered leakage-safe workflow |
| 7 | Create tests listed below; update `src/vaaniswitch/m6_reconstruction/README.md`, `tests/m6_reconstruction/README.md`, M6 `MODULE.md`, `TEST_PLAN.md`, `TEST_REPORT.md`; update live tracker/status/README/changelog/defect log | Development smoke, freeze manifest, bounded evaluation, documentation and acceptance evidence |

Avoid importing the M5 harness even for metadata. Standard-library JSON reading suffices for retained manifest/handoff data. No ledger APIs, checkpoint deserialization or aggregation/retraining helpers are needed. All future writes resolve under `artifacts/module6/` (plus the listed new environment/config/code/test/docs); reject symlink/junction/path traversal into raw data, caches, predecessor artifacts or `.venv`. Existing completion directories are immutable; invalid resume identity is an error, not a reason to overwrite a transcript.

## 11. Exact output contract

Use `artifacts/module6/<config_sha256>/<run_id>/`, exclusively created. Run identity binds the code/config, environment lock, weight hash, voice/settings, panel/source hashes, device/dtype, normalizer and correction policy. Proposed files:

| Path within run | Content |
|---|---|
| `config.resolved.json`, `environment.json`, `reproducibility.json` | All identities, Python/packages/driver, command, platform/voice, timestamps, seeds and limitations |
| `panel.json`, `recognition_jobs.jsonl`, `reference_eligibility.jsonl` | Frozen membership, reference-free jobs, evaluator-only eligibility/reasons |
| `records/<speaker>/<session>/<prompt>/raw_asr.txt`, `raw_asr.json` | Exact raw string and backend segments/options/timing/audio binding |
| same record directory: `reference_source.txt`, `reference.txt`, `reference.json` | Original decoded prompt, derived spoken reference, source hash/rules; evaluator-only |
| same record directory: `corrections/<revision>/corrected.txt`, `correction.json` | Only explicit submitted edits; raw-parent identity and assistance disclosure |
| same record directory: `metrics_raw.json`, optional `metrics_corrected.json` | Counts, ratios, normalized strings, alignments, exclusions and provenance |
| same record directory: `tts/<raw-or-correction-hash>/synthesis_text.txt`, `synthesis.json`, `clear.wav` | Exact input text, voice/runtime/audio validation and generated WAV |
| `metrics_raw.csv`, `metrics_raw.json`, optional `metrics_corrected.csv/json` | Per-record, paired-subset, speaker/group and corpus summaries; completion denominators |
| `demo.md`, `run_events.jsonl`, `failures.jsonl` | Reproducible examples, real timings, failures and explicit assistance labels |
| `checksums.json`, `acceptance.json`, `COMPLETE.json` | Artifact binding; completion only after required acceptance gates |

Reference material is stored only by the evaluator after ASR/review is sealed, not made accessible to a recognition/correction worker. A live user demo without a supplied reference has null WER/CER and an explicit reason. Evaluation failure cannot be relabeled success. Store raw and corrected TTS variants under different hashes. Do not include raw TORGO audio in a public source commit; local examples link to authorized input paths. The generated WAV and metrics must be made available in the local submission bundle even though WAV is gitignored.

## 12. Acceptance and minimal tests before sign-off

Required acceptance:

1. All seven upstream identities and all three C2 checkpoints remain unchanged; preserve the Fold1/A1 exception and UNKNOWN history. No M5 execution.
2. Locked separate environment passes import/CUDA-or-declared-CPU/model-checksum/SAPI-voice checks; offline inference works after initial acquisition.
3. Reference/panel bindings validate, development/evaluation speakers are disjoint, one primary recording per logical ID, and every exclusion has a reason. No reference reaches recognition/review/TTS.
4. All 130 planned evaluation records have immutable raw results and independently recomputable raw WER/CER; no unexplained missing/failed records. If incomplete, label PARTIAL and withhold benchmark completion. No WER/CER threshold is required for correctness.
5. Corpus, speaker and group arithmetic agrees with saved edit counts, including mean ± sample SD; all normalization/coverage limits are reported.
6. Optional corrections never change raw artifacts. Corrected evaluation is paired and assistance-labeled, or explicitly not evaluated. Never fabricate human edits to satisfy an output list.
7. A real ASR-to-TTS run produces a decodable, nonempty, audible English WAV from its recorded synthesis text; the two predetermined development examples retain their actual outcomes. Human listening confirms the file is intelligible and matches intended synthesis text, without claiming recovery of original voice or original intent.
8. Tests/lint/type checks for M6 pass; a replay on fixed development audio reproduces the transcript under the frozen execution profile. Cross-device bitwise identity is not required. Protected-file and no-overwrite gates pass; report failures honestly.

Minimum tests, created in `tests/m6_reconstruction/`:

| File | Essential cases |
|---|---|
| `test_config_artifacts.py` | Protected outputs, traversal/junctions, immutable writes, config/weight/source drift, incompatible resume |
| `test_references.py` | Composite ID join including `Session2_3`, wrong speaker/session, primary-vs-alternate, `.jpg`, `xxx`, missing/blank, sense rule vs unsupported brackets, dev/evaluation disjointness |
| `test_normalization_metrics.py` | Identity, substitution/insertion/deletion, >100%, empty hypothesis, unavailable reference, punctuation/apostrophes/digits, CER spacing, corpus weighting, independent alignment oracle on tiny fixtures, sample SD |
| `test_audio_asr.py` | Complete PCM frames including >30s synthetic file, malformed/unsupported WAV, exact backend text, frozen/no-grad options, no context carryover across records, reference-sentinel never passed to ASR |
| `test_review_tts.py` | Disabled/explicit correction, raw hash invariant, stale parent rejection, hidden reference/metrics, correct text selection, no-text outcome, missing voice, timeout, corrupt/zero WAV |
| `test_pipeline_cli.py` | Synthetic full artifact round trip, interrupted run/atomic publication, reference-free recognition, failure accounting, no M5 imports/calls, corrected/raw report separation |
| `test_runtime_smoke.py` | Explicit opt-in real model/voice tests in M6 environment; never auto-download or benchmark during ordinary pytest |

Use tiny synthetic fixtures in the default suite; never commit TORGO audio. Every implementation bug fix needs a reproducing regression test and a defect-log update. Run M6-only pytest (explicit coverage source so root M0–M5 coverage settings are not reused), Ruff, strict mypy and whitespace checks. No full M5 regression/accepted-audit rerun is needed for this isolated implementation. Real hardware smoke and WAV listening remain distinct from unit tests. No executable M6 tests ran in Phase 0 because there is no implementation.

## 13. Deadline path, decisions and session deliverables

Engineering estimate, **not measured execution time**: 8–12 focused hours after authorization, with dependency/download/setup uncertainty. Suggested sequence: isolated runtime and mechanical smoke 1–2 h; source/metric/artifact contracts 2–3 h; ASR/review/TTS/CLI 2–3 h; tests, 130-record run, demo and report 3–4 h. Use 2026-10-07 for environment/contracts and 2026-10-08 for implementation/validation, leaving 2026-10-09 for packaging and any separately authorized M7 work. Actual inference duration must be measured on development smoke and extrapolated before the panel. The 9.09 minutes of selected source audio is not a runtime promise.

No model-choice decision blocks the plan: honor `small.en` and `pyttsx3`. No dataset download, classifier choice or training is required. The following are implementation gates, not completed achievements: dependency resolution, actual checkpoint acquisition, CUDA availability in `.venv-m6`, SAPI WAV generation, target-reference rule tests and live ASR metrics. If the specified model/backend proves technically unusable, report the concrete failure before proposing a substitution. Human correction is optional; a human-supplied blind edit is required only to claim independently evaluated corrected text. A full-corpus benchmark or clinical-quality claim would require scope and evidence beyond this deadline plan.

**Decision required before the next phase:** authorize Module 6 implementation against this plan (including isolated dependency/model acquisition and bounded smoke/panel inference). The current request explicitly stops at Phase 0; no approval is assumed and no implementation is started.

This Phase-0 session creates only `PHASE0_AUDIT_AND_PLAN.md` and `PHASE0_EVIDENCE.json`, and updates the M6 module/test-report navigation plus README, PROJECT_STATUS, MODULE_TRACKER, CHANGELOG and DEFECT_LOG to record the audit and pending reference safeguards. Scientific source/configs, tests, package locks, existing environments, raw data and Module 5 artifacts remain unchanged. No WER/CER values or generated audio are claimed.

Final preservation verification: all 124 captured protected files, 9,494 raw prompt files and 134 proposed primary WAVs are unchanged. The proposed panel ID digest and development/evaluation speaker separation pass. Documentation whitespace checks pass. No M6 runtime or output directory was created; no executable feature tests were run.

**STOP: Phase 0 complete; Module 6 implementation NOT STARTED; Module 7 NOT STARTED.**
