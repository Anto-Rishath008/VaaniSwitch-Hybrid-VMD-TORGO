# Module 6 recovery handoff — 2026-10-08

**Recovered development outputs; final evaluation BLOCKED by five source RIFF headers.**

1. Recovery state: validated and preserved 40 partial-run files and 68 Phase-0/1 files;
   no overwrites. [RECOVERY_VALIDATION.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/RECOVERY_VALIDATION.json)
2. Exact partial run found (one): `artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z`.
3. Reused ASR: all four frozen development IDs listed below, including the Phase-1
   M03/Session2/0013 output already reused by the parent.
4. New development Whisper forwards in this continuation: **0**. Parent: three new,
   one Phase-1 reuse. No new TTS synthesis.
5. RIFF cause: outer declarations overstate physical size by eight bytes; complete
   PCM subchunks, not trailing-data cases. No claim of missing recorded PCM samples.
6. First failure: M01/Session1/0007,
   `data/raw/torgo/extracted/M/M01/Session1/wav_headMic/0007.wav`.
7. Physical 83,884 bytes; ChunkSize 83,884; declared total 83,892; difference -8.
8. Same condition in 0008, 0014, 0017, 0020 of M01/Session1. Other 129 selected
   sources have exact lengths; all 134 hashes and PCM frame counts match.
9. Parser repair: source-only valid declared-container/trailing-byte support and
   metadata, structural chunk bounds and complete-frame checks; overdeclared files
   rejected; generated files remain strict. No source bytes or hashes changed.
10. Regression/contract tests: six RIFF tests plus ordered-panel hash, signed CORE,
    blocked freeze, audio-only jobs, composite joins, reference exclusions,
    normalization, failures versus empty hypotheses and aggregation. Total M6 suite: 69.
11. Phase 2: **PASS**; 9,494 records / 15 speakers / 8,673 signed CORE / 111 excluded
    CORE images / 8,562 eligible (6,458 word, 2,104 sentence).
12. Phase 3: **PASS**, recovered development/smoke only. Config binding reconstructed
    from unchanged Phase-1 ASR implementation/YAML; original record lacked config field.
13. Exact raw development transcripts, including leading spaces:

| Logical ID | Exact raw backend text (JSON string) | WER | CER | ASR seconds |
|---|---|---:|---:|---:|
| `M03/Session2/0002` | `" Now."` | 0.00000000% | 0.00000000% | 1.041873200 |
| `M03/Session2/0013` | `" Well, he is nearly 93 years old."` | 25.00000000% | 33.33333333% | 1.437337500 |
| `MC02/Session1/0007` | `" Jarg it."` | 200.00000000% | 50.00000000% | 0.234502700 |
| `MC02/Session1/0006` | `" He slowly takes a short walk into open air each day."` | 16.66666667% | 4.76190476% | 0.271736000 |


14. Development corpus WER **6/22 = 27.27272727%**, CER **16/84 = 19.04761905%**.
    No final benchmark or clinical claim. TORGO prompt references are not guaranteed
    verbatim recordings of produced speech.
15. Phase 4: **FAIL overall** because final preflight fails; TTS mechanical **PASS**.
16. Two retained demos and mechanical metadata:

| Development ID | WAV | Frames | Duration (s) | Bytes | Nonzero samples |
|---|---|---:|---:|---:|---:|
| `M03/Session2/0013` | [clear.wav](../../../artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/M03/Session2/0013/tts_raw/clear.wav) | 70428 | 3.194013605 | 140902 | 47658 |
| `MC02/Session1/0006` | [clear.wav](../../../artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/MC02/Session1/0006/tts_raw/clear.wav) | 68442 | 3.103945578 | 136930 | 52559 |


    Both PCM16 mono 22,050 Hz, finite/nonzero, Zira/SAPI5/pyttsx3 2.99, rate 175,
    volume 1.0. Mechanical validity does not establish listening quality.

17. Correction: **DISABLED / NOT EVALUATED**; zero human or automated corrections.
18. Ruff: **PASS**.
19. Strict mypy: **PASS**, 30 files, Python 3.12.
20. Pytest: **69 passed, 1 skipped** (6.01 s); opt-in retained Phase-1 evidence
    separately **1 passed** (0.43 s), with zero new runtime work. Compileall,
    whitespace and dependency compatibility pass; 44 installed distributions.
21. Preservation: **PASS**; 124 protected files, 20,366 old environment files,
    39 ledgers and 134 selected WAVs unchanged. Scientific M0-M5 code hash:
    `7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8`.
    Protocol `677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9`;
    M5 closure `449b1a8237efbb21e7912868d4a0586bdca30e85806248ca646b6d647f5683e2`.
    Fold1/A1 remains RECOVERED_PROTOCOL_EXCEPTION; original forward count UNKNOWN.
22. Freeze: `artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/PRE_EVALUATION_FREEZE.json`, SHA `385c3be9ec4cdee0b5397d9cac9dc4d713952f945125592a7145940f5ee5d169`.
    **BLOCKED / invalid for execution**; no PASS freeze exists.
23. Panel membership/hash/counts **PASS**: 130 ordered records, 13 speakers,
    70 dysarthric / 60 control, 65 word / 65 sentence, 545.6203125 seconds;
    M03/MC02 excluded. Panel SHA `15703d20490c83ac814664c48c3d8c8bf1197d2054d7d9d5ce495c68804a536b`.
    Source-format gate **FAIL** on five records; full preflight **FAIL**.
24. Evaluation-panel Whisper forwards: **0**.
25. Final evaluation: **NOT STARTED / BLOCKED**.
26. Final raw micro WER/CER and percentages: **null / NOT COMPUTED**.
27. Requested 130; attempted 0; transcribed 0; scored 0; ASR failures 0;
    evaluation exclusions 0; unattempted 130. Five preflight failures are not
    ASR runtime failures and were not removed from the panel.
28. Final per-speaker, condition, prompt-type, inherited-group and equal-weight
    three-group mean/sample-SD: **NOT COMPUTED**. Development groups are retained in
    [metrics_development_verified.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/metrics_development_verified.json) and are not final results.
29. Retained development ASR total **2.985449400 s**; min/median/max
    **0.234502700 / 0.656804600 / 1.437337500 s**; parent model loading **1.929684400 s**.
    Final runtime NOT RUN. Original TTS synthesis runtime was not recorded.
30. Recovery artifacts: `artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z`. Final status report:
    [FINAL_EVALUATION_REPORT.md](FINAL_EVALUATION_REPORT.md). No final run or
    completion evidence exists; [FINAL_EVALUATION_STATUS.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/FINAL_EVALUATION_STATUS.json) records
    the blocked state without fabricating completion.
31. Exact files requiring human listening:
    `artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/M03/Session2/0013/tts_raw/clear.wav` and
    `artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/MC02/Session1/0006/tts_raw/clear.wav` (clickable links above).
32. References never entered recognition; waveform-only model input and frozen
    decoding; reference_context_used=false in all validated records.
33. No ASR correction was used.
34. Module 5 was **not executed**; only read-only hash verification occurred.
35. Module 7 was **not started**.
36. **COMPUTATIONAL MODULE 6 INCOMPLETE; HUMAN LISTENING PENDING;
    MODULE 6 SIGN-OFF PENDING.** Stop at the explicit failed gate.

Detailed evidence: [Phase 2](PHASE2_CONTRACT_REPORT.md),
[Phase 3](PHASE3_DEVELOPMENT_REPORT.md), [Phase 4](PHASE4_PRE_EVALUATION_REPORT.md),
[test report](TEST_REPORT.md), and
[134-source RIFF diagnosis](../../../artifacts/module6/recovery_20261008/RIFF_DIAGNOSIS.json).
