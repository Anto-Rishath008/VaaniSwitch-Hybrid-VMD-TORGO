# Module 6 Phase 4 — updated disposition 2026-10-08

**PASS under the user's explicit five-source container exception.** Development ASR/TTS
reuse remains unchanged: zero new development forwards, zero new synthesis, correction disabled.
New preflight and freeze PASS; final 130-record computation COMPLETE.
Human listening PASS — USER CONFIRMED for both original demos; Module 6 COMPLETE / PASSED / SIGNED OFF.
See [listening evidence](HUMAN_LISTENING_REPORT.md) and [final sign-off](FINAL_SIGNOFF_REPORT.md).

See [final evaluation report](FINAL_EVALUATION_REPORT.md) and
[exception policy/proof](SOURCE_CONTAINER_EXCEPTION.md).
New freeze SHA: `79738cd85a8b251ddab654a217eba9e192ea054030fe362db8b352f4701d8fb8`.
Old blocked freeze SHA: `385c3be9ec4cdee0b5397d9cac9dc4d713952f945125592a7145940f5ee5d169` (unchanged).

The previous rejection rule was superseded only for the five verified frozen sources.
Their outer containers remain malformed; complete original PCM is explicitly validated.

---

Historical blocked-stage report (preserved; no longer the current disposition):

# Module 6 Phase 4 recovery and pre-evaluation gate

**Overall FAIL: source-audio preflight. TTS mechanical validation PASS.**
[PHASE4_EVIDENCE.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/PHASE4_EVIDENCE.json) and [FINAL_EVALUATION_PREFLIGHT.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/FINAL_EVALUATION_PREFLIGHT.json).

Correction remains **DISABLED / NOT EVALUATED**. No correction artifacts exist.
Both original demos were validated and reused without synthesis. The exact raw ASR
bytes match synthesis_text.txt and its recorded SHA. No reference substitution occurred.

| Development ID | WAV | Frames | Duration (s) | Bytes | Nonzero samples |
|---|---|---:|---:|---:|---:|
| `M03/Session2/0013` | [clear.wav](../../../artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/M03/Session2/0013/tts_raw/clear.wav) | 70428 | 3.194013605 | 140902 | 47658 |
| `MC02/Session1/0006` | [clear.wav](../../../artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z/records/MC02/Session1/0006/tts_raw/clear.wav) | 68442 | 3.103945578 | 136930 | 52559 |


Both files are RIFF/WAVE PCM16, mono, 22,050 Hz, with finite nonzero samples.
Settings: pyttsx3 2.99, SAPI5, Microsoft Zira Desktop, rate 175, volume 1.0.
The exact voice token is
`HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0`.
Human listening: **NOT PERFORMED / PENDING**. Synthesis elapsed time was not recorded
in the parent metadata and is not invented.

The exact 130 IDs/order and panel SHA validate, as do 13 speakers, 70 dysarthric /
60 control, 65 word / 65 sentence, 545.6203125 source seconds, exclusion of M03/MC02,
source hashes, reference eligibility and audio-only job schemas. Panel digest:
`15703d20490c83ac814664c48c3d8c8bf1197d2054d7d9d5ce495c68804a536b`.

However, five WAVs violate the explicit RIFF bound:

| Logical ID (all M01/Session1) | Physical bytes | RIFF ChunkSize | Declared total | Difference |
|---|---:|---:|---:|---:|
| `M01/Session1/0007` | 83884 | 83884 | 83892 | -8 |
| `M01/Session1/0008` | 129964 | 129964 | 129972 | -8 |
| `M01/Session1/0014` | 88364 | 88364 | 88372 | -8 |
| `M01/Session1/0017` | 94444 | 94444 | 94452 | -8 |
| `M01/Session1/0020` | 112364 | 112364 | 112372 | -8 |


First failing path:
`data/raw/torgo/extracted/M/M01/Session1/wav_headMic/0007.wav`.
Its fmt chunk starts at 12, payload 16 bytes, ends at 36; data chunk starts at 36,
payload 83,840 bytes, ends at physical EOF 83,884. Python wave reads all 41,920
manifest frames as PCM16 mono 16 kHz. All five PCM data chunks are complete and hashes
match Phase 0, but each outer RIFF declaration extends eight bytes beyond EOF.
The header pattern is consistent with storing total file size instead of size minus
eight; the original writer's behavior is not proven. These are malformed outer
headers, not examples of harmless trailing bytes. The other 129 selected WAVs have
exact declared lengths; no selected file has physical bytes beyond its declaration.

The bounded parser change accepts source trailing bytes only outside a structurally
complete declared container, records declared/trailing lengths, and keeps original
complete-file hashes. It rejects declarations beyond physical EOF, chunk overruns,
incomplete PCM frames and malformed declared regions. Generated WAVs remain strict.
It does not repair, normalize or rewrite these five source headers, substitute another
microphone, or remove/replace records. All selected source bytes remain unchanged.

Per the user's required rejection and stop rules, final inference did not run. Current
CUDA allocation was not rechecked after this failed gate. The retained development
CUDA probe passed; that does not turn this final preflight into a PASS.

[PRE_EVALUATION_FREEZE.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/PRE_EVALUATION_FREEZE.json) is a **BLOCKED status record, invalid for
execution**, SHA `385c3be9ec4cdee0b5397d9cac9dc4d713952f945125592a7145940f5ee5d169`. No accepted PASS freeze exists. A regression verifies
that a blocked freeze is rejected. This is an explicit source-format contract blocker,
not a transcript-quality decision.
