# Module 6 Phase 3 recovered development outputs

**PASS — DEVELOPMENT / SMOKE; NOT FINAL BENCHMARK.** [PHASE3_EVIDENCE.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/PHASE3_EVIDENCE.json)

Parent: `artifacts/module6/pre_evaluation/phase234_20261008T074831779949Z`. Exactly one partial pre-evaluation directory was found.
All four raw outputs were validated and reused; this continuation performed **0 new
development Whisper forwards**. The parent performed three new forwards and reused
M03/Session2/0013 from Phase 1. No successful record was transcribed again.

| Logical ID | Exact raw backend text (JSON string) | WER | CER | ASR seconds |
|---|---|---:|---:|---:|
| `M03/Session2/0002` | `" Now."` | 0.00000000% | 0.00000000% | 1.041873200 |
| `M03/Session2/0013` | `" Well, he is nearly 93 years old."` | 25.00000000% | 33.33333333% | 1.437337500 |
| `MC02/Session1/0007` | `" Jarg it."` | 200.00000000% | 50.00000000% | 0.234502700 |
| `MC02/Session1/0006` | `" He slowly takes a short walk into open air each day."` | 16.66666667% | 4.76190476% | 0.271736000 |


Development corpus WER: **6/22 = 0.2727272727 (27.27272727%)**.
Development corpus CER: **16/84 = 0.1904761905 (19.04761905%)**.
Four requested, four transcribed/scored, zero ASR failures and zero exclusions.
These are agreement scores against TORGO prompt-based references, not guaranteed
verbatim recordings of produced speech. No clinical accuracy claim is made.

Source SHA, original complete float32 input hash/frame count, model SHA, device/dtype,
raw text bytes/hash, backend result hash/text, reference-context flag, normalizer and
metric identity were checked. Per-record and CSV metrics were recomputed from the
retained raw strings and matched. The source ASR implementation and YAML config
remain byte-identical to Phase 1. Config identity is
`60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209`. The partial metadata lacked its own config field:
the config binding is a documented recovery verification, not a contemporaneous seal.

Whisper small.en remains frozen on CUDA float16 (parameter storage float32), RTX 4070
Laptop GPU. References never entered recognition: recognition jobs contain only audio
bindings and the model call receives the waveform plus frozen decoding settings.
No correction, model change, alternate decoding, reference prompting or metric-driven
tuning occurred.

Retained ASR runtime totals **2.985449400 s** excluding
loading; min/median/max **0.234502700 /
0.656804600 / 1.437337500 s**.
Parent model loading: 1.929684400 s. These retained
timings combine Phase-1 reuse and the three parent forwards; they are not a new run.

[metrics_development_verified.json](../../../artifacts/module6/recovery_20261008/recovered_20261008T111946461345Z/metrics_development_verified.json) contains all development group
summaries. Only inherited groups 2 and 3 occur in this development sample; no
three-group mean/sample-SD is reported for it.
