# Module 6 Phase 1 runtime report

Date: 2026-10-08, Asia/Calcutta. Deadline: 2026-10-09.

**Phase 0: PASS. Phase 1: PASS (CUDA). Module 6 implementation: IN PROGRESS. Module 6: NOT SIGNED OFF.**

[Machine-readable evidence](PHASE1_RUNTIME_EVIDENCE.json) is the detailed runtime/configuration record.

## Interruption recovery

`INTERRUPTION_RECOVERY_STATE = PARTIAL_SAFE`.
The interrupted attempt created only `artifacts/module6/setup/preservation_before.json`.
It captured 124 matching protected files, 20,366 existing `.venv` files and 39 ledger files.
There was no `.venv-m6`, Module 6 config/code, installed package, model or partial download,
ASR execution, TTS execution, or Phase-1 report. Git status/diff and process state were inspected;
all unrelated pre-existing changes were preserved. The completed snapshot was validated and reused.
Nothing was deleted or recreated. Phase-0 evidence and plan were read before implementation.

## Environment and dependency resolution

Created `.venv-m6` with CPython **3.12.2**, isolated system site-packages.
Resolved/installed **44** distributions from the standalone hashed Windows lock; no root project
installation, `uv sync`, root lock edits, or `.venv` package changes occurred.
The [official CUDA 12.6 index](https://download.pytorch.org/whl/cu126/torch/) provided the proposed
Windows `cp312-cp312-win_amd64` torch **2.13.0+cu126** wheel.
Whisper **20250625**, pyttsx3 **2.99**, JiWER **4.0.0** and NumPy **1.26.4** were retained.
No proposed runtime package version required substitution. PyYAML and the separately pinned
pytest/coverage/Ruff/mypy tools support this isolated implementation.

- Torch CUDA build: **12.6**; CUDA availability: **True**.
- Device: **NVIDIA GeForce RTX 4070 Laptop GPU**, capability 8.9, driver **617.14**, 8,188 MiB.
- OS: Windows 11 Pro 10.0.26300; CPU: i9-13980HX; RAM: 33,959,800,832 bytes.
- CUDA tensor allocation and reduction: **PASS**; small.en load on `cuda:0`: **PASS**.
- Model load: **2.135146800 s**.
- ASR compute profile: **float16**, 8 CPU threads; Whisper parameter storage: **torch.float32**.
- Every parameter has gradients disabled; inference runs under `torch.inference_mode()`.
- CPU fallback (`cpu`, float32, 8 threads) is declared and untested; it was not needed.
- Lock SHA-256: `49cc7da351e802a277bd5435dc520c35e1ad15b36fc8d52b4c6f9cc0bc5d6af1`.
- Config SHA-256: `60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209`.
- Final M6 runtime-code SHA-256: `389867ef6a26c3802ed6d02796f115aed88d1b0264590293a688bcc59f1c38fc`.

An import probe ran prematurely during package copying and failed; the completed installation
passed imports, `uv pip check`, CUDA allocation, model loading and transcription. No dependency
changes were made in response to that premature probe.

### Exact installed distributions

| Distribution | Version |
|---|---|
| Jinja2 | 3.1.6 |
| MarkupSafe | 3.0.4 |
| PyYAML | 6.0.3 |
| Pygments | 2.21.0 |
| RapidFuzz | 3.14.6 |
| certifi | 2026.7.22 |
| charset-normalizer | 3.5.2 |
| click | 8.5.0 |
| colorama | 0.4.6 |
| comtypes | 1.4.17 |
| coverage | 7.16.2 |
| filelock | 4.0.12 |
| fsspec | 2026.9.0 |
| idna | 3.20 |
| iniconfig | 2.3.1 |
| jiwer | 4.0.0 |
| llvmlite | 0.50.0 |
| more-itertools | 11.1.0 |
| mpmath | 1.3.0 |
| mypy | 1.18.2 |
| mypy_extensions | 1.1.0 |
| networkx | 3.7 |
| numba | 0.68.0 |
| numpy | 1.26.4 |
| openai-whisper | 20250625 |
| packaging | 26.3 |
| pathspec | 1.1.1 |
| pluggy | 1.6.0 |
| pypiwin32 | 223 |
| pytest | 8.4.2 |
| pytest-cov | 7.0.0 |
| pyttsx3 | 2.99 |
| pywin32 | 312 |
| regex | 2026.9.29 |
| requests | 2.34.2 |
| ruff | 0.14.0 |
| setuptools | 84.0.0 |
| sympy | 1.14.0 |
| tiktoken | 0.14.0 |
| torch | 2.13.0+cu126 |
| tqdm | 4.70.1 |
| types-PyYAML | 6.0.12.20250915 |
| typing_extensions | 4.16.0 |
| urllib3 | 2.8.0 |

The launcher-visible metadata additionally discovers the checkout's `vaaniswitch-final-hybrid-vmd-torgo`
1.0.0 via `src`; this is not an installation into `.venv-m6`. The standalone installed list above,
`artifacts/module6/setup/installed_packages.json`, freeze text and hash lock establish installed state.

## Model acquisition and raw ASR

Canonical checkpoint: `artifacts/module6/models/whisper/small.en.pt`.
Size: **483,615,683 bytes**. Acquisition: **46.210716 s**.
Independent SHA-256 verification passed before loading and again at closure:

`f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872`

[Official model source](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt). The installed backend is the pinned
[openai-whisper release](https://pypi.org/project/openai-whisper/20250625/).
No default-user Whisper cache is used. Runtime inference never downloads a model implicitly.

Development record: **M03/Session2/0013**.
Read-only manifest primary: `data/raw/torgo/extracted/M/M03/Session2/wav_headMic/0013.wav`.
Source SHA-256: `e4f57347c2442e49f650e1d4adf2b73bab47e248220c07ef42c35972a8c87689` — matches Phase 0.
Complete PCM16 mono 16 kHz audio: **61,360 frames / 3.835 s**; no crop or resampling.
No reference text, dictionary, target vocabulary, prefix or initial prompt entered the model interface.
The recognition worker receives an audio-only manifest projection. No reference join or metric runs.

Exact raw backend text (JSON representation; the initial space is preserved):

```json
" Well, he is nearly 93 years old."
```

Actual synchronized CUDA transcription time: **1.437337500 s**.
This excludes model loading and environment setup. Exactly **one** real ASR transcription occurred.
The exact backend dictionary, text bytes, options, device, dtype, timing and audio bindings are retained.

Decoding: English transcription, scalar temperature 0.0, greedy (`beam_size=None`, `best_of=None`),
`initial_prompt=None`, `prefix=None`, `condition_on_previous_text=False`, `carry_initial_prompt=False`,
`word_timestamps=False`, `clip_timestamps="0"`, CUDA `fp16=True`. Remaining backend defaults are explicit
in `pipeline.yaml` and `config.resolved.json`, including suppression, thresholds and punctuation.
No manual correction, WER/CER, sample replacement or quality tuning occurred.

## TTS and retained failure evidence

Backend **pyttsx3 2.99 / sapi5**. Actual enumerated identity:
**Microsoft Zira Desktop - English (United States)**; `en-US`, Female, Adult.
Exact token:

```text
HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0
```

Actual selected token, rate **175**, and volume **1.0** were checked before saving.
Exact synthesis input:

```text
This is the VaaniSwitch English speech synthesis runtime test.
```

Accepted WAV: [`clear.wav`](../../../artifacts/module6/60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209/phase1_20261008T035618759610Z/tts/clear.wav).
SHA-256: `67c77b7c5778bb8d6e2771055af530fd5068dac8bfab95d1ff202c22290ad370`.
RIFF/WAVE, PCM16, **22,050 Hz**, **1 channel**, **80,675 frames**, **3.6587301587301586 s**,
**161,396 bytes**, **64,528 nonzero samples**, all decoded samples finite.
Successful worker synthesis/validation runtime: **0.354845500 s**.
No human listening check is claimed.

First run `phase1_20261008T035139679012Z`: ASR PASS, TTS timeout after **60 s**.
Its `FAILURE.json`, voice identity, exact input and `clear.pending.wav` are preserved.
`M6-TTS-001` was reproduced by a synthetic queue-lifecycle regression and corrected with public
external-loop configuration flushing followed by `save_to_file` + `runAndWait`.
A first resume then rejected UTF-8 JSON under Windows' default CP1252 (`M6-IO-001`), before either
worker ran. Byte-based JSON parsing fixes this; a forced-CP1252 regression proves it.

Successful run: `artifacts/module6/60db6bb2d5d8ae87716e1bbd3b990a5935677eac294acfc616d2458410156209/phase1_20261008T035618759610Z`.
The retry validates and copies the successful ASR bytes into a new run and binds their parent hashes
in `asr_reuse.json`; it performs **zero** additional ASR forwards. Only TTS was retried.
The original failed run remains failed, and its evidence is unchanged.
`RUNTIME_COMPLETE.json` binds all successful smoke artifacts and states its limited Phase-1 scope.

## Tests and preservation

- Ordinary M6-only pytest: **45 passed, 1 explicit runtime test skipped**, 6.27 s.
- Explicit retained real-runtime evidence check: **1 passed**, 0.29 s; no inference rerun.
- Ruff: **PASS**. Strict mypy: **PASS, 16 files**, explicit Python 3.12 target.
- `uv pip check`: **PASS**, all 44 installed packages compatible. Actual CLI help: **PASS**.
- Whitespace/file inventory checks: **PASS**, 28 created/modified text files; successful runtime artifact and source hashes still match.
- Synthetic tests cover config/schema drift, model checksum rejection, immutable JSON/text writes,
  traversal/junction/hardlink protection, complete >30-second audio, invalid/empty/silent WAVs,
  audio-only inference/no-grad/options, exact text, explicit device, exact voice, bounded worker,
  SAPI queue lifecycle, UTF-8 retained-ASR loading and tamper rejection.
- Ordinary suite statement-plus-branch coverage: **54.592721%** (257/437 lines,
  58/140 branches). Real subprocess runtime validation is separate. Coverage is reported without
  inheriting M0-M5's source list or threshold; it is not represented as full Module-6 test closure.
- Regression records: M6-CONFIG-001 (3 initial failures), M6-TTS-001 (1 initial failure plus timeout),
  M6-IO-001 (1 initial failure plus UnicodeDecodeError): **CLOSED**, final tests and relevant runtime pass.

Final protected comparison **PASS**: all **124** Phase-0 files, **20,366** `.venv` files,
**39** ledger files, and **134** selected primary WAVs unchanged. This is a targeted hash comparison,
not a rerun of Module 5 audits or a full raw-tree inventory. Root dependency locks are byte-identical.
All closure-package evidence and all three C2 checkpoints match their Phase-0 hashes.

- Frozen M0-M5 scientific-code SHA-256: `7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8` (92 files).
- Protocol/config identity: `677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9`; defining code/protocol bytes unchanged.
- Module 5 closure SHA-256: `449b1a8237efbb21e7912868d4a0586bdca30e85806248ca646b6d647f5683e2`.
- Module 6 handoff SHA-256: `36f5a276137922fd585dfe052e2402218fe36a4b0569384e0c4bd4e241b2235f`.
- Fold1/A1 remains **RECOVERED_PROTOCOL_EXCEPTION**, original prior native-forward count **UNKNOWN**.

## Deviations and limits

No ASR/TTS backend, model, voice, runtime dependency proposal, decoding or sample-selection deviation.
- **M6-TTS-001**: Flush settings using public startLoop(False)/iterate/endLoop, then save_to_file/runAndWait in a fresh queue barrier. pyttsx3 2.99 idle-proxy synchronous save stranded the second runAndWait; the first attempt was stopped at 60 seconds. Backend, version, Zira, rate and volume unchanged.
- **PHASE1_LAYOUT**: Use a minimal Phase-1 smoke directory and explicit runtime launcher; reference/metric/review/full CLI layers are deferred. This authorization is runtime foundation only, not the entire Phase-0 implementation plan.
- **CONFIG_LOCATION**: Freeze operational ASR/TTS policy in configs/module6/pipeline.yaml; existing configs/asr/default.yaml and configs/tts/default.yaml unchanged. No modification of those placeholders is required for the isolated launcher.

- Mechanical smoke only, not a benchmark or Module 6 sign-off. No WER/CER was computed.
- Only one predetermined 3.835-second development recording was transcribed. No quality-based sample selection or tuning.
- No listening check was performed; WAV validity does not establish perceptual intelligibility.
- CPU fallback is declared (CPU/float32/8 threads) but unexercised because CUDA succeeded.
- ASR decoding uses float16 on CUDA; checkpoint parameter storage is torch.float32, as recorded by Whisper.
- The ordinary synthetic suite covers 54.592720970537265 percent of Module 6 statements plus branches. Runtime subprocess/model/acquisition branches are independently smoke-validated, not included in that coverage number. No Phase-1 coverage percentage gate was specified.
- Runtime environment metadata lists the local project distribution because the launcher exposes src metadata. It was not installed into .venv-m6; the standalone installed inventory and lock contain 44 distributions.
- Preservation is targeted: 124 Phase-0 files, the complete prior .venv file inventory, 39 ledger files and 134 planned WAV hashes. No full raw corpus/cache audit or scientific-result recalculation.
- Windows voice output is platform-dependent; no cross-device or cross-installation byte identity claim.
- M6-REF-001 remains OPEN for the next phase; no reference eligibility/normalization/metric implementation occurred.

## Files and reproducible commands

Created source/config/test/report files:
- `configs/module6/environment/coverage.ini`
- `configs/module6/environment/requirements-win-py312.lock`
- `configs/module6/environment/requirements.in`
- `configs/module6/pipeline.yaml`
- `docs/modules/6_english_asr_tts/PHASE1_RUNTIME_EVIDENCE.json`
- `docs/modules/6_english_asr_tts/PHASE1_RUNTIME_REPORT.md`
- `scripts/module6_preservation.py`
- `scripts/module6_runtime.py`
- `src/vaaniswitch/m6_reconstruction/artifacts.py`
- `src/vaaniswitch/m6_reconstruction/asr.py`
- `src/vaaniswitch/m6_reconstruction/audio.py`
- `src/vaaniswitch/m6_reconstruction/config.py`
- `src/vaaniswitch/m6_reconstruction/provenance.py`
- `src/vaaniswitch/m6_reconstruction/runtime.py`
- `src/vaaniswitch/m6_reconstruction/tts.py`
- `tests/m6_reconstruction/conftest.py`
- `tests/m6_reconstruction/test_audio_asr.py`
- `tests/m6_reconstruction/test_config_artifacts.py`
- `tests/m6_reconstruction/test_resume.py`
- `tests/m6_reconstruction/test_runtime_smoke.py`
- `tests/m6_reconstruction/test_tts_runtime.py`

Modified live files:
- `.gitignore`
- `PROJECT_STATUS.md`
- `MODULE_TRACKER.md`
- `CHANGELOG.md`
- `docs/modules/6_english_asr_tts/MODULE.md`
- `docs/modules/6_english_asr_tts/TEST_REPORT.md`
- `docs/defects/DEFECT_LOG.md`

Local generated trees: `.venv-m6/` and `artifacts/module6/` (model, lock/install/test logs,
initial snapshot, failed/successful run evidence and smoke WAV). Large runtime/model caches are ignored.
No existing upstream source/config/checkpoint, `__init__.py`, root lock or raw file was edited.

```powershell
# Fresh-environment reproduction only; the completed environment already exists.
uv venv .venv-m6 --python 'C:\Users\antor\AppData\Local\Programs\Python\Python312\python.exe' --no-project
uv pip install --python .venv-m6/Scripts/python.exe --require-hashes -r configs/module6/environment/requirements-win-py312.lock --cache-dir artifacts/module6/setup/uv-cache --link-mode copy

# Ordinary synthetic tests: no download or real model invocation.
.venv-m6/Scripts/python.exe -B -m pytest tests/m6_reconstruction -q

# Explicit smoke commands, already completed; do not rerun to improve the transcript.
.venv-m6/Scripts/python.exe -B scripts/module6_runtime.py acquire
.venv-m6/Scripts/python.exe -B scripts/module6_runtime.py smoke

# Read-only preservation check.
python -B scripts/module6_preservation.py
```

## Acceptance and stop boundary

All Phase-1 gates A-P pass. **No 130-record evaluation, final WER/CER, correction evaluation,
Module 5 execution/training/inference/checkpoint deserialization or Module 7 work occurred.**
One actual CUDA ASR transcription and two TTS save attempts (one timeout, one accepted output) are
explicitly accounted for. There is no final Module 6 sign-off or perceptual-quality claim.

Recommended next phase: **Module 6 Phase 2 — reference eligibility, normalization/metric and
immutable transcript/artifact contracts with synthetic tests**. Preserve the Phase-1 runtime,
keep M6-REF-001 open until implemented, and require separate authorization before real panel scoring.

**STOP after Phase 1.**
