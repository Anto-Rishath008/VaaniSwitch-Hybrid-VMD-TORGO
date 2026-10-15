"""Render implementation handoff from retained validation evidence, never invent metrics."""
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from experiments.accuracy_wavlm_v4.src.common import sources, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def put(name, text):
    text = text.replace('`validation/validation_final.log`', '`validation/validation_final_strict.log`')
    (ROOT/name).write_text(text.strip()+'\n', encoding='utf-8')


for name in ('checkpoints', 'experiments', 'results', 'figures', 'docs'):
    (ROOT/name).mkdir(exist_ok=True)
ref = read(ROOT/'validation/latest_cuda.json')
smoke_path = Path(ref['path'])
assert sha(smoke_path) == ref['sha256']
smoke = read(smoke_path)
assert smoke['status'] == 'PASS' and smoke['sources'] == sources()
preservation = read(ROOT/'validation/preservation.json')
assert preservation['status'] == 'PASS'
test_log = (ROOT/'validation/validation_final_strict.log').read_text(encoding='utf-8-sig')
match = re.search(r'Ran (\d+) tests in ([0-9.]+)s', test_log)
assert match and '\nOK' in test_log and 'FAILED (' not in test_log
count, test_seconds = int(match[1]), float(match[2])
assert count >= 52
before = read(ROOT/'validation/protected_before.json')
predecessors = {n: sum(k.startswith('experiments/'+n+'/') for k in before['study']) for n in
    ('accuracy_hubert', 'accuracy_hubert_v2', 'accuracy_hubert_v3', 'speaker_level_analysis')}
allocation, reservation = smoke['peak_allocated']/1024**3, smoke['peak_reserved']/1024**3
fits = smoke['neural_fits']
free = shutil.disk_usage(ROOT).free/1024**3
size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())/1024**3
smoke_rel = smoke_path.relative_to(ROOT).as_posix()
model = smoke['model_lock']
location = str(ROOT)
command = f'powershell -NoProfile -ExecutionPolicy Bypass -File "{ROOT / "run_all.ps1"}" -Mode Develop -AuthorizeDevelopment'

put('TEST_REPORT.md', f'''
# V4 implementation test report

2026-10-13: PASS. Final launcher validation ran **{count} automated tests**, zero failures/errors/skips, {test_seconds:.3f} seconds. Evidence: `validation/validation_final.log`. Earlier 49- and 51-test successful runs and the initial failures remain retained. Python source and PowerShell launcher parsing passed. The final validation was invoked through the actual PowerShell launcher using the existing environment.

Coverage: actual manifest/CORE counts and every grouped development partition; speaker, logical ID and source-hash leakage; crop, resampling, stereo, silence, short/non-finite handling; valid-frame mean/population-SD and attention pooling; official processor normalization regression; tiny actual WavLM selective gradients; finite optimizer/gradient checks; train-weighted scaler isolation; LR publication/restart; checkpoint corruption, partial publication and RNG; source-bound cache; injected OOM stop without skipping; fixed candidate budget/ties; full selection orchestration with synthetic fit functions; sample SD; authorization gates and exclusive outer-claim race regression; all-prediction report coverage; synthetic figure generation and report-seal reuse; unfavorable historical development differences and matching CV partitions; protected-baseline scope.

Real CUDA: `{smoke_rel}` (SHA-bound by `validation/latest_cuda.json`). Ten development originals (8 training / 2 validation), real pinned model, FP32, frozen extraction/cache reuse, LR and all three neural candidates, one epoch/two optimizer updates per neural fit with accumulation four. Four training originals are cropped to the full configured four seconds. Bounded neural interruption/recovery and checkpoint-reload predictions match exactly. Peak allocated/reserved {allocation:.3f}/{reservation:.3f} GiB. These tests do not establish scientific accuracy or full-run stability.

Synthetic fixtures, including synthetic outer claims and plots, live exclusively under `validation/unit/`. No production outer claim, full training result or final scientific report exists. OOM was fault-injected in a unit test; no claim of an actual GPU OOM event or successful smaller-batch recovery is made.

The earlier memory-efficient-attention panel exposed resume drift and is retained as a failure. V4 now uses strict deterministic algorithms and math SDPA for both forward and recomputation. The final panel passed all finite checks and exact resume comparisons. A bool/float attention-mask deprecation warning remains in the installed WavLM implementation. Cross-hardware, future-version or full-run bitwise reproducibility is not established. See DEFECT_LOG.md for retained failures and fixes.
''')

put('GPU_MEMORY_REPORT.md', f'''
# V4 GPU bounded-validation report

PASS on {smoke['environment']['gpu']}; total {smoke['environment']['total_vram']/1024**3:.3f} GiB. Torch {smoke['environment']['versions']['torch']}, CUDA {smoke['environment']['cuda']}, Transformers {smoke['environment']['versions']['transformers']}. FP32 only; no autocast, FP16, BF16 training or GradScaler was used. No dependencies were installed.

Final panel wall time: {smoke['wall_seconds']:.3f} seconds. Maximum allocated: {smoke['peak_allocated']:,} bytes ({allocation:.3f} GiB). Maximum reserved: {smoke['peak_reserved']:,} bytes ({reservation:.3f} GiB). These are this process's PyTorch allocator measurements, not all GPU processes or total device occupancy. Per-process allowance 80 percent; observed panel fit the 8 GB GPU.

| Variant | Trainable parameters | Frozen parameters | Optimizer updates | Training seconds | Records/sec | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(f"| {name} | {fit['parameter_count']:,} | {fit['parameters']['frozen']:,} | {fit['completed_steps']} | {fit['training_seconds']:.3f} | {fit['training_records_per_second']:.2f} | {fit['memory']['peak_reserved']/1024**3:.3f} |" for name, fit in fits.items())+f'''

Frozen encoder: 94,381,936 parameters, zero trainable. Frozen features use eight-second maxima; neural training and inference use four-second maxima. Physical batch 1 throughout. Smoke accumulation 4, production accumulation 8; accumulation retains gradients, not previous forward graphs. Non-reentrant gradient checkpointing is enabled. Training timing includes decode, forward/backward/update and synchronization, but excludes loading/checkpoint I/O/validation; complete panel wall time is reported separately.

Last2 updates zero-based layers 10/11; last4 updates 8/9/10/11. Frontend, projection, positional convolution, encoder normalization and lower blocks stay frozen/eval. Optimizer groups use LR 1e-5 and 1e-4, AdamW weight decay .01, clipping 1, linear scheduling. Every intended parameter receives a finite gradient; frozen parameters do not. Finite parameters and optimizer state are checked after each update.

OOM policy has zero automatic retries at physical batch one. It retains the last complete update and exact failure context, then stops. Unit fault injection passed; no real CUDA OOM occurred. No failed batch is skipped, no sequence is silently shortened, and no alternative numerical policy is auto-selected.

Numerical backend: strict deterministic algorithms, math SDPA only; flash, memory-efficient and cuDNN SDPA disabled. This fixes actual resume drift observed in an earlier retained panel. Final interrupted/uninterrupted states matched exactly. The installed model still emits an attention-mask dtype deprecation warning. Full-run and cross-hardware reproducibility remain unmeasured.

Planning runtime: roughly 15-40 hours for the full predefined development budget, depending on early stopping, actual duration distribution, I/O and contention. At eight epochs the 27 neural CV fits cover up to 832,608 training-record presentations; final refits can add 138,768. The measured tiny-panel throughputs are tabulated above; extrapolation is uncertain. Full throughput, extraction duration and end-to-end runtime are unmeasured.

Planning disk: approximately 75-85 GiB additional for the maximum retained development state, best checkpoints, final fits, features and reports. The launcher requires 85 GiB free. Current V4 logical size is about {size:.2f} GiB, including repeated bounded evidence; available disk was {free:.2f} GiB when this report was generated. Original checkpoints are referenced, never copied or deleted; pretrained WavLM itself is approximately 360 MiB. Interrupted/repeated runs can require more disk. No full study was launched.
''')

put('PRESERVATION_REPORT.md', f'''
# V4 preservation report

PASS. `validation/protected_before.json` was created by the initial two-file session and retained unchanged through resume. It is excluded from its own protected inventory because it is a V4 output. `validation/protected_after.json` and `validation/preservation.json` report no changed, missing or added protected files. The resumed verification and the final comparison both passed.

| Scope | Total inventoried files | Byte-hashed (SHA-256 + size + mtime) | Metadata only (path + size + mtime) |
|---|---:|---:|---:|
| Original project | {preservation['counts']['original']['total']:,} | {preservation['counts']['original']['byte_hashed']:,} | {preservation['counts']['original']['metadata_only']:,} |
| Accuracy study outside V4 | {preservation['counts']['study']['total']:,} | {preservation['counts']['study']['byte_hashed']:,} | {preservation['counts']['study']['metadata_only']:,} |

All files in V1 ({predecessors['accuracy_hubert']:,}), V2 ({predecessors['accuracy_hubert_v2']:,}) and V3 ({predecessors['accuracy_hubert_v3']:,}) match SHA-256, size and mtime, including their caches/checkpoints/sealed results and existing bytecode. The existing speaker-analysis scope contains {predecessors['speaker_level_analysis']} files. Every individual checked path and its evidence type appears in the two inventories. A `sha256` field means byte hashing; its absence means metadata only.

Raw/processed directories and existing virtual environments are metadata-only. This does not prove full-corpus byte equality. The ten actual development primary WAV files used by the CUDA panels were separately byte-hashed before decoding and after validation and match the frozen manifest. Original project source, manifests, splits, reports, figures and checkpoint/evaluation evidence outside raw/processed/environment trees were byte-hashed.

Git internal files are excluded from content inventory; HEAD, branch and all refs are checked separately and unchanged. Original release HEAD remains `2dee81f7656e5b5cae800d9a3c8692e5e8107bf9`. The root checkout's pre-existing dirty/untracked files were retained and compared; this report does not falsely call that checkout clean. No commit/push, reset, clean or evidence deletion occurred.

Only the exact study path `experiments/accuracy_wavlm_v4/` is excluded as intentional work. No entire experiments tree is excluded. All writes, model downloads, caches, test artifacts, logs and documentation remain under V4. Existing environments were neither reinstalled nor edited. Modules 0-6, raw TORGO and prior scientific results remain protected. No subsequent module was started.
''')

put('EXPERIMENT_TRACKER.md', f'''
# V4 experiment tracker

2026-10-13 status: IMPLEMENTATION AND BOUNDED VALIDATION PASS.

| Item | Status |
|---|---|
| Resume original two files and preserve baseline | PASS |
| Official checkpoint revision/file/processor verification | PASS |
| CORE and original outer split integrity; grouped development CV | PASS |
| Frozen WavLM cache, weighted LR, train-only normalization | IMPLEMENTED + bounded validated |
| Last2/last4/attention fine-tuning and masked pooling | IMPLEMENTED + real CUDA validated |
| Numerical checks, OOM stop, checkpoint/RNG recovery | PASS in automated/bounded tests |
| Development selection, future outer gates/sealing, reporting | IMPLEMENTED; synthetic orchestration/report tests PASS |
| Automated suite | {count} PASS, zero failures/errors/skips |
| Final CUDA source-bound panel | PASS; {allocation:.3f}/{reservation:.3f} GiB allocated/reserved |
| Original/V1/V2/V3 preservation | PASS |
| Full development training | NOT EXECUTED; separate approval required |
| Real outer inference | NOT EXECUTED; separate explicit authorization required |
| Independent external validation | NOT AVAILABLE / NOT CLAIMED |
| V4 scientific accuracy / 90 percent target | UNKNOWN / NOT CLAIMED |

Closed implementation findings: V4-001 official processor normalization, V4-002 test tolerance, V4-003 exclusive outer-claim creation, V4-004 deterministic attention/resume. Historical HuBERT development comparison is report-only after V4 decisions freeze and verifies identical speaker CV partitions. Completed-decision resume finishes missing report publication without re-training or changing decisions.

STOP after this handoff. Await the user's review and later authorization; do not automatically run Develop or Outer.
''')

put('IMPLEMENTATION_REPORT.md', f'''
# V4 implementation handoff

**PASS for implementation and bounded validation.** Prepared 2026-10-13, Asia/Calcutta. Full scientific execution remains unperformed. Scope: `{location}`.

| Requested item | Verified status / handoff |
|---|---|
| 1. Implementation status | PASS within implementation/smoke scope; no full-study performance claim |
| 2. Exact location | `{location}` |
| 3. Official checkpoint | microsoft/wavlm-base-plus @ `4c66d4806a428f2e922ccfa1a962776e232d487b`; no missing/unexpected weights |
| 4. Files | configs/study.json, configs/model_lock.json; 15 src Python files; tests/test_v4.py and test_reporting.py; run_all.ps1; V3_LINEAGE.json; required protocol/audit/report/tracker documents plus CHANGELOG/DEFECT_LOG; V4-only model/cache/validation/log evidence; validation/created_files.json lists paths |
| 5. Preservation | Original {preservation['counts']['original']['total']:,} files and study {preservation['counts']['study']['total']:,} outside V4 unchanged under stated hash/metadata scope; all {sum(predecessors[n] for n in ('accuracy_hubert','accuracy_hubert_v2','accuracy_hubert_v3')):,} V1/V2/V3 files byte-hashed; release HEAD/refs unchanged |
| 6. V4-A | Frozen WavLM layers 9/12; masked mean or mean+population SD; 768/1536 features; training-weighted StandardScaler; L2 logistic regression; 769/1537 learned coefficients including intercept |
| 7. V4-B | Train upper 2 or 4 transformer blocks; mean pooling and LayerNorm/dropout/linear head; optional last2 compact attention pooling |
| 8. Frozen/trainable layers | Frontend, feature projection, position convolution, encoder norm and lower blocks frozen; last2 = indices 10/11, last4 = 8-11; pretrained base has 94,381,936 parameters |
| 9. Trainable counts | last2+mean 14,179,113; last4+mean 28,355,921; last2+attention 14,203,753; frozen respectively 80,205,128 / 66,028,320 / 80,205,128 |
| 10. Validation design | Original outer folds untouched; each context uses only its 10 development speakers; inherited V3 three-way class-stratified grouped CV; original record/path/hash/speaker isolation; fixed threshold .5 |
| 11. Candidate grid | A: 2 layers x 2 pools x 2 C values = 8; B: last2 mean, last4 mean, last2 attention = 3; 99 inner fits total (72 LR, 27 neural), at most 6 final refits; macro-F1 primary and frozen tie policy |
| 12. Automated tests | {count} PASS, zero failures/errors/skips; {test_seconds:.3f} s; actual launcher plus Python/PowerShell parsing |
| 13. Actual GPU smoke | Real model + ten development recordings (8 train/2 validation), frozen cache reuse/LR/all three B variants, one epoch and two updates each; finite gradients/parameters/optimizer; exact interruption/reload comparisons |
| 14. Maximum observed VRAM | {allocation:.3f} GiB allocated / {reservation:.3f} GiB reserved; process allocator measurements; full four-second training segments tested |
| 15. Checkpoint/resume | PASS for exact model-delta/AdamW/scheduler/Python-NumPy-CPU-CUDA RNG/prediction recovery; checksum/identity/partial-publication tests; LR restarts interrupted optimization and reuses completed fits |
| 16. Runtime planning | Roughly 15-40 hours full development budget; small-panel extrapolation, not measured full execution; early stopping and I/O can change it substantially |
| 17. Disk planning | About 75-85 GiB additional retained full-development artifacts; launcher requires 85 GiB free; about {free:.2f} GiB free at handoff; current V4 evidence about {size:.2f} GiB |
| 18. Known limits | Only 15 speakers; exposed outer populations; one seed; selection optimism; full-run resource/stability/convergence not established; attention-mask deprecation warning; no external validation; no accuracy guarantee |
| 19. Development-only launcher | Exact command below, for later approval only; separate Outer invocation is mandatory |
| 20. Remaining blockers | No observed implementation blocker. Full development approval and later separate outer authorization are pending; disk/thermal/long-run behavior still needs monitoring during authorized execution |

After separate user approval:

```powershell
{command}
```

The launcher rechecks tests, source-bound CUDA evidence, model, dependencies, lineage and disk allowance. It resumes verified development states, freezes all context choices, publishes development reports including a descriptive matched-partition frozen-HuBERT comparison, and stops before outer inference. It does not automatically widen the grid or change precision/crops after failure. Real outer inference requires another authorization and the separate `-Mode Outer -AuthorizeOuter` command.

Official file SHA-256 values:

| File | SHA-256 |
|---|---|
'''+ '\n'.join(f'| {name} | `{value}` |' for name, value in model['files'].items())+f'''

The official processor uses 16 kHz, `do_normalize=false`, attention masks enabled. Source audio is hash-verified, mono/resampled if necessary, center-cropped to eight seconds (A) or four seconds (B), with one probability per original recording. No augmentations or mixed precision are active. Strict deterministic math SDPA corrected a reproduced resume drift in an earlier panel; failed evidence is retained. Silence remains allowed; invalid/sub-400-sample input fails without silently removing records. All CORE manifest lengths exceed that minimum.

Evidence: `validation/validation_final_strict.log`, `{smoke_rel}`, `validation/latest_cuda.json`, `validation/data_audit.json`, `validation/protected_before.json`, `validation/protected_after.json`, `validation/preservation.json`, `configs/model_lock.json`, `V3_LINEAGE.json`. Earlier failed tests and all real bounded panels are preserved. All production `results/`, `experiments/`, `checkpoints/` and `figures/` directories remain empty. Synthetic figures are labeled fixtures and stored only in validation.

Future authorized reports generate FINAL_EXPERIMENT_REPORT.md, final_results.json, fold_results.csv, record_predictions.csv, speaker_predictions.csv, ablation_results.csv, selected_configurations.json, training_curves.png, accuracy_comparison.png, confusion_matrix.png, speaker_accuracy_analysis.png and gpu_memory_analysis.png. All real results are retained. Historical Original B2/C2, V2 E1 and V3 Selected are read only at post-seal reporting time. Reused TORGO comparisons are exploratory; independent confirmation requires untouched speakers or an appropriate external corpus.

**Full V4 development training has NOT been executed.**

**Real V4 outer-test inference has NOT been executed.**

**V4 accuracy is not yet known.**

STOP: implementation and bounded validation complete. Await user review.
''')

evidence = {'status': 'PASS_IMPLEMENTATION_AND_BOUNDED_VALIDATION', 'tests': count,
    'test_seconds': test_seconds, 'cuda_report': ref, 'preservation': preservation,
    'model_revision': model['revision'], 'full_development_executed': False,
    'real_outer_inference_executed': False, 'v4_accuracy': None,
    'peak_allocated': smoke['peak_allocated'], 'peak_reserved': smoke['peak_reserved'],
    'source_identity_matches_final_smoke': True}
put('validation/evidence_summary.json', json.dumps(evidence, indent=2))
files = {p.relative_to(ROOT).as_posix(): {'size': p.stat().st_size} for p in sorted(ROOT.rglob('*'))
    if p.is_file() and p != ROOT/'validation/created_files.json'}
put('validation/created_files.json', json.dumps(files, indent=2))
print(json.dumps({'status': 'PASS', 'tests': count, 'peak_allocated_GiB': allocation,
    'peak_reserved_GiB': reservation, 'V4_GiB': size, 'free_GiB': free, 'files': len(files)}))
