# VaaniSwitch HuBERT accuracy study

**POST-HOC EXPLORATORY IMPROVEMENT STUDY.** Implementation and bounded development
smoke only. Full extraction, complete training and real outer evaluation await the
user's manual launch. No 80–90% outcome is asserted.

The original signed project remains read-only. This is a separate experiment on
`experiment/dysarthria-accuracy-improvement`, baseline
`2dee81f7656e5b5cae800d9a3c8692e5e8107bf9`. No Module 7 work, commit or push.

Read [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md),
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), [ARCHITECTURE.md](ARCHITECTURE.md)
and [TEST_REPORT.md](TEST_REPORT.md) before the full run.

From any PowerShell location, the complete noninteractive run is one command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert\run_all.ps1"
```

The launcher uses the existing `.venv-accuracy`; it installs nothing. It verifies
Python/CUDA/disk/data/source identities, audits speaker partitions, obtains and
verifies the pinned HuBERT checkpoint, extracts and validates features, selects
and trains all E1/E2/E3 fold models, freezes every decision, evaluates all three
outer folds sequentially, aggregates results, writes figures and checks preservation.
GPU work is serial. CPU threads and decoding workers are bounded at two.

Rerun exactly the same command to resume. Completed feature entries, model fits,
selections, checkpoints and sealed outer results are checked and reused. An
unfinished epoch restarts from its last complete checkpoint, including optimizer,
AMP and RNG state. Immutable checkpoint generations and stage event history remain.
A consumed outer claim without a checksum-bound result and seal **stops for explicit
review**. There is no automatic claim reset or outer retry. Do not edit source,
config, dependencies or inputs during a run. Changes require a separately recorded
new study; the launcher refuses to silently adopt them.

| Location | Purpose |
|---|---|
| `src/` | Isolated data, feature, classifier, training, orchestration and reporting code |
| `tests/` | Focused contracts, CUDA recovery and synthetic orchestration tests |
| `cache/checkpoint/` | Official pinned model and local verification manifest |
| `cache/features/` | Full-run immutable per-record embeddings; created only by manual full run |
| `checkpoints/` | Full-run candidate/final fits; created only by manual full run |
| `smoke/` | Explicitly separate smoke namespaces and temporary synthetic tests |
| `logs/` | Preflight, progress, stage events, preservation and smoke evidence |
| `results/` | Full-run frozen decisions, sealed outer cells and final reports |
| `plots/` | Full-run training curves, recording-accuracy comparison and confusion matrices |

After the full run, `results/` contains `final_results.json`, `fold_results.csv`,
`record_predictions.csv`, `speaker_predictions.csv` and `FINAL_EXPERIMENT_REPORT.md`.
`plots/` contains `training_curves.png`, `accuracy_comparison.png` and
`confusion_matrix.png`. These are generated from observed full results; they are
intentionally absent at implementation handoff. Synthetic test exports are created
only inside temporary smoke directories.

The dedicated `.gitignore` excludes generated caches, embeddings, checkpoints,
logs, smoke runs, results and figures. Raw audio stays in the original checkout.
The pre-existing untracked `.venv-accuracy/` is not added to Git. Do not stage it.

Short, separately scoped verification commands from the experimental worktree:

```powershell
& .\.venv-accuracy\Scripts\python.exe -B .\experiments\accuracy_hubert\src\pipeline.py audit
$env:ACCURACY_CUDA_FEATURE_TEST = '1'
& .\.venv-accuracy\Scripts\python.exe -B -m unittest discover -s .\experiments\accuracy_hubert\tests -p 'test_*.py' -v
```

`pipeline.py smoke` creates a new, explicitly recorded six-record development
smoke namespace. An unfinished smoke can be resumed using
`smoke --smoke-namespace <absolute-existing-smoke-directory>`. A completed smoke
cannot be overwritten. The `audit` and `smoke` commands never call the full runner.
