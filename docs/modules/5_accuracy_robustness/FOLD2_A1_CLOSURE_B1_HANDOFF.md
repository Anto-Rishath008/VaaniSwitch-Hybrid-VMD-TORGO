# Fold2/A1 closure and Fold2/B1-only handoff

Prepared from the supplied handoff evidence after manual Fold2/A1 completion.

## Independently checked from the supplied bundle

- Fold2/A1 native `outer_result.json` SHA-256:
  `13252f3660e5066c31dbd2ad0b2d0a9e6f85fd664c7a07cecce1f4f8389ff76b`
- Fold2/A1 claim: `a7e181ffc4de46c39582f2720c020df6`
- Fold2/A1 checkpoint: `6840e0688780b28ce5a103776993224880ed8137ce9122c7ba676b954ed69c67`
- Exactly 2,873 unique retained predictions were present. Recomputed confusion matrix and
  accuracy / balanced accuracy / macro-F1 / precision / ROC-AUC match the retained result.
- `events.csv` contains exact Fold2/A1 `OUTER_CLAIMED` event 148 and `OUTER_EVALUATED`
  event 149 with the claim/result hashes above.
- Fold2/A0 completion artifact SHA-256 remains:
  `34fecfc92758919c8b1a22a5a59e9f0680b1da1311f609cd1f55af00df21609a`.
- Fold2 selection bundle SHA-256 remains:
  `ff1c4e5a100e503dbc8e95f1772cd92dcf613a50d4f34cb90fd7b5db091cb8d5`.

## Fold2/B1 frozen target

- Experiment: `fold2_B1_stronger_decay_seed42_677ad640faf5`
- Candidate: `stronger_decay`
- Selected epoch: `1`
- Frozen final checkpoint SHA-256:
  `0325643987a235655f39da418622a09049f1ee2fc94f0c09421daac0666132d4`
- `latest.json` names exactly that checkpoint and the supplied checkpoint bytes hash to the
  same SHA-256.
- The checkpoint metadata itself identifies Fold2/B1, candidate `stronger_decay`, epoch 1,
  frozen protocol/code identity, threshold 0.5 and torch `2.14.0+cpu` / two CPU threads.
- Fold2 outer set remains 2,873 signed ordered records with hash:
  `e85a74cb8dd33249ac0f27d42521d5dc0cd2d9abd283970c978f278d9fb97353`.

## New helper

`scripts/module5_outer_fold2_b1_continuation.py`

The helper authorizes only Fold2/B1. It pins the previously used A1 helper, Fold1 closure,
A0 completion, exact A1 claim/result/checkpoint and exact B1 checkpoint/selection. It
accepts only the already-completed A1 two-event delta plus, in verify mode, one legitimate
B1 claim/completion delta. It performs no model forwarding or native writes.

## Required local validation before inference

After copying this patch into the project root, run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'

.\.venv\Scripts\python.exe -m pytest tests/m5_experiments/test_outer_fold2_b1_continuation.py -q
.\.venv\Scripts\python.exe -m ruff check scripts/module5_outer_fold2_b1_continuation.py tests/m5_experiments/test_outer_fold2_b1_continuation.py
.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_b1_continuation.py --help
.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_b1_continuation.py preflight --outer-fold 2 --ablation B1
```

Continue only if every command passes and the real preflight reports B1
`FINAL_TRAIN_COMPLETE`, claim `null`, exact checkpoint above, zero model forwards and zero
production writes.

## Manual B1 evaluation after PASS

```powershell
.\.venv\Scripts\vaaniswitch-m5.exe outer --outer-fold 2 --ablation B1
```

On successful completion only, run:

```powershell
.\.venv\Scripts\python.exe -B scripts/module5_outer_fold2_b1_continuation.py verify --outer-fold 2 --ablation B1
```

Then stop before B2. If the native evaluation consumes a claim and fails/interruption occurs,
do not clear or retry the claim. Preserve evidence and return for review.

No cross-fold aggregation, robustness, Module 5 sign-off, or Module 6 work is authorized here.
