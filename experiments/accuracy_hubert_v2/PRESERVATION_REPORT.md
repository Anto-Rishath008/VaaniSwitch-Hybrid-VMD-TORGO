# V2 repair preservation report

**PASS**, compared with live inventories captured before this repair.

| Protected scope | Verification |
|---|---|
| V1 study, all 17,540 files | Exact path set, SHA-256, size and mtime |
| Original project, 191,833 files | Exact path set and metadata |
| Original project/evidence subset, 22,175 files | Also SHA-256 |
| Original raw/processed corpus, 169,658 files | Path/size/mtime; no new full-corpus byte audit |
| Existing tracked accuracy-worktree files, 417 | SHA-256 |
| Existing speaker-analysis files, 23 | SHA-256 |
| Original/worktree HEAD and branch | Unchanged |
| Original pre-existing dirty/untracked status | Unchanged |

V1 frozen_run SHA-256:
`8413cfebd979ff7c9b5e72f305c26834c937623a37fc18a82123a3ad8420e4d2`.
V1 has no outer/result artifacts. Its feature caches, E1 and incomplete E2
checkpoints, stages, failure log and implementation evidence are retained.

All 8,673 V1 features were independently validated against their source/model/
preprocessing identities and payload/per-layer checksums. The bounded CUDA test
also SHA-256-checked primary audio and signed RP for all 4,760 selected training
records. Original scientific data were opened read-only.

Repair outputs stay under `experiments/accuracy_hubert_v2/`. The existing
.venv-accuracy was reused with no installation. No release-branch modification,
commit, push, clean or reset was performed.

Machine evidence:
`logs/protected_before.json`, `logs/protected_after.json`,
`logs/v1_before.json`, `logs/v1_after.json`, `logs/preservation_report.json`.
