# VaaniSwitch V3

Isolated post-hoc exploratory HuBERT channel-robustness study. Implementation and
bounded validation only; full development training and real outer inference have
not been executed. No V3 accuracy is known. See IMPLEMENTATION_REPORT.md for the
handoff and TEST_REPORT.md for measured evidence.

The signed VaaniSwitch Modules 0-6, original release, raw TORGO, manifests, splits,
speaker analysis and V1/V2 directories are protected. Every new artifact is in
this V3 directory. The original shared-VMD project architecture is unchanged.

After user approval, one command runs the complete DEVELOPMENT experiment using
the existing environment, verifies preservation and cache lineage, generates only
missing bounded augmentations, resumes fits and freezes all choices. It stops
without real outer inference:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v3\run_all.ps1"
```

Modes: `-Mode Prepare` audits without classifier fitting; `-Mode Validate` runs
only the bounded real-GPU panel. The latter was executed during implementation;
the default full development mode was not executed.

Only after separately authorizing real exploratory outer evaluation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_Accuracy_Study\experiments\accuracy_hubert_v3\run_all.ps1" -Mode Outer -AuthorizeOuter
```

Outer mode requires an existing complete, sealed FROZEN_DECISIONS file. An explicit
`-AuthorizeOuter` with default Develop also authorizes evaluation after development
finishes. Never use either authorization form without the user's approval.
There is no automatic ambiguous-claim retry. Inspect the error and retained
evidence instead of deleting claims/checkpoints. No commit/push is included.

Files: configs/study.yaml; source modules under src; tests/test_v3.py; validation
audits and bounded CUDA checkpoints; V2_LINEAGE.json; protocols; reports/tracker/
changelog/defects. Future production output goes to cache/augmented, checkpoints,
experiments/calibration, results, figures and logs. Checkpoint/source/cache IDs
are independent of V2; reused V2 code and V1 embeddings remain read-only dependencies.

Unit/regression tests (from the experimental worktree):

```powershell
.\.venv-accuracy\Scripts\python.exe -B -m unittest discover -s experiments/accuracy_hubert_v3/tests -v
```

All synthetic fixtures use V3 validation temporary directories and mark their
prediction/report evidence as SYNTHETIC_TEST_FIXTURE. Their end-to-end test does
not constitute execution of the real V3 study. No packages were reinstalled.
