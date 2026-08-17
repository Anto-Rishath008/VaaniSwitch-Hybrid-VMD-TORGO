# Module 5 — Accuracy Optimization, Ablation & Robustness

**Target:** Days 11–12
**Code:** `src/vaaniswitch/m5_optimization/`
**Tests:** `tests/m5_optimization/`

## Goal
Optimize honest speaker-level performance and run ablations/robustness.

## Detailed tasks
- small hyperparameter search only.
- train-only mild augmentation.
- early stopping.
- freeze hyperparameters before final folds.
- report fold mean±std.
- train final demo model after evaluation.

## Outputs
- frozen config
- fold metrics
- ablation table
- robustness report

## Acceptance
- [ ] Required outputs exist.
- [ ] Tests pass.
- [ ] No scientific leakage introduced.
- [ ] Test report completed.

Do not start the next module automatically.
