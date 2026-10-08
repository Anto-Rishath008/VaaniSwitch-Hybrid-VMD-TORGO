# Module 5 experiment harness

**IMPLEMENTATION READY / AWAITING MANUAL EXPERIMENTS.** Not complete or signed off.
Module 6 is not started. See `docs/modules/5_accuracy_robustness/MODULE.md` and its
`TEST_REPORT.md` for the protocol, gates and manual handoff.

The sole config is `configs/experiments/module5/protocol.yaml`. The CLI is
`vaaniswitch-m5`, also `python -m vaaniswitch.m5_experiments`. Global `--config`
goes before the subcommand. `dry-run` reads metadata only; `protocol --freeze`
locks protocol/code. A0 build/validate requires explicit `--smoke` or `--full`.
Search, ablation-inner, retrain and outer are manual phases. Aggregate and robustness
read immutable outer evidence; ledger inspects state. Importing/installing never
launches experiments. Signed Modules 0–4 remain unchanged.
