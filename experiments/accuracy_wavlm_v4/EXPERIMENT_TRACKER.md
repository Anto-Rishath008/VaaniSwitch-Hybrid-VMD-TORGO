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
| Automated suite | 53 PASS, zero failures/errors/skips |
| Final CUDA source-bound panel | PASS; 0.838/0.928 GiB allocated/reserved |
| Original/V1/V2/V3 preservation | PASS |
| Full development training | NOT EXECUTED; separate approval required |
| Real outer inference | NOT EXECUTED; separate explicit authorization required |
| Independent external validation | NOT AVAILABLE / NOT CLAIMED |
| V4 scientific accuracy / 90 percent target | UNKNOWN / NOT CLAIMED |

Closed implementation findings: V4-001 official processor normalization, V4-002 test tolerance, V4-003 exclusive outer-claim creation, V4-004 deterministic attention/resume. Historical HuBERT development comparison is report-only after V4 decisions freeze and verifies identical speaker CV partitions. Completed-decision resume finishes missing report publication without re-training or changing decisions.

STOP after this handoff. Await the user's review and later authorization; do not automatically run Develop or Outer.
