# V3 tracker — implementation boundary

| Work item | State | Evidence |
|---|---|---|
| Scope lock and actual V2 study inspection | COMPLETE | EXPERIMENT_PROTOCOL.md; V2_LINEAGE.json |
| Before preservation inventory | COMPLETE | validation/protected_before.json |
| Original 8,673 feature checksums/provenance | PASS | validation/feature_reuse.json |
| B0/B1/B2, N0/N1/N2, waveform augmentation | IMPLEMENTED | src/weighting.py; normalization.py; augmentation.py; features.py |
| Faithful reference and grouped CV selection | IMPLEMENTED; synthetic flow tested | src/selection.py; tests/test_v3.py |
| Optional calibration diagnostic | IMPLEMENTED; synthetic test passed; disabled by default | src/calibration.py; configs/study.yaml |
| Checkpoint/RNG/interrupt recovery | PASS in bounded tests | validation/latest_cuda.json; tests_final2.log |
| CUDA extraction and finite LR optimization | PASS in bounded tests | validation/latest_cuda.json |
| Launcher and required future reports | IMPLEMENTED; parsing and synthetic report tests passed | run_all.ps1; src/reporting.py |
| Preservation after validation | PASS | validation/preservation.json; PRESERVATION_REPORT.md |
| Open implementation defects | NONE KNOWN | DEFECT_LOG.md |
| Full-size V3 development experiment | NOT EXECUTED — awaiting user approval | No production checkpoints/FROZEN_DECISIONS |
| Real V3 outer evaluation | NOT EXECUTED — separate explicit approval required | No production outer claims/results |
| V3 final accuracy | UNKNOWN | No genuine V3 outer metrics exist |
| Original Module 7 or signed pipeline changes | NOT STARTED | Outside scope |

The original and V2 tracker/report documents predate some completed runs and are
preserved exactly. V3 uses their actual sealed artifacts rather than rewriting history.
