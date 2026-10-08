# Tests — Module 4 Token-Level Cross-Attention Fusion & Dysarthria Classification

Run `uv run --python 3.12 --extra dev pytest tests/m4_fusion`. Fixtures use synthetic
Module 0 speaker records, the unchanged Module 3 dataset/collation and encoder classes,
and an in-memory representation source that rejects every reserved-speaker load.
Tests require no TORGO arrays or network download. Live production smoke and protected
input audits are recorded separately in Module 4 artifacts. No test is a scientific
classification performance result.
