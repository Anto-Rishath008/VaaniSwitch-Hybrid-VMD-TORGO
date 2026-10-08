# Tests — Module 3 Parallel Token Encoders: Mel-CNN, VG-GCN & RP-CNN

The 48-test suite covers strict signed-source rejection, provenance/folds, collation,
fixed preprocessing, graph normalization/masking, all token contracts, determinism,
gradient flow, frozen backbone behavior, state round-trip, parameter/timing diagnostics,
and both CLIs. Unit models use `weights=None`; only the real production smoke loads
ImageNet-1K V1. No TORGO audio is committed.
