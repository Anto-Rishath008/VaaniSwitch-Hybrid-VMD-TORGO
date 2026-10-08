# Module 3 Code — Parallel Token Encoders: Mel-CNN, VG-GCN & RP-CNN

Strictly loads signed Module 2 caches and produces Mel `[B,16,128]`, visibility-graph
`[B,1,128]`, recurrence `[B,16,128]`, and nonlinear `[B,17,128]` tokens on demand.

Use `vaaniswitch-m3 validate --config configs/models/encoders.yaml --limit 8` for source
validation and `vaaniswitch-m3 smoke ...` for the pretrained production smoke. Module 3
contains no attention, classifier, training loop, or learned-token cache. Read
`docs/modules/3_parallel_token_encoders/MODULE.md` for the signed contract.
