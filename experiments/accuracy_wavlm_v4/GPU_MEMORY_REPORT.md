# V4 GPU bounded-validation report

PASS on NVIDIA GeForce RTX 4070 Laptop GPU; total 7.996 GiB. Torch 2.13.0+cu126, CUDA 12.6, Transformers 4.57.6. FP32 only; no autocast, FP16, BF16 training or GradScaler was used. No dependencies were installed.

Final panel wall time: 31.214 seconds. Maximum allocated: 899,546,112 bytes (0.838 GiB). Maximum reserved: 996,147,200 bytes (0.928 GiB). These are this process's PyTorch allocator measurements, not all GPU processes or total device occupancy. Per-process allowance 80 percent; observed panel fit the 8 GB GPU.

| Variant | Trainable parameters | Frozen parameters | Optimizer updates | Training seconds | Records/sec | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| B_last2_attention | 14,203,753 | 80,205,128 | 2 | 0.679 | 11.78 | 0.713 |
| B_last2_mean | 14,179,113 | 80,205,128 | 2 | 0.706 | 11.32 | 0.717 |
| B_last4_mean | 28,355,921 | 66,028,320 | 2 | 0.687 | 11.65 | 0.928 |

Frozen encoder: 94,381,936 parameters, zero trainable. Frozen features use eight-second maxima; neural training and inference use four-second maxima. Physical batch 1 throughout. Smoke accumulation 4, production accumulation 8; accumulation retains gradients, not previous forward graphs. Non-reentrant gradient checkpointing is enabled. Training timing includes decode, forward/backward/update and synchronization, but excludes loading/checkpoint I/O/validation; complete panel wall time is reported separately.

Last2 updates zero-based layers 10/11; last4 updates 8/9/10/11. Frontend, projection, positional convolution, encoder normalization and lower blocks stay frozen/eval. Optimizer groups use LR 1e-5 and 1e-4, AdamW weight decay .01, clipping 1, linear scheduling. Every intended parameter receives a finite gradient; frozen parameters do not. Finite parameters and optimizer state are checked after each update.

OOM policy has zero automatic retries at physical batch one. It retains the last complete update and exact failure context, then stops. Unit fault injection passed; no real CUDA OOM occurred. No failed batch is skipped, no sequence is silently shortened, and no alternative numerical policy is auto-selected.

Numerical backend: strict deterministic algorithms, math SDPA only; flash, memory-efficient and cuDNN SDPA disabled. This fixes actual resume drift observed in an earlier retained panel. Final interrupted/uninterrupted states matched exactly. The installed model still emits an attention-mask dtype deprecation warning. Full-run and cross-hardware reproducibility remain unmeasured.

Planning runtime: roughly 15-40 hours for the full predefined development budget, depending on early stopping, actual duration distribution, I/O and contention. At eight epochs the 27 neural CV fits cover up to 832,608 training-record presentations; final refits can add 138,768. The measured tiny-panel throughputs are tabulated above; extrapolation is uncertain. Full throughput, extraction duration and end-to-end runtime are unmeasured.

Planning disk: approximately 75-85 GiB additional for the maximum retained development state, best checkpoints, final fits, features and reports. The launcher requires 85 GiB free. Current V4 logical size is about 11.54 GiB, including repeated bounded evidence; available disk was 89.76 GiB when this report was generated. Original checkpoints are referenced, never copied or deleted; pretrained WavLM itself is approximately 360 MiB. Interrupted/repeated runs can require more disk. No full study was launched.
