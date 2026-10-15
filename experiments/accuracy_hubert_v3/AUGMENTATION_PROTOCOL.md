# Frozen mild channel augmentation

For each TRAINING speaker, select at most 16 originals by a deterministic hash of
[augment-select,42,logical_id]. One variant per original; no error-, label-,
microphone-, or named-speaker-specific selection. A variant is generated only
when that speaker is in the active training role. Validation/assessment/outer
loaders never request variants. Across the full study the canonical per-speaker
budget permits at most 15*16=240 unique production variants. The same original can
be training in one outer fold and held-out in another; cache availability never
authorizes consuming its variant in a held-out role. Partition access receipts
record every training use. Smoke uses a separate cache and limited samples.

Load and checksum the primary waveform read-only using V2's decoder and centered
eight-second crop, then perturb a private waveform copy. A seed from SHA256 of
[v3-mild-channel-1,42,logical_id,audio_sha256] determines one of two fixed profiles:

- band_noise: gain -3 to +3 dB; second-order causal Butterworth high-pass 60-120 Hz
  and low-pass 4,000-6,500 Hz; Gaussian noise at 25-35 dB SNR relative to filtered RMS.
- tilt_echo: gain -3 to +3 dB; mild low-frequency response tilt -3 to +3 dB using
  a first-order 1,000 Hz low-pass shelf mixture; one delayed reflection at 10-30 ms
  with amplitude 0.05-0.12; Gaussian noise at 25-35 dB SNR.

All draws are deterministic NumPy PCG64 with pinned dependency versions. Rescale
the final whole waveform if needed to peak 0.99; never hard-clip. Duration and
sample rate stay unchanged. Silent input remains silent and is flagged; no label
is synthesized. No pitch shifting, time stretching or speaker-specific correction.
Gain alone is mostly removed by HuBERT input normalization; no gain-only efficacy
is claimed. Simple delayed reflection is an approximation, not measured room audio.

Apply unchanged V2 HuBERT processor/model after perturbation. Every feature is
newly inferred from that waveform, never fabricated by modifying pooled features.
GPU FP16 encoder/FP32 pooling, eval mode, frozen parameters, layers 6/9/12,
population-SD pooling, output [3,1536] float32. Exact-length batches only, production
batch one for the 8 GiB GPU. Bounded OOM helper halves larger test batches and stops
at one. No unlimited retries or corrupt-entry repair.

Cache identity includes logical record, speaker/label, primary path/hash, complete
parameters/seed, encoder revision and all file hashes, V2 preprocessing config,
all V3 source/protocol hashes, and installed dependency versions. Publication is
atomic; complete entries validate overall/per-layer checksums, dtype, shape,
finiteness and identity. Unpublished payloads remain as interrupted evidence;
new publication uses a new UUID. Completed entries cannot be overwritten.
Each completion also records crop interval, waveform SHA256, actual RMS/noise RMS,
peak-rescale factor and peak. Only V3 paths can be written.

Functional spectral/SNR/determinism/immutability tests and bounded actual-audio
CUDA extraction validate mechanics, not a proven clinical label-preservation
guarantee or improvement. Candidate augmentation benefit remains unknown until
the authorized development experiment is run.
