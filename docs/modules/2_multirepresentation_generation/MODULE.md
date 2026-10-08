# Module 2 — Mel, Visibility-Graph & Recurrence Representation Generation

**Status:** COMPLETE / PASSED / SIGNED OFF
**Code:** `src/vaaniswitch/m2_representations/`  
**Tests:** `tests/m2_representations/`

## Locked source contract

Every branch reads only `reconstruction.npy` from the signed Module 1 production
namespace:

`data/processed/vmd/bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685/`

The loader cross-checks the actual CORE selection against the signed Module 0 summary,
Module 1 full summary, and Module 1 sign-off audit. It verifies the logical ID, manifest
fingerprint, primary-audio binding, 16 kHz mono working semantics, `all_modes` policy,
array shape/dtype/finiteness, byte count, and reconstruction SHA-256. It rejects
tau-sweep namespaces and selected-mode reconstructions. Solver convergence is descriptive
only; valid iteration-ceiling outputs remain eligible.

## Log-Mel representation

- project-owned NumPy STFT/Mel wrapper; no raw WAV is read;
- 16 kHz, 512-point FFT, periodic Hann window of 400 samples (25 ms), 160-sample
  hop (10 ms), non-centered frames;
- 80 native HTK-spaced triangular Mel bands from 20 Hz through 8 kHz with Slaney area
  normalization;
- power spectrum (`|FFT|^2 / sum(window^2)`), then `10*log10(max(power, 1e-10))`;
- fixed physical 8.0-second waveform canvas: right-zero-pad shorter signals and take a
  deterministic centered 8.0-second crop of longer signals;
- 798 fixed-canvas STFT frames are area-averaged to 224 time pixels; 80 Mel rows are
  linearly interpolated to 224 frequency pixels;
- output: finite `float32`, shape `224x224`.

This does not stretch each utterance to fill the width. Shorter speech occupies its true
fraction of the shared 8-second axis and the remainder is the -100 dB numerical floor.
The CORE audit found 8,438/8,673 (97.2904%) records pad and 235/8,673 (2.7096%) crop.
The real production crop gate exercised six of those 235 records, including the
193.828625-second maximum, and confirmed the crop is exactly the centered 128,000
samples (798 STFT frames). The 235-record limitation is retained for robustness analysis
in Module 5; it is not an exclusion rule.

## Binary recurrence representation

- partition the complete reconstruction into 228 contiguous equal-duration bins and
  compute block RMS (an anti-aliasing aggregation);
- per-utterance z-score using epsilon `1e-12`;
- embedding dimension 3 and delay 2, yielding exactly 224 state vectors;
- pairwise Euclidean distance;
- choose the off-diagonal distance quantile at the configured 15% recurrence rate
  (NumPy `method="higher"`), threshold by `distance <= threshold`, and force the
  diagonal to one;
- output: symmetric binary `uint8`, shape `224x224`.

No class, speaker, fold, or model statistic enters the threshold.

## Natural visibility graph

- partition the complete reconstruction into at most 256 contiguous equal-duration bins
  and compute block RMS, then per-utterance z-score;
- node features are only normalized time `[0,1]` and standardized amplitude (`float32`);
- strict natural visibility: nodes `i<j` connect iff every intermediate `k` lies strictly
  below the line segment from `(i,y_i)` to `(j,y_j)`;
- the O(N²) implementation uses increasing maximum slopes and is tested on hand-checkable
  peak, valley, and collinear sequences;
- store both directions, no self-loops, as compact `int16 edge_index[2,E]`;
- Module 3 contract: add self-loops and apply symmetric degree normalization there.

The graph is not k-NN, correlation-based, horizontal visibility, learned, or dependent on
PyTorch Geometric.

## Cache contract

Outputs live under:

`data/processed/representations/<representation-config-hash>/<speaker>/<session>/<prompt>/`

Each complete directory contains `mel.npy`, `recurrence.npy`, `graph_nodes.npy`,
`graph_edges.npy`, and `metadata.json`. Arrays are atomically replaced first and
`metadata.json` is published last. Metadata binds logical ID, Module 0 fingerprint,
Module 1 config/source path/reconstruction SHA-256, Module 2 config hash, numerical
parameters, shapes, dtypes, artifact hashes, scientific statistics, and timings.
Incomplete, stale, corrupt, or hash-mismatched entries are rejected; valid entries are
skipped on resume.

Production representation config hash:

`d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d`

## Acceptance

- [x] Implementation.
- [x] Smoke validation.
- [x] Real crop validation.
- [x] Full 8,673-record CORE build.
- [x] Full 8,673-record CORE validation.
- [x] Full numerical, provenance, inventory, storage, runtime, and immutability audit.
- [x] Regression gates.
- [x] Signed off.

The production namespace contains 8,673 complete logical caches and 43,365 expected
files, with zero failures or incomplete entries. The measured size is 2,378,359,184 bytes
(2.21502 GiB). All 235 long records pass the Mel-crop-only/full-RP/full-NVG contract.
Module 3 remains not started and requires explicit authorization.
