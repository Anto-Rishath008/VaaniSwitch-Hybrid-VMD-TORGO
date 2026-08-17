# Module 2 Test Plan

## Source boundary

- production Module 1 `reconstruction.npy` is loaded, never raw WAV;
- tau-sweep paths, wrong VMD hash, selected-mode policy, stale logical ID, non-finite
  arrays, bad byte count/hash, and non-16-kHz semantics are rejected;
- signed CORE counts agree across the actual manifest and Module 0/1 evidence;
- excluded and non-normal records cannot enter CORE;
- valid Module 1 iteration-ceiling records remain processable.

## Numerical representations

- Mel determinism, finite float32 `224x224`, 16 kHz semantics, 25 ms/10 ms framing,
  fixed 8-second padding/cropping, explicit >8-second 128,000-sample/798-frame center
  crop, production 80-to-224 linear-resize constant preservation, generic resize weight
  conservation, and non-empty native Mel filters;
- recurrence determinism, exact binary uint8 `224x224`, symmetry, diagonal, density,
  embedding size, and short/constant-signal behavior;
- natural-visibility peak/valley/collinear hand checks, <=256 nodes, deterministic node
  features/edges, valid unique bidirectional indices, no self-loops, connectivity, and
  short-signal behavior;
- identical waveform transforms remain independent of label and speaker.

The pre-full-build real crop panel must cover both conditions and genders, multiple
speakers, moderate 8-12-second records, a >12-second record, a >30-second record, and
the longest CORE record. Only Mel receives the centered crop; recurrence and visibility
must demonstrably consume the complete reconstruction.

## Cache and integration

- atomic write/read round trip, artifact SHA-256, partial/corrupt rejection, config-hash
  namespace change, resume skip, strict validation, CLI build/validate, diagnostic plots,
  raw immutability, and Module 1 cache immutability.

Every defect follows reproduce → failing regression test → fix → targeted test → Module 2
regression → full project regression → defect log.

## Gates

Run in order:

```text
uv run --python 3.12 --extra dev pytest tests/representations
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev pytest --cov=components.data --cov=components.vmd --cov=components.representations --cov-branch --cov-report=term-missing
uv run --python 3.12 --extra dev ruff check .
uv run --python 3.12 --extra dev mypy src/components/m0_data src/components/m1_vmd src/components/m2_representations tests/data tests/vmd tests/representations
uv run --python 3.12 python -m compileall -q src tests
```
