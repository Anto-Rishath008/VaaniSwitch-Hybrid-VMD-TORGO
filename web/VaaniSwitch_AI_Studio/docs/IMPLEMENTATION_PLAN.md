# VaaniSwitch AI Studio implementation

Separate Windows application; Modules 0–6 and V1–V4 remain read-only. This is not Module 7.

- `frontend/`: React, TypeScript, Vite, Motion, Lucide. Overview, speech studio, clear speech, model lab, pipeline explorer, results, speakers, limitations, about and presentation views. Browser Web Audio and MediaRecorder provide actual sample visualization and recording.
- `backend/`: FastAPI API; typed inputs; validated audio; private expiring sessions; bounded serial job queue; cancellable isolated Python workers; read-only model registry and historical data adapter.
- `scripts/`: audit, setup, start, inference smoke and browser checks. No training or evaluation launchers.
- `tests/`: security, audio, job, schema, frontend and integration regressions.
- `docs/`: integration audit, preservation evidence, validation, defects, tracker and handoff.
- `runtime/`: ignored audio, temporary files and caches; never copied to research checkouts.

HuBERT V2 E1 and V3 reuse frozen preprocessing, fitted heads and normalization. WavLM previews load the exact final checkpoint named by preserved outer-result provenance, not a newly selected best epoch. Fine-tuned weights restore the upper-layer delta on the pinned local backbone. Every classifier call requires context 1, 2 or 3; no automatic context choice or probability ensemble. Each subprocess exits after inference, releasing its GPU allocation. Original B2/C2 are historical-only until arbitrary-upload VMD and normalization parity are established.

Whisper uses the verified local small.en checkpoint and Module 6 decoding. Zira uses the same SAPI5 voice, rate, volume and public-loop workaround, with new output paths inside this application. Recognized text remains separate from edited synthesis text. Full pipeline stops for text review before synthesis.

Historical figures are display transformations of sealed reports and CSVs, with no new model selection or outer evaluation. SHA-256 protects integration assets, scientific sources and reports; a complete size/mtime inventory and git status protects both checkouts against incidental writes.
