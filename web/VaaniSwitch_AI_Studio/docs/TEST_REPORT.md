# Application test report — 15 October 2026

The separate web application was tested without training or repeating TORGO evaluation.

| Check | Result | Scope |
| --- | --- | --- |
| Backend / runtime regressions | **42 passed** | Actual audio decoding for WAV/MP3/M4A/FLAC/OGG/WEBM, resampling, amplitude, invalid input, RIFF bounds, silence, duration, upload/play/download/delete, expiry, host/origin controls, public-path privacy, registry/labels, best-delta provenance, version drift, jobs/cancellation, research metrics and preserved representation checks. |
| Frontend components | **10 passed** | Navigation, verified metrics, availability, explicit research context, transcript editing, microphone errors, architecture interaction, unavailable backend, late microphone permission release. |
| Production build | **Passed** | TypeScript and Vite; local bundled assets; main JS ~422kB (~134kB gzip), CSS ~52kB (~12kB gzip). |
| Real inference smoke | **14 passed** | 12 classifier calls: four families × three contexts; one Whisper transcription; one actual Microsoft Zira synthesis. Only newly synthesized English speech was used. |
| Browser workflows | **6 passed** | All 10 desktop/mobile routes, no horizontal overflow, real upload/compare/ASR/edit/TTS/play/download/delete, recording pause/resume/stop/analyze, permission denial, invalid upload, accessibility and browser navigation timings. |
| Accessibility scans | **Passed** | WCAG2 A/AA + WCAG2.1 AA tags on 9 routes; no serious or critical findings after observed contrast and semantic fixes. No human screen-reader audit. |
| Dependency audit | **0 npm findings** | Locked frontend dependencies after fixing observed vulnerable versions. |
| Windows startup | **Passed** | Exact scripts/Start.ps1 started production frontend and API on 127.0.0.1:8000. |
| Research preservation | **Passed** | All 313,416 non-Git files matched initial size/mtime inventories; 679 baseline + 3 supplemental distinct assets matched SHA-256; original git statuses unchanged; all 13 V4 REPORT_SEAL entries matched. |

One non-failing warning remains: the current Starlette TestClient warns that its httpx transport is deprecated in favor of httpx2. It affects the test helper, not the served application or the passing tests. No blanket research-environment upgrade was performed.

The real smoke used the sentence: “Every voice deserves to be understood. This is a local speech research demonstration.” Whisper returned that complete sentence. Zira produced finite, non-silent WAV audio and the browser played and downloaded generated speech. No human listening or MOS claim was added.

On the RTX 4070 Laptop GPU, classifier inference in this one-clip smoke took approximately 288–766 ms after loading; classifier model loading/checksum/preparation took 3.42–4.71 s. Whisper inference took 1.35 s after a 1.85 s model load. Zira synthesis took 0.50 s. These do not include all worker/import overhead and are not standardized benchmarks. Maximum observed peak allocation was 1.370 GiB and peak reservation 1.533 GiB (Whisper). CPU utilization was not benchmarked. Per-case measurements are in inference_smoke.json and MODEL_INTEGRATION_AUDIT.md.

Browser microphone tests exercised an actual MediaRecorder and backend decode/inference, with Chromium's virtual microphone fed by a newly synthesized WAV. Physical microphone hardware, device quality and human intelligibility listening are not verified by that automation.

Screenshots in `screenshots/` include desktop and mobile overview/studio, pipeline/results, actual multi-model and ASR/TTS outputs, and recorded-audio analysis. Full-page screenshots are engineering review evidence, not synthetic UI mockups.

Original research checkouts already contained modified/untracked artifacts at the start. Preservation means no additional changes relative to the captured baseline. It does not claim the original checkout was git-clean or that every raw/cache file was rehashed; full inventory checks cover those files, while SHA-256 covers scientific code/configuration/reports and integrated weights/checkpoints. Details: preservation_before.json, integration_hashes_before.json, preservation_after.json.
