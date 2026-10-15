# VaaniSwitch AI Studio handoff

Delivered directory:

`C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_AI_Studio`

Start from that directory:

```powershell
.\scripts\Start.ps1
```

Website: http://127.0.0.1:8000  
Backend health: http://127.0.0.1:8000/api/health  
Typed OpenAPI: http://127.0.0.1:8000/openapi.json

The prepared production frontend and new Python web environment are included locally. Setup and troubleshooting instructions are in README.md. The QA server is stopped at handoff; run the command above to begin a fresh audio session.

Real integrated research previews: HuBERT V2 E1, selected HuBERT V3, selected WavLM V4 and frozen WavLM V4 reference, each with explicit context 1/2/3. These use preserved fitted heads/scalers and exact original inference preprocessing, including selected V4's best saved upper-layer delta. No fold averaging, automatic context selection or new refit occurs. Checkpoint paths, hashes, configurations and contracts are in MODEL_INTEGRATION_AUDIT.md and backend/model_registry.json.

Original B2 and C2 are historical-only: three final checkpoints each exist, but arbitrary-upload shared-VMD / representation / train-normalization parity has not been validated. Their metrics and architectures remain visible, with live controls disabled and the blocker explained. There is no universal production classifier or validated CPU fallback.

Actual Module 6 Whisper small.en recognizes supplied English audio; the transcript can be edited before actual Microsoft Zira synthesis. Generated audio has real waveform, playback and download. This is ASR → human text review → standard TTS, not original-voice restoration.

Validation: 42 backend/runtime tests, 10 frontend component tests, 6 browser workflow scenarios and 14 real inference smoke cases passed. Production build and Windows startup passed. Automated accessibility scans found no serious/critical findings on 9 routes; frontend npm audit has zero findings. Complete evidence is in TEST_REPORT.md, inference_smoke.json, browser_results.json, accessibility_browser.json and screenshots/.

GPU observations: one model worker at a time; peak measured allocation 1.370 GiB and reservation 1.533 GiB. Classifier inference 288–766 ms after loading for the single synthetic smoke clip; loading/checksum/preparation 3.42–4.71 s. These are engineering observations, not guaranteed latency. Physical microphone quality and human listening remain unverified; browser recording was tested with a real MediaRecorder and virtual microphone source.

Preservation passed: both original git statuses stayed identical; 313,416 non-Git file size/mtime entries stayed unchanged; 682 unique protected integration/evidence SHA-256 checks passed; every V4 sealed report/figure/CSV hash matched. No training, new outer-fold evaluation, checkpoint rewrite, dataset reorganization, cache deletion or protected-source edit occurred. Original pre-existing dirty/untracked files were retained. Full preservation scope is documented in TEST_REPORT.md.

The application runs locally and does not send speech to third-party APIs. Sessions expire after one hour, can be explicitly removed, and are removed on graceful shutdown or next startup after interruption. The web server is intentionally bound to loopback.
