# VaaniSwitch AI Studio

A separate local Windows speech research application. Original Modules 0â€“6 and the V1â€“V4 accuracy studies remain read-only. This is not a new research module.

## Start

The application environment and production frontend have already been prepared on this machine.

```powershell
Set-Location -LiteralPath 'C:\Users\antor\OneDrive\Desktop\7th sem\Speech processing\Project\VaaniSwitch_AI_Studio'
.\scripts\Start.ps1
```

Website: **http://127.0.0.1:8000**  
Backend: **http://127.0.0.1:8000/api/health**  
Typed API schema: **http://127.0.0.1:8000/openapi.json**

If PowerShell blocks local scripts, use `powershell -ExecutionPolicy Bypass -File .\scripts\Start.ps1` for this invocation. Do not change system execution policy.

For a fresh application environment, run `.\scripts\Setup.ps1`. Setup writes only into this studio. It reuses the existing research Python packages through read-only paths and installs the small web dependencies and Windows speech bindings locally; it does not upgrade research environments or download model weights. Node 22.17.1 and Python 3.12 were used here.

## Use the studio

1. Upload WAV, MP3, M4A, FLAC, OGG or WEBM (25 MB, 60 seconds), or choose Record audio / Live microphone. Microphone recording requires localhost or HTTPS.
2. Replay the original, view its actual waveform and STFT, and inspect the decoder metadata.
3. For classification, explicitly select fold context 1, 2 or 3 and acknowledge the research-preview limitation. Analyze one model or compare all four available classifier families on the same audio.
4. For conversion, recognize English speech, review/edit the actual transcript, generate Microsoft Zira speech, then play or download the WAV. Full pipeline pauses for transcript review.
5. Remove the audio session with the trash control. Sessions expire after one hour, on shutdown, and after interrupted startup recovery.

The live microphone mode records then processes; no continuous streaming inference is claimed. Classification probabilities are uncalibrated model outputs, not diagnoses or known accuracy for a new recording. No automatic fold selection or ensemble occurs. A chosen context changes the fitted model, not the input's clinical status.

## Models

| Component | Studio behavior |
| --- | --- |
| Original C2 hybrid / B2 recurrence CNN | Historical metrics and architectures; live-upload inference unavailable pending frozen VMD and normalization parity validation. Three final checkpoints each are discovered. |
| HuBERT V2 E1 | Real research preview for each of 3 contexts; original frozen FP16 feature extraction and saved logistic head/scaler. |
| HuBERT V3 selected | Real research preview; layer 9, preserved context-specific weighting/normalization and fitted head. |
| WavLM V4 selected | Real research preview; exact saved best upper-layer delta on pinned local backbone, context-specific last-2/last-4 mean/attention configurations. |
| WavLM V4 reference | Real research preview; frozen layer 12 mean + population std with fitted LR C=0.01 and train-only scaler. |
| Module 6 ASR | Real local Whisper small.en, original frozen decoding. |
| Module 6 TTS | Real local Microsoft Zira, pyttsx3 2.99 / SAPI5, 175 words/minute. |

No model is presented as a universal clinical classifier. The 80.18% frozen reference is separate from the 79.22% development-selected V4 endpoint. V4 evidence is post-hoc exploratory reused-fold evaluation. Pretrained HuBERT, WavLM and Whisper weights are externally pretrained; this project trained the classifier heads and selected upper-layer adaptation, not those entire backbones from scratch.

## Architecture and files

`frontend/`: React + TypeScript + Vite, Motion, Lucide, Web Audio / MediaRecorder.  
`backend/`: FastAPI, audio decoder, typed schemas, serial subprocess queue, model worker, registry and historical adapters.  
`scripts/`: Windows startup/setup, preservation audit, focused inference smoke.  
`tests/` and `frontend/e2e/`: API/audio regressions and browser workflow tests.  
`docs/`: model audit, test evidence, screenshots, preservation, defects and handoff.  
`runtime/`: ignored local temporary audio/caches. Never expose this folder as a public static route.

The same FastAPI process serves the built frontend and API. Model workers are isolated, inference-only and bounded to one at a time. Classifier/ASR workers have a 300-second timeout; TTS has 60 seconds. Cancellation terminates the owned worker. Every worker exits after execution to release GPU memory. CUDA is required for the validated frozen extraction contract; CPU inference has not been validated or advertised. A failed CUDA/model load is reported instead of silently changing precision or models.

Audio is decoded with soundfile or a local FFmpeg binary. Mono downmix uses the arithmetic mean, polyphase resampling preserves scale, and source amplitude is never peak-normalized. Long uploads are rejected with trimming guidance. Classifiers transparently center-crop to 8 seconds (HuBERT/frozen WavLM) or 4 seconds (selected WavLM); provenance includes crop indices. Silence is previewable but cannot be analyzed. Generated speech supports at most 2,000 text characters and 180 seconds of output.

The displayed STFT is a visualization, not a frozen model's Mel input. Pipeline explorer also reads checksum-verified preserved Mel, recurrence and graph arrays for one historical development example; no dataset audio is served. Historical CSV downloads contain preserved prediction metadata, not speech recordings.

## Validation

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
Push-Location frontend
npm test
npm run build
Pop-Location
```

Focused real inference (new synthetic speech only, no TORGO evaluation):

```powershell
.\.venv\Scripts\python.exe -B scripts\smoke.py
```

Browser tests require the server running. The config uses the installed Chromium browser. A newly synthesized WAV feeds Chromium's virtual microphone, so recording controls and actual MediaRecorder are tested without claiming a physical microphone test.

```powershell
Push-Location frontend
npm run test:e2e
Pop-Location
```

Run backend tests with the local server stopped: their application lifecycle cleans temporary sessions. Do not run smoke and browser inference simultaneously; both exercise the GPU. Test results and exact measured scope are in `docs/TEST_REPORT.md`, `inference_smoke.json`, `browser_results.json`, and `accessibility_browser.json`.

Preservation verification (hashes, full before/after inventory and unchanged git status; no scientific evaluation):

```powershell
.\.venv\Scripts\python.exe -B scripts\audit.py after
```

`preservation_before.json` is the original baseline. Never regenerate it to conceal drift. Original checkout already had modified/untracked research artifacts; preservation means no additional changes relative to that observed starting state, not a clean git checkout.

## Troubleshooting and limits

- **Port 8000 occupied:** stop the prior owned studio server before starting another instance. The application is intentionally bound to loopback, not public hosting.
- **Microphone denied:** allow microphone access for localhost in browser site settings. Device removal/interruption produces an explicit error. No speech is sent to third-party APIs.
- **CUDA/model failure:** confirm the original read-only runtime environments still exist and the RTX 4070 is available. Private diagnostics are the `.result.json` files inside the current runtime session; browser errors omit machine paths and transcripts. There are no replacement-weight downloads or automatic CPU/precision substitutions.
- **Voice unavailable:** install/enable the exact Microsoft Zira EN-US SAPI5 voice. The adapter refuses silent replacement voices. It does not clone or recover original voice identity.
- **Slow first response:** each worker loads and checksum-verifies its model; the job reports measured loading and inference timings separately. A one-clip engineering smoke is not a standardized latency benchmark.
- **Empty ASR text:** some inputs have no recognizable speech. Review the input; you may type corrected synthesis text. Text editing does not fabricate a recognition result.
- **Moved research folders:** registry references are deliberately explicit local provenance. Reconfigure and re-audit the read-only paths before use; do not edit or move the protected research assets.
- **Physical microphone / listening:** browser automation uses a virtual microphone. Human hardware recording quality and intelligibility listening remain user checks; no MOS or clinical claims are made.

See `docs/HANDOFF.md` for delivery status and `docs/MODEL_INTEGRATION_AUDIT.md` for exact checkpoint provenance and runtime contracts.
