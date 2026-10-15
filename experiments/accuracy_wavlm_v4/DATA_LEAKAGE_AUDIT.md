# V4 data-isolation audit

The inherited frozen manifest and fold hashes are verified before records are constructed. `validation/data_audit.json` records 8,673 CORE identities, the original fold mapping and V3's grouped development partitions. Unit tests exercise all real partitions, role rejection, duplicate recording and audio-hash rejection, source identity binding, held-out speaker rejection and train-only scaler fitting.

Each original outer context reserves five speakers and admits only its other ten to development. Three class-stratified grouped CV buckets preserve both classes, assign each development speaker once to validation and check speaker/logical-ID/primary-path/audio-hash separation. No alternate microphone becomes another row. A scaler and weights fit only the current training partition. B has no population-fitted feature scaler; its learned LayerNorm/head and upper-block weights see only training-role examples.

Selection functions accept development partitions and explicitly identified development metrics. They do not import historical predictions or reporting code. Threshold 0.5, center crops, no augmentation, grid, training seed and tie policy are declared before selection. No named-speaker corrections, unfavorable-speaker exclusions, resplits or test-population normalization are implemented.

Outer data access for predictions sits inside an exclusively created, durable claim after explicit authorization and all-context decision/checkpoint sealing. Tests cover missing authorization, combined-mode rejection, claim races, interrupted claims and sealed reuse. A fixture report never writes to production results. All-record coverage/label/threshold binding is rechecked during future reporting.

Bounded real-audio checks use only outer-context-1 development speakers, with separate training and validation speakers. The exact ten record identities, source hashes and forbidden outer speakers are retained in the latest CUDA report. Inspect that report for runtime status; smoke metrics never enter selection.

The three original outer fold populations have historically exposed results. Speaker separation remains enforceable, but these speakers cannot become independently untouched again. Future comparisons are POST-HOC EXPLORATORY. No external validation dataset was supplied and no independent confirmation has been claimed.
