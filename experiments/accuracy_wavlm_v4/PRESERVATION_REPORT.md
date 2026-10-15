# V4 preservation report

PASS. `validation/protected_before.json` was created by the initial two-file session and retained unchanged through resume. It is excluded from its own protected inventory because it is a V4 output. `validation/protected_after.json` and `validation/preservation.json` report no changed, missing or added protected files. The resumed verification and the final comparison both passed.

| Scope | Total inventoried files | Byte-hashed (SHA-256 + size + mtime) | Metadata only (path + size + mtime) |
|---|---:|---:|---:|
| Original project | 240,570 | 30,802 | 209,768 |
| Accuracy study outside V4 | 54,234 | 20,170 | 34,064 |

All files in V1 (17,540), V2 (835) and V3 (1,355) match SHA-256, size and mtime, including their caches/checkpoints/sealed results and existing bytecode. The existing speaker-analysis scope contains 23 files. Every individual checked path and its evidence type appears in the two inventories. A `sha256` field means byte hashing; its absence means metadata only.

Raw/processed directories and existing virtual environments are metadata-only. This does not prove full-corpus byte equality. The ten actual development primary WAV files used by the CUDA panels were separately byte-hashed before decoding and after validation and match the frozen manifest. Original project source, manifests, splits, reports, figures and checkpoint/evaluation evidence outside raw/processed/environment trees were byte-hashed.

Git internal files are excluded from content inventory; HEAD, branch and all refs are checked separately and unchanged. Original release HEAD remains `2dee81f7656e5b5cae800d9a3c8692e5e8107bf9`. The root checkout's pre-existing dirty/untracked files were retained and compared; this report does not falsely call that checkout clean. No commit/push, reset, clean or evidence deletion occurred.

Only the exact study path `experiments/accuracy_wavlm_v4/` is excluded as intentional work. No entire experiments tree is excluded. All writes, model downloads, caches, test artifacts, logs and documentation remain under V4. Existing environments were neither reinstalled nor edited. Modules 0-6, raw TORGO and prior scientific results remain protected. No subsequent module was started.
