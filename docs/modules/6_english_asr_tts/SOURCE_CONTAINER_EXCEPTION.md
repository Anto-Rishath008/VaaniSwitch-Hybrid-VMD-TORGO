# Authorized TORGO RIFF +8 source exception

**MODULE 6: COMPLETE / PASSED / SIGNED OFF. FINAL EVALUATION: COMPLETE / PASS.
HUMAN LISTENING: PASS — USER CONFIRMED.**

Policy: **TORGO_RIFF_CHUNKSIZE_PLUS8_EXCEPTION**. It applies only through the explicit frozen TORGO reader
to the five approved IDs with their original Phase-0 hashes and signed manifest frame
counts. Generic overdeclared WAVs and generated TTS WAVs remain rejected.

Each accepted source satisfies: canonical raw-tree path and frozen complete-file SHA;
valid RIFF/WAVE signatures; outer overdeclaration exactly +8; complete 16-byte PCM fmt;
PCM tag 1, mono, 16 kHz, 16-bit, byte rate 32,000, block align 2; complete chunk walk;
data starting at byte 44 and ending at physical EOF; integral complete frames; exact
manifest frame count; Python wave reads the same complete payload. No source header
or sample byte is repaired, even in a temporary buffer. Original source hashes remain
over the complete unchanged files. Decoded float32 hashes are also bound to each result.

These containers are **not fully standards-conformant**. They are **accepted under a
narrow source-container exception after verifying complete PCM payload and frozen
source identity**. The PCM payload is complete while the outer ChunkSize is malformed.

| Logical ID | Physical bytes / ChunkSize | Declared total | PCM bytes | Frames | Source SHA-256 |
|---|---:|---:|---:|---:|---|
| `M01/Session1/0007` | 83884 | 83892 | 83840 | 41920 | `932952115252e32944534f68d9618febb7d260ee1e069df6d9f0999549f8b8cb` |
| `M01/Session1/0008` | 129964 | 129972 | 129920 | 64960 | `f26d9007142c0a3c2e6c1bacddf06d8e3a5f29a9705bf1b689d35ef70ee99458` |
| `M01/Session1/0014` | 88364 | 88372 | 88320 | 44160 | `7ba62962dfd075607496f93d4020e967f8f07ea3dc12b499c90e9216b829c6fc` |
| `M01/Session1/0017` | 94444 | 94452 | 94400 | 47200 | `dd48d341cfb8ee19bee0ca22ea08c59cc0f57531c2f9f37e5875da8961651d33` |
| `M01/Session1/0020` | 112364 | 112372 | 112320 | 56160 | `820b899f0e8e811bc38c3b114cb14e5ddbc331c7ca5f0af6d8547ad142333450` |


All five have overdeclaration 8, data offset 44, decoded frames equal to manifest frames,
and source_bytes_modified=false. Final preflight and post-run source audit independently
confirmed **125 strict sources + exactly these 5 exceptions; zero other exceptions**.
Detailed proof: [source exceptions](../../../artifacts/module6/exception_evaluation_20261008/preflight_20261008T173011117192Z/source_exceptions.json)
and [post-run verification](../../../artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f/POST_RUN_VERIFICATION.json).

