# V3 preservation verification

PASS: validation/protected_before.json and protected_after.json compare equal.
The machine-readable result is validation/preservation.json. Both worktrees remain
at 2dee81f7656e5b5cae800d9a3c8692e5e8107bf9, with their original branch identities.
The release working tree's pre-existing dirty/untracked status was captured and
preserved; it was never reset or represented as clean.

| Protected group | Count | Verification scope |
|---|---:|---|
| Original project inventory | 191,833 files | Paths, size, nanosecond mtime |
| Original project/evidence subset | 22,175 files | Additional complete SHA-256 |
| Original raw/processed corpus remainder | 169,658 files | Metadata inventory only; not a full byte-level corpus audit |
| Tracked experimental-worktree files | 417 | SHA-256 |
| Existing speaker-level analysis | 23 files | SHA-256 |
| V1 experiment | 17,540 files | SHA-256, size, mtime, including existing bytecode |
| V2 experiment | 835 files | SHA-256, size, mtime, including existing bytecode |

Separately verified: manifest/fold hashes; all 8,673 original feature entries and
per-layer hashes; original crop intervals; pinned model/config/preprocessor files;
V1/V2 frozen source identity; nine V2 outer claims/results/seals and selected
checkpoint hashes; all eight sealed V2 report/figure artifacts. The 20 actual
development audio files used in bounded validation were individually byte-hashed
before/after. The full raw TORGO corpus is NOT claimed byte-verified.

Excluded from original project traversal: Git internals, virtual environments and
tool/bytecode caches, consistent with the inherited preservation policy. V1/V2
directory snapshots themselves include every file, including existing bytecode.
The experimental untracked V3 directory is the intentional new output boundary.
No commit, push, destructive cleanup, package installation or protected write occurred.
