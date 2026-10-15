"""Content identities, atomic publication and resumable stage records."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parents[1]
ORIGINAL = STUDY.parent / "VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project"
BASELINE = "2dee81f7656e5b5cae800d9a3c8692e5e8107bf9"
TITLE = "POST-HOC EXPLORATORY IMPROVEMENT STUDY"
MANIFEST_SHA = "91c3327ea1f6a6b2057ffa02198f86d682668d22f247beef27f1466e959a9061"
RP_HASH = "d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d"
VMD_HASH = "bfd7aec36d99a88cfacc72df85e124d3f4b99b76ed47d71c3ced9a153fa92685"
FOLDS_SHA = "3d0303d9cb8af96da7b0bd4e4ad26b67ea0350c24a5607a7571d067e3a08aa8c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        # Windows rename never replaces an existing destination. On POSIX use a hard link.
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()


def write_json(path, value, *, replace=False):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
                        + "\n").encode(), replace=replace)


def write_text(path, value):
    atomic_bytes(path, value.encode())


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).decode().strip()


def config():
    import yaml
    cfg = yaml.safe_load((ROOT / "TRAINING_CONFIG.yaml").read_text(encoding="utf-8"))
    require(cfg["seed"] == 42 and cfg["training"]["threshold"] == 0.5, "Frozen seed/threshold")
    require(cfg["training"]["max_epochs"] == 20 and cfg["training"]["patience"] == 4,
            "Frozen bounded training contract changed")
    require(cfg["features"]["layers"] == [6, 9, 12], "Predeclared layer set changed")
    require(cfg["features"]["model"] == "facebook/hubert-base-ls960", "Wrong model")
    features, training = cfg["features"], cfg["training"]
    require(features["sample_rate"] == 16000 and features["segment_seconds"] == 8
            and features["crop"] == "center" and features["feature_dimension_per_layer"] == 1536,
            "Unsupported feature preprocessing contract")
    require(training["optimizer"] == "AdamW" and training["projection_dimension"] == 128
            and training["class_weight"] is None and training["augmentation"] == "none",
            "Unsupported classifier/training contract")
    require(training["logistic_C"] == [.01,.1,1.] and len(training["neural_candidates"]) == 2,
            "Predeclared search size changed")
    require(1 <= features["decode_workers"] <= 2 and 1 <= features["extraction_batch_cap"] <= 4
            and 1 <= training["batch_cap"] <= 16 and training["cpu_threads"] == 2
            and 0 < cfg["execution"]["vram_fraction"] <= .8, "Resource bounds changed")
    return cfg


def source_identity():
    files = [ROOT / "TRAINING_CONFIG.yaml", ROOT / "run_all.ps1"]
    files += sorted((ROOT / "src").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def progress(stage, completed=None, total=None, started=None):
    from datetime import datetime, timedelta, timezone
    suffix = ""
    if total is not None:
        suffix = f" {completed}/{total}"
        if started and completed:
            suffix += f"; ETA {(time.perf_counter()-started)*(total-completed)/completed/60:.1f} min"
    now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%H:%M:%S IST")
    print(f"[{now}] {stage}{suffix}", flush=True)


class ProcessLock:
    """OS releases this lock on process death; the persistent file is harmless."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            if self.path.stat().st_size == 0:
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *unused):
        self.file.close()


class Stages:
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        self.root.mkdir(parents=True, exist_ok=True)

    def event(self, name, status, **details):
        write_json(self.root/"events"/(f"{time.time_ns()}-{uuid.uuid4().hex}.json"),
                   {"stage": name, "status": status, "identity": self.identity, **details})

    def execute(self, name, action):
        path = self.root / f"{name}.json"
        previous = read(path) if path.exists() else {}
        require(not previous or previous["identity"] == self.identity, "Stage identity changed")
        if previous.get("status") == "COMPLETE":
            for filename, checksum in previous["artifacts"].items():
                require(sha(filename) == checksum, f"Sealed artifact changed: {filename}")
            progress(f"REUSE {name}")
            return previous["value"]
        progress(f"START {name}")
        self.event(name, "RUNNING", previous_status=previous.get("status"))
        write_json(path, {"identity": self.identity, "status": "RUNNING"}, replace=True)
        started = time.perf_counter()
        try:
            value, artifacts = action()
            write_json(path, {"identity": self.identity, "status": "COMPLETE", "value": value,
                              "seconds": time.perf_counter()-started,
                              "artifacts": {str(p): sha(p) for p in artifacts}}, replace=True)
            self.event(name, "COMPLETE", stage_record_sha256=sha(path))
            progress(f"COMPLETE {name}")
            return value
        except BaseException as exc:
            self.event(name, "FAILED", error=f"{type(exc).__name__}: {exc}")
            write_json(path, {"identity": self.identity, "status": "FAILED",
                              "error": f"{type(exc).__name__}: {exc}"}, replace=True)
            raise


EXCLUDED = {".git", ".venv", ".venv-m6", ".venv-accuracy", "__pycache__",
            ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def preservation_snapshot():
    """SHA every project/evidence file; metadata inventory for raw/processed corpora."""
    result = {}
    for directory, dirs, files in os.walk(ORIGINAL, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(ORIGINAL).as_posix()
            stat = path.stat()
            entry = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if not relative.startswith(("data/raw/", "data/processed/")):
                entry["sha256"] = sha(path)
            result[relative] = entry
    tracked = {p: sha(STUDY / p) for p in git(STUDY, "ls-files").splitlines()}
    analysis = {p.relative_to(STUDY).as_posix(): sha(p)
                for p in sorted((STUDY / "experiments/speaker_level_analysis").rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts}
    return {"original": result, "experimental_tracked": tracked, "speaker_analysis": analysis,
            "original_git": {"head": git(ORIGINAL, "rev-parse", "HEAD"),
                             "branch": git(ORIGINAL, "branch", "--show-current"),
                             "status": git(ORIGINAL, "status", "--porcelain=v1")},
            "study_git": {"head": git(STUDY, "rev-parse", "HEAD"),
                          "branch": git(STUDY, "branch", "--show-current")}}


def verify_preservation(before):
    progress("Checking original project and existing analysis preservation")
    after = preservation_snapshot()
    require(before == after, "Protected project changed; compare preservation snapshots")
    return {"status": "PASS", "original_files": len(after["original"]),
            "hashed_project_files": sum("sha256" in x for x in after["original"].values()),
            "tracked_experimental_files": len(after["experimental_tracked"]),
            "existing_analysis_files": len(after["speaker_analysis"]),
            "corpus_scope": "path/size/mtime inventory; selected inputs separately SHA-256 checked"}
