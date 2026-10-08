"""One specifically authorized external A1 recovery. Never mutates native evidence.

preflight/verify construct no model. Only the explicit run subcommand can infer.
An intent without a matching durable commit is terminal pending human review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from vaaniswitch.m3_encoders.collate import collate_encoder_samples  # noqa: E402
from vaaniswitch.m4_fusion.losses import batch_labels, label_for_condition  # noqa: E402
from vaaniswitch.m4_fusion.metrics import binary_metrics  # noqa: E402
from vaaniswitch.m4_fusion.reproducibility import seed_everything  # noqa: E402
from vaaniswitch.m5_experiments.checkpoint import validate_progress  # noqa: E402
from vaaniswitch.m5_experiments.config import (  # noqa: E402
    ExperimentConfig,
    canonical_path,
    digest,
    file_hash,
)
from vaaniswitch.m5_experiments.data import ExperimentDataset, OuterPermit  # noqa: E402
from vaaniswitch.m5_experiments.source import prepare_experiments, require_frozen  # noqa: E402
from vaaniswitch.m5_experiments.trainer import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "677ad640faf584d1b3420ac32eebec7c496ba048f3f40782d4f6d886e04442a9"
CODE = "7696cad010914ec37f494d9f1a11a2e1761e46f353e61d97a07399f746a34ff8"
RUN = "fold1_A1_stronger_dropout_seed42_677ad640faf5"
CLAIM = "cb66ac287b054a8b8cd3e68d3d8ee385"
CHECKPOINT = "5c2cc1de2f8e46d9729829f1eeb303e03d4733e86dab862ce6e85ad1c1eb34a8"
RECORD_HASH = "2936df532e7ada33d7b571841c1026ddb30219fda3029cd89b3325f0f32d0569"
BASE = f"artifacts/module5/{PROTOCOL}"
RECOVERY = f"{BASE}/outer_recovery/fold1_A1"
FORENSICS = f"{BASE}/forensics/fold1_A1_20261005T060808Z"
FORENSIC_HASHES = {
    "forensic_audit.json": "319c562c0509142f23d4d300d3157aa0b209f5a9e1de049cbfb816b3a5562c73",
    "FORENSIC_REPORT.md": "6d69eed61b0c58ef701f6bf35364d75f1eaf5ed28bf785bc160f92f020918ccf",
    "host_evidence.json": "4f90ece527d1b88c02120010be7355d4736db7446b7dcbe45686a5c11de3ca84",
    "SHA256SUMS.txt": "fc53beb505e353fc823cdf1d841f0b80769027346d13f98a52758dcc09f0ed02",
}
TARGET = {
    "experiment_id": RUN,
    "original_claim": CLAIM,
    "outer_fold": 1,
    "ablation": "A1",
    "candidate": "stronger_dropout",
    "checkpoint_sha256": CHECKPOINT,
    "selected_epoch": 1,
    "record_count": 2819,
    "record_hash": RECORD_HASH,
    "speakers": ["F03", "FC01", "M04", "MC03", "MC04"],
    "threshold": 0.5,
    "seed": 42,
    "torch": "2.14.0+cpu",
    "cpu_threads": 2,
    "batch_size": 16,
    "shuffle": False,
    "num_workers": 0,
    "gradients": False,
    "evaluation_mode": True,
    "protocol_hash": PROTOCOL,
    "frozen_code_hash": CODE,
    "status": "RECOVERED_AFTER_INTERRUPTION",
    "classification": "PROTOCOL_EXCEPTION",
    "original_native_state": "FINAL_TRAIN_COMPLETE",
    "original_prior_forward_count": "UNKNOWN",
}
NOTICE = "THIS IS NOT A NATIVE OUTER_EVALUATED RESULT"
REASON = (
    "Host restart after native one-use claim left no retained result and "
    "prior forward count cannot be established."
)


class ReviewRequired(ValueError):
    """Preserve all files and stop; never repair/replay automatically."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReviewRequired(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_hash(path)}


def inventory(directory: Path) -> dict[str, Any]:
    result = {}
    for path in sorted(directory.rglob("*")):
        require(path.resolve() == path, f"Redirected evidence path: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = fingerprint(path)
    return result


def read_production(path: Path) -> tuple[dict[str, Any], list[Any]]:
    """No native Ledger instance, transaction writer, lock or CSV exporter."""
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT id,identity,state,data,claim FROM runs").fetchall()
        events = db.execute("SELECT * FROM events ORDER BY seq").fetchall()
    return {
        i: {"identity": json.loads(j), "state": s, "data": json.loads(d), "claim": c}
        for i, j, s, d, c in rows
    }, events


def validate_target(run: dict[str, Any], events: list[Any], native_exists: bool) -> None:
    require(not native_exists, "Native A1 result exists; stop")
    require(run["claim"] == CLAIM, "Wrong original claim")
    require(run["state"] == "FINAL_TRAIN_COMPLETE", "Wrong original native state")
    identity = run["identity"]
    for key, expected in {
        "experiment_id": RUN,
        "outer_fold": 1,
        "ablation": "A1",
        "seed": 42,
        "threshold": 0.5,
        "torch": TARGET["torch"],
        "cpu_threads": 2,
        "config_hash": PROTOCOL,
        "code_hash": CODE,
    }.items():
        require(identity[key] == expected, f"Wrong target {key}")
    require(identity["candidate"]["id"] == "stronger_dropout", "Wrong candidate")
    own = [e for e in events if e[1] == RUN and ("OUTER" in e[3] or e[3] == "FAILED")]
    require(len(own) == 1, "Unexpected A1 outer/failure events")
    require(
        tuple(own[0][:4]) == (135, RUN, "2026-09-29T04:44:19.996521+00:00", "OUTER_CLAIMED")
        and json.loads(own[0][4]) == {"claim": CLAIM},
        "Original event differs",
    )


def validate_records(records: tuple[Any, ...]) -> None:
    ids = [r.logical_id for r in records]
    require(len(ids) == 2819 and len(set(ids)) == 2819, "Wrong record count/duplicates")
    require(digest(ids) == RECORD_HASH, "Wrong ordered record hash")
    require(sorted({r.speaker_id for r in records}) == TARGET["speakers"], "Wrong speakers")


@dataclass
class Prepared:
    root: Path
    context: Any
    run: dict[str, Any]
    records: tuple[Any, ...]
    binding: dict[str, Any]
    production: dict[str, Any]
    policy: dict[str, Any]

    @property
    def namespace(self) -> Path:
        return canonical_path(self.root, RECOVERY)

    def directory(self, attempt_id: str) -> Path:
        require(
            bool(re.fullmatch(r"a1_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{12}", attempt_id))
            and attempt_id == self.policy["attempt_id"],
            "Unauthorized recovery attempt ID",
        )
        return canonical_path(self.root, f"{RECOVERY}/{attempt_id}")


def checkpoint_payload(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    require(file_hash(path) == CHECKPOINT, "Changed checkpoint")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(saved, dict)
        and set(saved) == {"schema", "identity", "progress", "model", "optimizer", "rng"}
        and saved["schema"] == "module5_checkpoint_v1"
        and saved["identity"] == identity,
        "Checkpoint schema/identity differs",
    )
    validate_progress(saved["progress"])  # Checks stored metadata; computes no class weight.
    require(saved["progress"]["epoch"] == 1, "Wrong selected epoch")
    return saved


def prepare(root: Path = ROOT) -> Prepared:
    """Hash/metadata checks only. No datasets, models, optimizers or forwards."""
    root = root.resolve()
    namespace = canonical_path(root, RECOVERY)
    policy_path = namespace / "authorization_policy.json"
    policy = read_json(policy_path)
    allowed_names = {
        "authorization_policy.json",
        "authorization_request.txt",
        "execution.lock",
        "preparation",
        policy["attempt_id"],
    }
    require(
        {p.name for p in namespace.iterdir()} <= allowed_names,
        "Another recovery attempt or unexpected namespace file exists",
    )
    require(
        policy["target"] == TARGET and policy["authorized_before_recovered_metrics"] is True,
        "Authorization is not for the exact frozen target/settings",
    )
    require(
        file_hash(Path(__file__)) == policy["recovery_script_sha256"],
        "Recovery implementation changed after preparation",
    )
    for name, checksum in FORENSIC_HASHES.items():
        require(
            file_hash(canonical_path(root, f"{FORENSICS}/{name}")) == checksum,
            f"Missing/changed forensic evidence: {name}",
        )
    forensic = read_json(root / FORENSICS / "forensic_audit.json")
    require(
        forensic["status"] == "FORENSIC_AUDIT_COMPLETE_RECOVERY_BLOCKED"
        and forensic["integrity_checks"] == "PASS"
        and forensic["zero_a1_forwards_proven"] is False
        and forensic["a1"]["claim"] == CLAIM,
        "Wrong forensic conclusion",
    )
    production = forensic["production_evidence_before_and_after"]
    ex = canonical_path(root, f"{BASE}/experiments")
    require(inventory(ex) == production, "Production evidence changed since forensic audit")
    for relative, expected in forensic["frozen_source_fingerprints_before_and_after"].items():
        require(fingerprint(canonical_path(root, relative)) == expected, "Frozen source changed")
    protected = read_json(root / "artifacts/module5/implementation/protected_inputs.json")
    require(protected["status"] == "PASS", "Protected reference is not PASS")
    require(
        file_hash(root / "artifacts/module5/implementation/protected_inputs.json")
        == policy["protected_reference_sha256"],
        "Protected input reference changed",
    )
    for relative, expected in protected["protected_files"].items():
        require(
            fingerprint(canonical_path(root, relative)) == expected,
            f"Predecessor changed: {relative}",
        )
    config = ExperimentConfig.from_yaml(
        root / "configs/experiments/module5/protocol.yaml", repo_root=root
    )
    context = prepare_experiments(config)
    require_frozen(context)
    require(config.config_hash == PROTOCOL and context.code_hash == CODE, "Frozen identity changed")
    require(
        str(torch.__version__) == TARGET["torch"] and config.cpu_threads == 2,
        "Runtime/settings changed",
    )
    runs, events = read_production(ex / "ledger.sqlite3")
    run = runs[RUN]
    validate_target(run, events, (ex / f"runs/{RUN}/outer_result.json").exists())
    require(len(runs) == 30 and len(events) == 135, "Unexpected later production activity")
    selection_path = ex / "selection/fold1_ablations.json"
    selection = read_json(selection_path)["selections"]["A1"]
    require(
        selection == run["data"]["selection"]
        and selection["epoch"] == 1
        and selection["candidate_id"] == "stronger_dropout",
        "Selection changed",
    )
    pointer = {"filename": f"{CHECKPOINT}.pt", "sha256": CHECKPOINT}
    require(run["data"]["final_checkpoint"] == pointer, "Wrong final checkpoint pointer")
    require(read_json(ex / f"runs/{RUN}/final/latest.json") == pointer, "Final pointer changed")
    ckpt = canonical_path(root, f"{BASE}/experiments/runs/{RUN}/final/{CHECKPOINT}.pt")
    checkpoint_payload(ckpt, run["data"]["final_identity"])
    records = context.split(1).records(context.predecessor.source.contract, "outer_test")
    validate_records(records)
    inputs = {
        "authorization_request": namespace / "authorization_request.txt",
        "forensic_report": root / FORENSICS / "FORENSIC_REPORT.md",
        "forensic_json": root / FORENSICS / "forensic_audit.json",
        "original_ledger": ex / "ledger.sqlite3",
        "original_events_csv": ex / "events.csv",
        "original_lock_journal": ex / f"locks/{digest(RUN)}.sqlite3-journal",
        "final_checkpoint": ckpt,
        "selection": selection_path,
        "protocol_yaml": root / "configs/experiments/module5/protocol.yaml",
        "protocol_json": root / BASE / "protocol.json",
        "authorization_policy": policy_path,
        "recovery_script": Path(__file__).resolve(),
    }
    require(
        file_hash(inputs["authorization_request"]) == policy["authorization_request_sha256"],
        "Original recovery authorization text changed",
    )
    binding = {
        "target": TARGET,
        "reason": REASON,
        "notice": NOTICE,
        "authorized_before_recovered_metrics": True,
        "inputs": {
            k: {"path": p.relative_to(root).as_posix(), "sha256": file_hash(p)}
            for k, p in inputs.items()
        },
        "production_fingerprints_hash": digest(production),
        "original_event_sequence": 135,
        "selection": selection,
        "final_identity": run["data"]["final_identity"],
    }
    require(inventory(ex) == production, "Production changed during preflight")
    return Prepared(root, context, run, records, binding, production, policy)


class ReadOnlyPermitLedger:
    """Only the two read methods used by the unchanged OuterPermit guard."""

    def __init__(self, path: Path, expected: dict[str, Any]) -> None:
        self.path, self.expected = path, expected

    def get(self, run_id: str) -> dict[str, Any]:
        require(run_id == RUN, "Permit is A1-only")
        runs, _ = read_production(self.path)
        require(runs[run_id] == self.expected, "Native claim/identity changed during recovery")
        return runs[run_id]

    def verify_claim(self, run_id: str, token: str) -> None:
        run = self.get(run_id)
        require(
            token == CLAIM == run["claim"] and run["state"] == "FINAL_TRAIN_COMPLETE",
            "Original claim no longer valid",
        )


def now() -> str:
    return datetime.now(UTC).isoformat()


def encoded(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def publish_bytes(path: Path, payload: bytes) -> None:
    """Atomic no-replace publication; identical completed artifacts are reused."""
    require(path.resolve() == path, "Redirected publication path")
    if path.exists():
        require(path.read_bytes() == payload, f"Immutable artifact differs: {path.name}")
        return
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    # Never replace an existing destination; leave evidence on publication failure.
    os.link(temp, path)
    temp.unlink()


def publish_json(path: Path, value: Any) -> None:
    publish_bytes(path, encoded(value))


@contextmanager
def execution_lock(namespace: Path):
    """OS-released lock in the EXTERNAL namespace; never touch native SQLite locks."""
    path = namespace / "execution.lock"
    require(path.resolve() == path, "Redirected external lock")
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def event_fields(attempt: str, index: int, records: tuple[Any, ...], prior: int) -> dict[str, Any]:
    ids = [r.logical_id for r in records]
    return {
        "attempt_id": attempt,
        "batch_index": index,
        "ordered_logical_ids": ids,
        "batch_record_hash": digest(ids),
        "checkpoint_hash": CHECKPOINT,
        "prior_cumulative_committed_count": prior,
    }


def append_event(path: Path, event: str, fields: dict[str, Any], previous: str) -> str:
    require(path.resolve() == path, "Redirected journal")
    value = {**fields, "event": event, "timestamp": now(), "previous_event_sha256": previous}
    value["event_sha256"] = digest(value)
    with path.open("ab") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    return value["event_sha256"]


def validate_rows(rows: list[Any], records: tuple[Any, ...]) -> None:
    require(len(rows) == len(records), "Prediction count differs")
    for row, record in zip(rows, records, strict=True):
        require(
            set(row) == {"logical_id", "speaker_id", "label", "probability"}
            and row["logical_id"] == record.logical_id
            and row["speaker_id"] == record.speaker_id
            and type(row["label"]) is int
            and row["label"] == label_for_condition(record.condition.value),
            "Prediction record/order/speaker/label differs",
        )
        score = row["probability"]
        require(
            type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1,
            "Invalid probability",
        )


@dataclass
class Progress:
    rows: list[Any]
    previous_hash: str
    finished_at: str
    fragments: list[str]


def inspect_journal(directory: Path, records: tuple[Any, ...]) -> Progress:
    """Read-only replay of evidence, NEVER replay of a model forward."""
    raw = (directory / "journal.jsonl").read_bytes()
    require(raw.endswith(b"\n"), "Truncated journal; STOP FOR REVIEW")
    entries = [json.loads(line) for line in raw.splitlines()]
    require(bool(entries), "Missing RECOVERY_AUTHORIZED")
    previous, rows, fragments, finished = "0" * 64, [], [], ""
    for position, entry in enumerate(entries):
        value = {k: v for k, v in entry.items() if k != "event_sha256"}
        require(
            entry["previous_event_sha256"] == previous and digest(value) == entry["event_sha256"],
            "Journal hash chain differs",
        )
        require(
            datetime.fromisoformat(entry["timestamp"]).utcoffset() is not None,
            "Journal timestamp is not timezone-aware",
        )
        previous, finished = entry["event_sha256"], entry["timestamp"]
        if position == 0:
            expected = event_fields(directory.name, -1, records, 0)
            require(entry["event"] == "RECOVERY_AUTHORIZED", "Authorization event missing")
        else:
            index = (position - 1) // 2
            start = index * 16
            batch_records = records[start : start + 16]
            require(bool(batch_records), "Extra journal batch")
            expected = event_fields(directory.name, index, batch_records, start)
            event = "BATCH_INTENT" if position % 2 else "BATCH_COMMITTED"
            require(entry["event"] == event, "Journal intent/commit order differs")
            if event == "BATCH_COMMITTED":
                relative = f"batches/{index:04d}.json"
                require(entry["fragment"] == relative, "Wrong fragment path")
                path = directory / relative
                require(file_hash(path) == entry["fragment_sha256"], "Fragment checksum differs")
                fragment = read_json(path)
                require(fragment["binding"] == expected, "Fragment binding differs")
                validate_rows(fragment["predictions"], batch_records)
                rows.extend(fragment["predictions"])
                fragments.append(relative)
        require(
            all(entry.get(k) == v for k, v in expected.items()), "Journal record binding differs"
        )
    require(len(entries) % 2 == 1, "Ambiguous BATCH_INTENT; STOP FOR REVIEW; never replay")
    existing = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    allowed = {
        "authorization.json",
        "journal.jsonl",
        "predictions.jsonl",
        "recovery_result.json",
        "recovery_manifest.json",
        *fragments,
    }
    require(existing <= allowed, "Unexpected/partial recovery artifact; STOP FOR REVIEW")
    require(
        len(rows) == len(records)
        or not existing.intersection(
            {"predictions.jsonl", "recovery_result.json", "recovery_manifest.json"}
        ),
        "Final artifact exists before full batch commitment",
    )
    require(not any(p.resolve() != p for p in directory.rglob("*")), "Redirected recovery artifact")
    return Progress(rows, previous, finished, fragments)


def authorization(prepared: Prepared, directory: Path) -> dict[str, Any]:
    value = read_json(directory / "authorization.json")
    require(
        value["binding"] == prepared.binding
        and value["attempt_id"] == directory.name
        and value["event"] == "RECOVERY_AUTHORIZED",
        "Authorization binding differs",
    )
    require(
        datetime.fromisoformat(value["authorized_at"]).utcoffset() is not None,
        "Invalid authorization timestamp",
    )
    return value


def assert_production(prepared: Prepared) -> None:
    require(
        inventory(prepared.root / BASE / "experiments") == prepared.production,
        "Production fingerprints changed; STOP FOR REVIEW",
    )
    for item in prepared.binding["inputs"].values():
        require(
            file_hash(canonical_path(prepared.root, item["path"])) == item["sha256"],
            "Authorized input changed; STOP FOR REVIEW",
        )
    require_frozen(prepared.context)


def recovery_model(prepared: Prepared):
    """Inference-only strict weight loading. No optimizer is constructed or loaded."""
    seed_everything(42, 2)
    candidate = prepared.context.config.candidate("stronger_dropout")
    model = build_model(prepared.context, "A1", candidate, load_pretrained=False)
    path = prepared.root / prepared.binding["inputs"]["final_checkpoint"]["path"]
    saved = checkpoint_payload(path, prepared.run["data"]["final_identity"])
    expected = model.state_dict()
    require(set(saved["model"]) == set(expected), "Model architecture keys differ")
    for name, tensor in saved["model"].items():
        require(
            isinstance(tensor, torch.Tensor)
            and tensor.shape == expected[name].shape
            and tensor.dtype == expected[name].dtype
            and bool(torch.isfinite(tensor).all()),
            "Invalid checkpoint model tensor",
        )
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    model.verify_frozen_backbone()
    return model


def recovery_data(prepared: Prepared, start: int):
    ledger = ReadOnlyPermitLedger(prepared.root / BASE / "experiments/ledger.sqlite3", prepared.run)
    dataset = ExperimentDataset(
        prepared.context,
        prepared.context.split(1),
        "outer_test",
        "A1",
        permit=OuterPermit(ledger, RUN, CLAIM),
    )
    loader = DataLoader(
        Subset(dataset, range(start, len(prepared.records))),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_encoder_samples,
        generator=torch.Generator().manual_seed(42),
    )
    return dataset, iter(loader)


def final_result(
    prepared: Prepared, directory: Path, auth: dict[str, Any], progress: Progress
) -> dict[str, Any]:
    require(len(progress.rows) == len(prepared.records), "Recovery incomplete")
    validate_rows(progress.rows, prepared.records)
    hashes = {
        p: file_hash(directory / p)
        for p in ["authorization.json", "journal.jsonl", "predictions.jsonl", *progress.fragments]
    }
    return {
        **TARGET,
        "attempt_id": directory.name,
        "notice": NOTICE,
        "provenance_flag": "RECOVERED_PROTOCOL_EXCEPTION",
        "authorization": auth,
        "recovery_forward_count": len(progress.rows),
        "recovery_source_load_count": len(progress.rows),
        "recovery_batch_count": len(progress.fragments),
        "started_at": auth["authorized_at"],
        "completed_at": progress.finished_at,
        "metrics": binary_metrics(
            [r["label"] for r in progress.rows],
            [r["probability"] for r in progress.rows],
            threshold=0.5,
        ),
        "predictions": progress.rows,
        "artifact_hashes": hashes,
        "production_fingerprints_hash_after": digest(prepared.production),
    }


def verify(prepared: Prepared, attempt_id: str) -> dict[str, Any]:
    """Separate read-only verifier; no model/dataset construction."""
    directory = prepared.directory(attempt_id)
    auth = authorization(prepared, directory)
    progress = inspect_journal(directory, prepared.records)
    expected_lines = b"".join(encoded(row) for row in progress.rows)
    require(
        (directory / "predictions.jsonl").read_bytes() == expected_lines,
        "Prediction stream differs from committed fragments",
    )
    result = read_json(directory / "recovery_result.json")
    require(
        result == final_result(prepared, directory, auth, progress),
        "Result identity/metrics/counts differ from retained predictions",
    )
    manifest = read_json(directory / "recovery_manifest.json")
    expected_files = {
        **result["artifact_hashes"],
        "recovery_result.json": file_hash(directory / "recovery_result.json"),
    }
    require(
        manifest
        == {
            "status": TARGET["status"],
            "classification": "PROTOCOL_EXCEPTION",
            "attempt_id": attempt_id,
            "notice": NOTICE,
            "artifact_hashes": expected_files,
            "production_fingerprints_hash": digest(prepared.production),
        },
        "Manifest differs",
    )
    require(
        {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
        == {*expected_files, "recovery_manifest.json"},
        "Unexpected recovery file",
    )
    assert_production(prepared)
    return {
        "status": "PASS",
        "attempt_id": attempt_id,
        "record_count": len(progress.rows),
        "record_hash": digest([r["logical_id"] for r in progress.rows]),
        "classification": "PROTOCOL_EXCEPTION",
        "notice": NOTICE,
        "verifier_model_forwards": 0,
        "production_unchanged": True,
        "result_sha256": file_hash(directory / "recovery_result.json"),
        "manifest_sha256": file_hash(directory / "recovery_manifest.json"),
    }


def run(prepared: Prepared, attempt_id: str, *, resume: bool = False) -> dict[str, Any]:
    directory = prepared.directory(attempt_id)
    with execution_lock(prepared.namespace):
        assert_production(prepared)
        if directory.exists():
            require(resume, "Attempt already exists; inspect preflight before explicit --resume")
            auth = authorization(prepared, directory)
            progress = inspect_journal(directory, prepared.records)
            if (directory / "recovery_manifest.json").exists():
                return verify(prepared, attempt_id)
        else:
            require(not resume, "Cannot resume a missing attempt")
            directory.mkdir()
            auth = {
                "event": "RECOVERY_AUTHORIZED",
                "attempt_id": attempt_id,
                "authorized_at": now(),
                "binding": prepared.binding,
            }
            publish_json(directory / "authorization.json", auth)
            (directory / "batches").mkdir()
            append_event(
                directory / "journal.jsonl",
                "RECOVERY_AUTHORIZED",
                event_fields(attempt_id, -1, prepared.records, 0),
                "0" * 64,
            )
            progress = inspect_journal(directory, prepared.records)
        start = len(progress.rows)
        if start < len(prepared.records):
            model = recovery_model(prepared)
            dataset, iterator = recovery_data(prepared, start)
            with torch.no_grad():
                for offset in range(start, len(prepared.records), 16):
                    index = offset // 16
                    records = prepared.records[offset : offset + 16]
                    fields = event_fields(attempt_id, index, records, offset)
                    previous = append_event(
                        directory / "journal.jsonl", "BATCH_INTENT", fields, progress.previous_hash
                    )
                    batch = next(iterator)  # Intent is durable BEFORE even loading this batch.
                    dataset.guard(batch)
                    require(
                        [p.logical_id for p in batch.provenance] == fields["ordered_logical_ids"],
                        "Batch order differs",
                    )
                    output = model(batch)
                    rows = [
                        {
                            "logical_id": p.logical_id,
                            "speaker_id": p.speaker_id,
                            "label": int(label),
                            "probability": score,
                        }
                        for p, label, score in zip(
                            batch.provenance,
                            batch_labels(batch).tolist(),
                            output.probabilities.tolist(),
                            strict=True,
                        )
                    ]
                    validate_rows(rows, records)
                    require(
                        dataset.loaded
                        == [r.logical_id for r in prepared.records[start : offset + len(records)]],
                        "Loaded source coverage differs",
                    )
                    relative = f"batches/{index:04d}.json"
                    publish_json(directory / relative, {"binding": fields, "predictions": rows})
                    progress.previous_hash = append_event(
                        directory / "journal.jsonl",
                        "BATCH_COMMITTED",
                        {
                            **fields,
                            "fragment": relative,
                            "fragment_sha256": file_hash(directory / relative),
                        },
                        previous,
                    )
                    print(
                        f"BATCH_COMMITTED {index + 1}/177 records={offset + len(rows)}", flush=True
                    )
            progress = inspect_journal(directory, prepared.records)
        assert_production(prepared)
        publish_bytes(
            directory / "predictions.jsonl", b"".join(encoded(row) for row in progress.rows)
        )
        result = final_result(prepared, directory, auth, progress)
        publish_json(directory / "recovery_result.json", result)
        publish_json(
            directory / "recovery_manifest.json",
            {
                "status": TARGET["status"],
                "classification": "PROTOCOL_EXCEPTION",
                "attempt_id": attempt_id,
                "notice": NOTICE,
                "artifact_hashes": {
                    **result["artifact_hashes"],
                    "recovery_result.json": file_hash(directory / "recovery_result.json"),
                },
                "production_fingerprints_hash": digest(prepared.production),
            },
        )
        return verify(prepared, attempt_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "verify"))
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly continue only fully committed batches; never replay intent",
    )
    args = parser.parse_args(argv)
    if args.resume and args.mode != "run":
        parser.error("--resume is only valid with run")
    try:
        prepared = prepare()
        directory = prepared.directory(args.attempt_id)
        if args.mode == "preflight":
            committed = 0
            if directory.exists():
                authorization(prepared, directory)
                committed = len(inspect_journal(directory, prepared.records).rows)
                if (directory / "recovery_manifest.json").exists():
                    verify(prepared, args.attempt_id)
            result = {
                "status": "PASS",
                "mode": "preflight",
                "attempt_id": args.attempt_id,
                "target": TARGET,
                "output_directory": str(directory),
                "batch_count": 177,
                "committed_records": committed,
                "model_forwards": 0,
                "production_unchanged": True,
                "authorization_created": False,
                "authorization_creation": "Only immediately before manual execution",
            }
        elif args.mode == "verify":
            result = verify(prepared, args.attempt_id)
        else:
            result = run(prepared, args.attempt_id, resume=args.resume)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        print(f"STOP FOR REVIEW: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
