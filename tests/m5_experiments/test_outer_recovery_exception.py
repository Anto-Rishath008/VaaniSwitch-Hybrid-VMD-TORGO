"""External recovery only: synthetic metadata and mocked forward, never real inference."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from scripts import module5_recover_outer_exception as r

ATTEMPT = "a1_20261005T120000Z_abcdef123456"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.encoded(value))


def record(index: int):
    return NS(
        logical_id=f"record/{index:04d}",
        speaker_id=r.TARGET["speakers"][index % 5],
        condition=NS(value="control" if index % 2 == 0 else "dysarthric"),
    )


def target_run():
    return {
        "claim": r.CLAIM,
        "state": "FINAL_TRAIN_COMPLETE",
        "identity": {
            "experiment_id": r.RUN,
            "outer_fold": 1,
            "ablation": "A1",
            "seed": 42,
            "threshold": 0.5,
            "torch": r.TARGET["torch"],
            "cpu_threads": 2,
            "config_hash": r.PROTOCOL,
            "code_hash": r.CODE,
            "candidate": {"id": "stronger_dropout"},
        },
        "data": {},
    }


def target_events():
    return [
        (
            135,
            r.RUN,
            "2026-09-29T04:44:19.996521+00:00",
            "OUTER_CLAIMED",
            json.dumps({"claim": r.CLAIM}),
        )
    ]


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real model/native writer/training/optimizer must never execute")

    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    from vaaniswitch.m5_experiments import evaluator, ledger, trainer

    monkeypatch.setattr(ledger.Ledger, "__init__", forbidden)
    monkeypatch.setattr(evaluator, "evaluate_outer", forbidden)
    monkeypatch.setattr(trainer, "make_optimizer", forbidden)
    monkeypatch.setattr(trainer, "pos_weight", forbidden)
    monkeypatch.setattr(trainer, "run_epoch", forbidden)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    ex = root / r.BASE / "experiments"
    ex.mkdir(parents=True)
    (ex / "ledger.sqlite3").write_bytes(b"protected-ledger-sentinel")
    (ex / "events.csv").write_bytes(b"protected-events-sentinel")
    namespace = root / r.RECOVERY
    namespace.mkdir(parents=True)
    records = tuple(record(i) for i in range(35))
    monkeypatch.setattr(r, "require_frozen", Mock())
    return r.Prepared(
        root,
        NS(),
        target_run(),
        records,
        {"inputs": {}, "target": r.TARGET},
        r.inventory(ex),
        {"attempt_id": ATTEMPT},
    )


@pytest.fixture
def mock_runtime(monkeypatch):
    model = Mock()
    starts, loaded = [], []

    def predict(batch):
        assert not torch.is_grad_enabled()
        return NS(
            probabilities=torch.tensor(
                [0.2 if p.condition == "control" else 0.8 for p in batch.provenance]
            )
        )

    model.side_effect = predict
    factory = Mock(return_value=model)
    monkeypatch.setattr(r, "recovery_model", factory)

    def data(prepared, start):
        starts.append(start)
        dataset = NS(loaded=[], guard=Mock())

        def batches():
            for offset in range(start, len(prepared.records), 16):
                subset = prepared.records[offset : offset + 16]
                dataset.loaded.extend(x.logical_id for x in subset)
                loaded.extend(x.logical_id for x in subset)
                yield NS(
                    provenance=[
                        NS(
                            logical_id=x.logical_id,
                            speaker_id=x.speaker_id,
                            condition=x.condition.value,
                        )
                        for x in subset
                    ],
                    mel=torch.empty((len(subset), 1)),
                )

        return dataset, iter(batches())

    monkeypatch.setattr(r, "recovery_data", data)
    return NS(model=model, factory=factory, starts=starts, loaded=loaded)


@pytest.mark.parametrize(
    "key,value",
    [
        ("outer_fold", 2),
        ("ablation", "A0"),
        ("experiment_id", "other"),
        ("seed", 43),
        ("threshold", 0.6),
        ("torch", "different"),
        ("cpu_threads", 4),
        ("config_hash", "different"),
        ("code_hash", "different"),
    ],
)
def test_exact_target_and_settings_only(key, value):
    run = target_run()
    run["identity"][key] = value
    with pytest.raises(r.ReviewRequired):
        r.validate_target(run, target_events(), False)


@pytest.mark.parametrize("damage", ["claim", "state", "event", "native", "candidate"])
def test_claim_state_native_result_and_event_guards(damage):
    run, events = target_run(), target_events()
    if damage == "claim":
        run["claim"] = "other"
    elif damage == "state":
        run["state"] = "OUTER_EVALUATED"
    elif damage == "event":
        events.append((136, r.RUN, "later", "OUTER_EVALUATED", "{}"))
    elif damage == "candidate":
        run["identity"]["candidate"]["id"] = "baseline"
    with pytest.raises(r.ReviewRequired):
        r.validate_target(run, events, damage == "native")


def test_valid_native_claim_metadata():
    r.validate_target(target_run(), target_events(), False)


def test_missing_forensic_evidence_rejected_without_production_writes(tmp_path):
    policy = {
        "attempt_id": ATTEMPT,
        "target": r.TARGET,
        "authorized_before_recovered_metrics": True,
        "recovery_script_sha256": r.file_hash(Path(r.__file__)),
    }
    dump(tmp_path / r.RECOVERY / "authorization_policy.json", policy)
    before = r.inventory(tmp_path)
    with pytest.raises(FileNotFoundError):
        r.prepare(tmp_path)
    assert r.inventory(tmp_path) == before


def test_changed_checkpoint_rejected_before_torch_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"changed")
    loader = Mock(side_effect=AssertionError("must not deserialize"))
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(r.ReviewRequired, match="Changed checkpoint"):
        r.checkpoint_payload(checkpoint, {})
    loader.assert_not_called()


def test_ordered_record_hash_duplicates_and_speakers(monkeypatch):
    records = tuple(record(i) for i in range(2819))
    monkeypatch.setattr(r, "RECORD_HASH", r.digest([x.logical_id for x in records]))
    r.validate_records(records)
    with pytest.raises(r.ReviewRequired, match="ordered record hash"):
        r.validate_records(tuple(reversed(records)))
    with pytest.raises(r.ReviewRequired, match="duplicates"):
        r.validate_records((*records[:-1], records[0]))


def test_sqlite_uri_is_read_only_and_permit_cannot_write(tmp_path, monkeypatch):
    path = tmp_path / "ledger.sqlite3"
    run = target_run()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE runs(id,identity,state,data,claim)")
        db.execute("CREATE TABLE events(seq,run_id,time,event,detail)")
        db.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?)",
            (r.RUN, json.dumps(run["identity"]), run["state"], "{}", r.CLAIM),
        )
    before = r.fingerprint(path)
    real = sqlite3.connect

    def connect(database, **kwargs):
        assert database.endswith("?mode=ro") and kwargs == {"uri": True}
        connection = real(database, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("UPDATE runs SET claim='bad'")
        return connection

    monkeypatch.setattr(r.sqlite3, "connect", connect)
    adapter = r.ReadOnlyPermitLedger(path.resolve(), run)
    adapter.verify_claim(r.RUN, r.CLAIM)
    with pytest.raises(r.ReviewRequired):
        adapter.verify_claim(r.RUN, "bad")
    assert r.fingerprint(path) == before


@pytest.mark.parametrize("attempt", ["../../native", "other", "a1_20261005T120000Z_000000000000"])
def test_only_reserved_attempt_allowed(prepared, attempt):
    with pytest.raises(r.ReviewRequired):
        prepared.directory(attempt)


def test_full_mocked_recovery_verifier_no_native_writes(prepared, mock_runtime):
    before = r.inventory(prepared.root / r.BASE / "experiments")
    result = r.run(prepared, ATTEMPT)
    assert result["status"] == "PASS"
    assert mock_runtime.model.call_count == 3
    directory = prepared.directory(ATTEMPT)
    journal = [json.loads(line) for line in (directory / "journal.jsonl").read_text().splitlines()]
    assert [x["event"] for x in journal] == ["RECOVERY_AUTHORIZED"] + [
        "BATCH_INTENT",
        "BATCH_COMMITTED",
    ] * 3
    assert [e["prior_cumulative_committed_count"] for e in journal[1::2]] == [0, 16, 32]
    retained = r.read_json(directory / "recovery_result.json")
    assert retained["notice"] == r.NOTICE and retained["original_prior_forward_count"] == "UNKNOWN"
    assert retained["metrics"]["accuracy"] == 1.0
    assert retained["recovery_forward_count"] == retained["recovery_source_load_count"] == 35
    assert r.inventory(prepared.root / r.BASE / "experiments") == before
    assert not (prepared.root / r.BASE / "experiments/runs" / r.RUN / "outer_result.json").exists()
    saved = r.inventory(directory)
    assert r.verify(prepared, ATTEMPT)["status"] == "PASS"
    assert r.run(prepared, ATTEMPT, resume=True)["status"] == "PASS"
    assert r.inventory(directory) == saved
    assert mock_runtime.model.call_count == 3
    with pytest.raises(r.ReviewRequired, match="already exists"):
        r.run(prepared, ATTEMPT)


@pytest.mark.parametrize("point", ["before_forward", "after_forward", "after_fragment"])
def test_ambiguous_batch_refuses_replay(prepared, mock_runtime, monkeypatch, point):
    if point == "before_forward":
        monkeypatch.setattr(r, "recovery_data", lambda *_: (NS(guard=Mock()), iter(())))
    elif point == "after_forward":
        original = r.publish_json

        def publish(path, value):
            if path.parent.name == "batches":
                raise OSError("interrupted publication")
            return original(path, value)

        monkeypatch.setattr(r, "publish_json", publish)
    else:
        original = r.append_event

        def append(path, event, fields, previous):
            if event == "BATCH_COMMITTED":
                raise OSError("interrupted commit")
            return original(path, event, fields, previous)

        monkeypatch.setattr(r, "append_event", append)
    with pytest.raises((OSError, StopIteration)):
        r.run(prepared, ATTEMPT)
    count = mock_runtime.model.call_count
    snapshot = r.inventory(prepared.directory(ATTEMPT))
    with pytest.raises(r.ReviewRequired, match="Ambiguous BATCH_INTENT"):
        r.run(prepared, ATTEMPT, resume=True)
    assert mock_runtime.model.call_count == count
    assert r.inventory(prepared.directory(ATTEMPT)) == snapshot


def test_safe_resume_skips_committed_batches(prepared, mock_runtime, monkeypatch):
    original = r.append_event

    def stop_before_intent(path, event, fields, previous):
        if event == "BATCH_INTENT" and fields["batch_index"] == 1:
            raise OSError("stopped between committed batches")
        return original(path, event, fields, previous)

    monkeypatch.setattr(r, "append_event", stop_before_intent)
    with pytest.raises(OSError):
        r.run(prepared, ATTEMPT)
    assert mock_runtime.model.call_count == 1
    monkeypatch.setattr(r, "append_event", original)
    assert r.run(prepared, ATTEMPT, resume=True)["status"] == "PASS"
    assert mock_runtime.starts == [0, 16]
    assert mock_runtime.loaded == [x.logical_id for x in prepared.records]
    assert mock_runtime.model.call_count == 3


def test_truncated_journal_and_orphan_fragment_refuse(prepared, mock_runtime):
    r.run(prepared, ATTEMPT)
    directory = prepared.directory(ATTEMPT)
    path = directory / "journal.jsonl"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(r.ReviewRequired, match="Truncated"):
        r.inspect_journal(directory, prepared.records)


@pytest.mark.parametrize(
    "field,value",
    [
        ("probability", float("nan")),
        ("probability", 1.1),
        ("probability", True),
        ("logical_id", "different"),
        ("speaker_id", "other"),
        ("label", True),
    ],
)
def test_prediction_identity_and_probability_validation(field, value):
    rec = record(0)
    row = {
        "logical_id": rec.logical_id,
        "speaker_id": rec.speaker_id,
        "label": 0,
        "probability": 0.2,
    }
    row[field] = value
    with pytest.raises(r.ReviewRequired):
        r.validate_rows([row], (rec,))


def test_metric_recomputation_rejects_tampered_result(prepared, mock_runtime):
    r.run(prepared, ATTEMPT)
    path = prepared.directory(ATTEMPT) / "recovery_result.json"
    result = r.read_json(path)
    result["metrics"]["accuracy"] = 0.0
    dump(path, result)  # Synthetic fixture only.
    with pytest.raises(r.ReviewRequired, match="metrics"):
        r.verify(prepared, ATTEMPT)


def test_immutable_publication_and_fsync(tmp_path, monkeypatch):
    synced = Mock(wraps=r.os.fsync)
    monkeypatch.setattr(r.os, "fsync", synced)
    path = tmp_path / "result.json"
    r.publish_json(path, {"valid": True})
    before = r.fingerprint(path)
    r.publish_json(path, {"valid": True})
    with pytest.raises(r.ReviewRequired, match="Immutable"):
        r.publish_json(path, {"valid": False})
    assert r.fingerprint(path) == before and synced.call_count >= 1


def test_production_change_blocks_resume_before_forward(prepared, mock_runtime):
    path = prepared.root / r.BASE / "experiments/events.csv"
    path.write_bytes(b"changed")
    with pytest.raises(r.ReviewRequired, match="Production fingerprints"):
        r.run(prepared, ATTEMPT)
    mock_runtime.factory.assert_not_called()


def test_preflight_cli_constructs_no_model_or_authorization(prepared, monkeypatch, capsys):
    monkeypatch.setattr(r, "prepare", lambda: prepared)
    forbidden = Mock(side_effect=AssertionError("No model in preflight"))
    monkeypatch.setattr(r, "recovery_model", forbidden)
    before = r.inventory(prepared.namespace)
    assert r.main(["preflight", "--attempt-id", ATTEMPT]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_forwards"] == 0 and result["authorization_created"] is False
    assert r.inventory(prepared.namespace) == before
    forbidden.assert_not_called()


def test_inference_loader_has_no_optimizer_and_strict_weights(prepared, monkeypatch):
    model = Mock()
    model.state_dict.return_value = {"weight": torch.tensor([1.0])}
    prepared.context.config = NS(candidate=lambda _: NS())
    prepared.binding["inputs"]["final_checkpoint"] = {"path": "fake.pt"}
    prepared.run["data"]["final_identity"] = {"stage": "final"}
    monkeypatch.setattr(r, "build_model", Mock(return_value=model))
    monkeypatch.setattr(
        r, "checkpoint_payload", Mock(return_value={"model": {"weight": torch.tensor([2.0])}})
    )
    assert r.recovery_model(prepared) is model
    model.eval.assert_called_once()
    model.requires_grad_.assert_called_once_with(False)
    model.load_state_dict.assert_called_once()
    assert model.load_state_dict.call_args.kwargs == {"strict": True}


def test_external_lock_excludes_concurrent_writer_and_releases(prepared):
    with (
        r.execution_lock(prepared.namespace),
        pytest.raises((OSError, BlockingIOError)),
        r.execution_lock(prepared.namespace),
    ):
        pytest.fail("Concurrent lock acquired")
    with r.execution_lock(prepared.namespace):
        pass


def test_original_loader_settings_and_readonly_permit(prepared, monkeypatch):
    prepared.context.split = Mock(return_value=NS())
    dataset = Mock()
    dataset_factory = Mock(return_value=dataset)
    loader = Mock(return_value=iter(()))
    monkeypatch.setattr(r, "ExperimentDataset", dataset_factory)
    monkeypatch.setattr(r, "DataLoader", loader)
    monkeypatch.setattr(r, "Subset", Mock(return_value="remaining-records"))
    r.recovery_data(prepared, 16)
    kwargs = loader.call_args.kwargs
    assert kwargs["batch_size"] == 16 and kwargs["shuffle"] is False
    assert kwargs["num_workers"] == 0 and kwargs["drop_last"] is False
    assert kwargs["generator"].initial_seed() == 42
    permit = dataset_factory.call_args.kwargs["permit"]
    assert isinstance(permit.ledger, r.ReadOnlyPermitLedger) and permit.token == r.CLAIM


@pytest.mark.parametrize("damage", ["orphan", "chain", "fragment", "manifest", "threshold"])
def test_tampered_recovery_evidence_rejected(prepared, mock_runtime, damage):
    r.run(prepared, ATTEMPT)
    directory = prepared.directory(ATTEMPT)
    if damage == "orphan":
        (directory / "batches/9999.json").write_bytes(b"orphan")
    elif damage == "chain":
        path = directory / "journal.jsonl"
        text = path.read_text()
        path.write_text(
            text.replace(
                '"prior_cumulative_committed_count":0', '"prior_cumulative_committed_count":1', 1
            )
        )
    elif damage == "fragment":
        path = directory / "batches/0000.json"
        fragment = r.read_json(path)
        fragment["predictions"][0]["probability"] = 0.6
        dump(path, fragment)
    elif damage == "manifest":
        path = directory / "recovery_manifest.json"
        value = r.read_json(path)
        value["artifact_hashes"]["recovery_result.json"] = "0" * 64
        dump(path, value)
    else:
        path = directory / "recovery_result.json"
        value = r.read_json(path)
        value["threshold"] = 0.4
        dump(path, value)
    before = mock_runtime.model.call_count
    with pytest.raises(r.ReviewRequired):
        r.verify(prepared, ATTEMPT)
    assert mock_runtime.model.call_count == before


def test_model_exception_intent_is_already_fsynced(prepared, mock_runtime, monkeypatch):
    original = r.os.fsync
    durable = []

    def fsync(fd):
        original(fd)
        durable.append(True)

    monkeypatch.setattr(r.os, "fsync", fsync)

    def forward(batch):
        directory = prepared.directory(ATTEMPT)
        lines = (directory / "journal.jsonl").read_bytes().splitlines()
        assert json.loads(lines[-1])["event"] == "BATCH_INTENT"
        assert len(durable) >= 4  # lock, authorization, authorized event, intent
        raise KeyboardInterrupt

    mock_runtime.model.side_effect = forward
    with pytest.raises(KeyboardInterrupt):
        r.run(prepared, ATTEMPT)
    with pytest.raises(r.ReviewRequired, match="Ambiguous"):
        r.run(prepared, ATTEMPT, resume=True)
    assert mock_runtime.model.call_count == 1


def test_all_committed_finalization_can_resume_without_model(prepared, mock_runtime, monkeypatch):
    original = r.publish_bytes

    def stop_final(path, payload):
        if path.name == "predictions.jsonl":
            raise OSError("interrupted before final publication")
        original(path, payload)

    monkeypatch.setattr(r, "publish_bytes", stop_final)
    with pytest.raises(OSError):
        r.run(prepared, ATTEMPT)
    assert mock_runtime.model.call_count == 3
    monkeypatch.setattr(r, "publish_bytes", original)
    assert r.run(prepared, ATTEMPT, resume=True)["status"] == "PASS"
    assert mock_runtime.model.call_count == 3
