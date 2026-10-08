"""Transactional experiment states, explicit failure recovery, and one-shot outer claims."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vaaniswitch.m5_experiments.config import ExperimentConfig, digest

TRANSITIONS = {
    "PLANNED": {"RUNNING", "FAILED", "INVALIDATED"},
    "RUNNING": {"INNER_COMPLETE", "FAILED", "INVALIDATED"},
    "INNER_COMPLETE": {"SELECTED", "FAILED", "INVALIDATED"},
    "SELECTED": {"FINAL_TRAIN_COMPLETE", "FAILED", "INVALIDATED"},
    "FINAL_TRAIN_COMPLETE": {"OUTER_EVALUATED", "FAILED", "INVALIDATED"},
    "OUTER_EVALUATED": {"INVALIDATED"},
    "FAILED": {"INVALIDATED"},
    "INVALIDATED": set(),
}


def experiment_id(
    config: ExperimentConfig, fold: int, ablation: str, candidate: str, *, smoke: bool = False
) -> str:
    if (
        type(fold) is not int
        or fold not in (1, 2, 3)
        or ablation not in config.identity["ablations"]
    ):
        raise ValueError("invalid experiment fold/ablation")
    config.candidate(candidate)
    return f"fold{fold}_{ablation}_{candidate}_seed42_{config.config_hash[:12]}" + (
        "_smoke" if smoke else ""
    )


class Ledger:
    def __init__(self, config: ExperimentConfig, *, smoke: bool = False) -> None:
        self.config = config
        self.prefix = "smoke" if smoke else "experiments"
        self.path = config.output(f"{self.prefix}/ledger.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, "
                "identity TEXT NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL, claim TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, "
                "run_id TEXT NOT NULL, time TEXT NOT NULL, event TEXT NOT NULL, "
                "detail TEXT NOT NULL)"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def lock(self, name: str) -> Iterator[None]:
        # A separate SQLite exclusive transaction is an OS-managed process lock. A killed
        # process releases it automatically; no stale PID files or guessed deletions.
        path = self.config.output(f"{self.prefix}/locks/{digest(name)}.sqlite3")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=0.1)
        try:
            connection.execute("BEGIN EXCLUSIVE")
            yield
        except sqlite3.OperationalError as exc:
            raise ValueError("experiment is locked by another process") from exc
        finally:
            connection.rollback()
            connection.close()

    @staticmethod
    def event(connection: sqlite3.Connection, run_id: str, event: str, detail: Any) -> None:
        connection.execute(
            "INSERT INTO events(run_id,time,event,detail) VALUES(?,?,?,?)",
            (run_id, datetime.now(UTC).isoformat(), event, json.dumps(detail, allow_nan=False)),
        )

    def plan(self, run_id: str, identity: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT identity FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            encoded = json.dumps(identity, sort_keys=True, allow_nan=False)
            if previous:
                if previous[0] != encoded:
                    raise ValueError("experiment identity changed; resume forbidden")
                return
            connection.execute(
                "INSERT INTO runs VALUES(?,?,?,?,NULL)", (run_id, encoded, "PLANNED", "{}")
            )
            self.event(connection, run_id, "PLANNED", identity)

    def get(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT identity,state,data,claim FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise ValueError("experiment is not planned")
        return {
            "id": run_id,
            "identity": json.loads(row[0]),
            "state": row[1],
            "data": json.loads(row[2]),
            "claim": row[3],
        }

    def all_runs(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            ids = [row[0] for row in connection.execute("SELECT id FROM runs ORDER BY id")]
        return [self.get(run_id) for run_id in ids]

    def transition(self, run_id: str, state: str, data: dict[str, Any] | None = None) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state,data FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None or state not in TRANSITIONS.get(row[0], set()):
                raise ValueError(f"invalid ledger transition to {state}")
            merged = {**json.loads(row[1]), **(data or {})}
            if state in {"FAILED", "INVALIDATED"}:
                if not data or not data.get("reason"):
                    raise ValueError("failure/invalidation requires an explicit reason")
                merged["previous_state"] = row[0]
            if state == "SELECTED" and not merged.get("selection"):
                raise ValueError("selection evidence required")
            if state == "FINAL_TRAIN_COMPLETE" and not merged.get("final_checkpoint"):
                raise ValueError("final checkpoint evidence required")
            if state == "OUTER_EVALUATED" and not merged.get("outer_result_sha256"):
                raise ValueError("immutable outer result evidence required")
            connection.execute(
                "UPDATE runs SET state=?,data=? WHERE id=?",
                (state, json.dumps(merged, allow_nan=False, sort_keys=True), run_id),
            )
            self.event(connection, run_id, state, data)

    def retry_failed(self, run_id: str, reason: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state,data,claim FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None or row[0] != "FAILED" or row[2] is not None or not reason.strip():
                raise ValueError("explicit recovery applies only to failed pre-outer runs")
            target = json.loads(row[1])["previous_state"]
            if target not in {"PLANNED", "RUNNING", "SELECTED"}:
                raise ValueError("failure cannot be resumed automatically")
            connection.execute("UPDATE runs SET state=? WHERE id=?", (target, run_id))
            self.event(connection, run_id, "EXPLICIT_RETRY", {"reason": reason, "state": target})

    def claim_outer(self, run_id: str) -> str:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            token = uuid.uuid4().hex
            cursor = connection.execute(
                "UPDATE runs SET claim=? WHERE id=? AND "
                "state='FINAL_TRAIN_COMPLETE' AND claim IS NULL",
                (token, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("outer evaluation requires final retrain and can be claimed once")
            self.event(connection, run_id, "OUTER_CLAIMED", {"claim": token})
            return token

    def verify_claim(self, run_id: str, token: str) -> None:
        run = self.get(run_id)
        if run["state"] != "FINAL_TRAIN_COMPLETE" or not token or run["claim"] != token:
            raise ValueError("invalid outer evaluation permit")

    def export(self) -> Path:
        import csv

        path = self.config.output(f"{self.prefix}/events.csv")
        with self.connect() as connection, path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sequence", "experiment_id", "time", "event", "detail"])
            writer.writerows(connection.execute("SELECT * FROM events ORDER BY seq"))
        return path
