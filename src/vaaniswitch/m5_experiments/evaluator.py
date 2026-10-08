"""One-use outer evaluator; no fitting, weighting, selection or threshold controls."""

from __future__ import annotations

from typing import Any

import torch

from vaaniswitch.m4_fusion.losses import batch_labels
from vaaniswitch.m4_fusion.metrics import binary_metrics
from vaaniswitch.m4_fusion.reproducibility import seed_everything
from vaaniswitch.m5_experiments.checkpoint import load_checkpoint
from vaaniswitch.m5_experiments.config import ABLATIONS, file_hash
from vaaniswitch.m5_experiments.data import ExperimentDataset, OuterPermit, batches
from vaaniswitch.m5_experiments.ledger import Ledger
from vaaniswitch.m5_experiments.reporting import read_json, write_json
from vaaniswitch.m5_experiments.source import ExperimentContext, require_frozen
from vaaniswitch.m5_experiments.trainer import build_model, make_optimizer


def evaluate_outer(context: ExperimentContext, fold: int, ablation: str) -> dict[str, Any]:
    require_frozen(context)
    if ablation not in ABLATIONS:
        raise ValueError("unknown outer ablation")
    ledger = Ledger(context.config)
    bundle = read_json(context.config.output(f"experiments/selection/fold{fold}_ablations.json"))
    # Freeze all comparison models before exposing any outer predictions from this fold.
    for variant in ABLATIONS:
        run = ledger.get(bundle["selections"][variant]["experiment_id"])
        if run["state"] not in {"FINAL_TRAIN_COMPLETE", "OUTER_EVALUATED"}:
            raise ValueError("all ablations in this fold must be selected and retrained first")
    selection = bundle["selections"][ablation]
    run_id = selection["experiment_id"]
    path = context.config.output(f"experiments/runs/{run_id}/outer_result.json")
    with ledger.lock(run_id):
        run = ledger.get(run_id)
        if run["state"] == "OUTER_EVALUATED":
            if file_hash(path) != run["data"]["outer_result_sha256"]:
                raise ValueError("completed outer evidence was modified")
            return read_json(path)
        if run["state"] != "FINAL_TRAIN_COMPLETE" or run["data"]["selection"] != selection:
            raise ValueError("outer evaluation blocked before final selected retrain")
        if run["claim"] is not None:
            raise ValueError(
                "outer already claimed; preserve interrupted evidence for defect review"
            )
        candidate = context.config.candidate(selection["candidate_id"])
        split = context.split(fold)
        seed_everything(42, context.config.cpu_threads)
        model = build_model(context, ablation, candidate, load_pretrained=False)
        optimizer = make_optimizer(model, candidate)
        saved = load_checkpoint(
            context.config.output(f"experiments/runs/{run_id}/final"),
            model,
            optimizer,
            run["data"]["final_identity"],
            pointer=run["data"]["final_checkpoint"],
        )
        if saved["progress"]["epoch"] != selection["epoch"]:
            raise ValueError("outer checkpoint epoch differs from selected epoch")
        token = ledger.claim_outer(run_id)
        try:
            dataset = ExperimentDataset(
                context, split, "outer_test", ablation, permit=OuterPermit(ledger, run_id, token)
            )
            predictions: list[dict[str, Any]] = []
            model.eval()
            with torch.no_grad():
                for batch in batches(dataset):
                    dataset.guard(batch)
                    output = model(batch)
                    labels = batch_labels(batch).tolist()
                    for item, label, score in zip(
                        batch.provenance, labels, output.probabilities.tolist(), strict=True
                    ):
                        predictions.append(
                            {
                                "logical_id": item.logical_id,
                                "speaker_id": item.speaker_id,
                                "label": int(label),
                                "probability": score,
                            }
                        )
            if [p["logical_id"] for p in predictions] != [r.logical_id for r in dataset.records]:
                raise ValueError("outer predictions must cover assigned records exactly once")
            result = {
                "status": "OUTER_EVALUATED",
                "experiment_id": run_id,
                "ablation": ablation,
                "outer_fold": fold,
                "selection": selection,
                "identity": run["identity"],
                "checkpoint": run["data"]["final_checkpoint"],
                "threshold": 0.5,
                "parameter_counts": run["data"]["parameter_counts"],
                "inner_training_seconds": run["data"]["inner_training_seconds"],
                "final_training_seconds": run["data"]["final_training_seconds"],
                "source_load_count": len(dataset.loaded),
                "forward_count": len(predictions),
                "metrics": binary_metrics(
                    [p["label"] for p in predictions], [p["probability"] for p in predictions]
                ),
                "predictions": predictions,
                "claim": token,
            }
            write_json(path, result, immutable=True)
            ledger.transition(run_id, "OUTER_EVALUATED", {"outer_result_sha256": file_hash(path)})
            return result
        except Exception as exc:
            ledger.transition(run_id, "FAILED", {"reason": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            ledger.export()
