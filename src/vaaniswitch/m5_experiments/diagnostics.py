"""Bounded real development-only checks; no outer dataset is constructed."""

from __future__ import annotations

from typing import Any

import torch

from vaaniswitch.m4_fusion.losses import batch_labels, make_loss
from vaaniswitch.m4_fusion.reproducibility import seed_everything
from vaaniswitch.m5_experiments.config import ABLATIONS, CANDIDATES
from vaaniswitch.m5_experiments.data import ExperimentDataset, batches
from vaaniswitch.m5_experiments.ledger import Ledger
from vaaniswitch.m5_experiments.pipeline import run_inner
from vaaniswitch.m5_experiments.reporting import write_json
from vaaniswitch.m5_experiments.source import ExperimentContext
from vaaniswitch.m5_experiments.trainer import build_model, make_optimizer


def smoke_models(context: ExperimentContext) -> dict[str, Any]:
    seed_everything(42, context.config.cpu_threads)
    split = context.split(1)
    results = {}
    for ablation in ABLATIONS:
        dataset = ExperimentDataset(context, split, "inner_train", ablation, smoke=True)
        batch = next(batches(dataset, smoke=True))
        dataset.guard(batch)
        model = build_model(context, ablation, CANDIDATES[0], load_pretrained=True)
        optimizer = make_optimizer(model, CANDIDATES[0])
        frozen = {n: t.clone() for n, t in model.state_dict().items() if ".backbone." in n}
        model.train()
        output = model(batch)
        loss = make_loss(1.0)(output.logits, batch_labels(batch))
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise ValueError("smoke gradient failure")
        optimizer.step()
        model.verify_frozen_backbone()
        if any(not torch.equal(t, model.state_dict()[n]) for n, t in frozen.items()):
            raise ValueError("frozen backbone weights/buffers changed")
        model.eval()
        with torch.no_grad():
            first = model(batch).logits
            second = model(batch).logits
        if not torch.equal(first, second):
            raise ValueError("smoke evaluation is not deterministic")
        results[ablation] = {
            "logits_shape": list(first.shape),
            "finite": True,
            "token_shapes": {name: list(t.shape) for name, t in output.tokens.items()},
            "parameter_counts": model.parameter_counts(),
            "frozen_state_equal": True,
            "deterministic": True,
            "logical_ids": [p.logical_id for p in batch.provenance],
        }
    inner = run_inner(
        context, Ledger(context.config, smoke=True), 1, "C2", CANDIDATES[0], smoke=True, resume=True
    )
    report = {
        "status": "PASS",
        "ablations": results,
        "tiny_inner": inner.identity,
        "outer_evaluation_count": 0,
        "fold1_source_loads": 0,
        "scientific_performance": False,
        "module5_signed_off": False,
    }
    write_json(context.config.output("smoke/diagnostics.json"), report)
    return report
