"""Module 3 validation and production encoder-smoke orchestration."""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torchvision

from vaaniswitch.m3_encoders.collate import collate_encoder_samples
from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.dataset import EncoderRepresentationDataset
from vaaniswitch.m3_encoders.diagnostics import (
    deterministic_eval,
    gradient_sanity,
    measure_cpu_timings,
    parameter_audit,
    state_dict_roundtrip,
    token_statistics,
)
from vaaniswitch.m3_encoders.models import ParallelTokenEncoders
from vaaniswitch.m3_encoders.schemas import EncoderBatch, EncoderTokenOutput
from vaaniswitch.m3_encoders.source import SignedRepresentationSource

LOGGER = logging.getLogger(__name__)
FAILURE_FIELDS = ("logical_id", "error_type", "message")
_HANDLERS: list[logging.Handler] = []


class Module3PipelineError(RuntimeError):
    """Raised when Module 3 validation or smoke evidence cannot pass."""


def validate_module3(config: Module3Config, *, limit: int = 8) -> dict[str, Any]:
    source = SignedRepresentationSource(config)
    logical_ids = _selected_ids(config, source, limit)
    dataset = EncoderRepresentationDataset(config, logical_ids=logical_ids, source=source)
    for index in range(len(dataset)):
        dataset[index]
    return {
        "schema_version": "1.0",
        "module": "3_parallel_token_encoders",
        "status": "PASS",
        "selected_record_count": len(dataset),
        "valid_representation_entries": len(dataset),
        "logical_ids": list(logical_ids),
        "encoder_config_hash": config.config_hash,
        "representation_config_hash": source.contract.representation_config_hash,
        "module1_config_hash": source.contract.module1_config_hash,
        "manifest_fingerprint": source.contract.manifest_fingerprint,
        "speaker_fold_source": config.source.speaker_folds_path.relative_to(
            config.repo_root
        ).as_posix(),
    }


def run_module3_smoke(
    config: Module3Config,
    *,
    limit: int = 8,
    batch_size: int = 2,
    timing_repetitions: int = 3,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise Module3PipelineError("batch size must be positive")
    _configure_logging(config.outputs.pipeline_log)
    _write_failures(config.outputs.failures, [])
    try:
        validation = validate_module3(config, limit=limit)
        source = SignedRepresentationSource(config)
        logical_ids = tuple(validation["logical_ids"])
        dataset = EncoderRepresentationDataset(config, logical_ids=logical_ids, source=source)
        torch.manual_seed(42)
        model = ParallelTokenEncoders(config, load_pretrained=True).cpu()
        model.eval()
        batches = _batches(dataset, batch_size)
        first_batch = batches[0]
        outputs: list[EncoderTokenOutput] = []
        with torch.inference_mode():
            for batch in batches:
                output = model(batch)
                output.validate(config.token_dimension)
                outputs.append(output)
        aggregate = EncoderTokenOutput(
            mel_tokens=torch.cat([output.mel_tokens for output in outputs]),
            graph_token=torch.cat([output.graph_token for output in outputs]),
            recurrence_tokens=torch.cat([output.recurrence_tokens for output in outputs]),
        )
        aggregate.validate(config.token_dimension)
        determinism = deterministic_eval(model, first_batch)
        gradients = gradient_sanity(model, first_batch)
        roundtrip = state_dict_roundtrip(model, config, first_batch)
        parameters = parameter_audit(model)
        timing = measure_cpu_timings(
            model, first_batch, repetitions=timing_repetitions
        )
        summary = {
            "schema_version": "1.0",
            "module": "3_parallel_token_encoders",
            "status": "PASS",
            "configuration": {
                "encoder_config_hash": config.config_hash,
                "processing_identity": config.processing_identity,
                "architecture_resolution": {
                    "provisional_token_dimension": 192,
                    "locked_token_dimension": 128,
                    "gcn_hidden_dimension": config.graph.hidden_dim,
                    "recurrence_channels": list(config.recurrence.channels),
                },
            },
            "source_validation": validation,
            "runtime": {
                "python_device": "cpu",
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(),
                "pretrained_weights": model.mel_encoder.pretrained_weights_identifier,
                "pretrained_weights_loaded": model.mel_encoder.pretrained_weights_identifier
                == "ResNet18_Weights.IMAGENET1K_V1",
            },
            "outputs": {
                "mel_tokens": list(aggregate.mel_tokens.shape),
                "graph_token": list(aggregate.graph_token.shape),
                "recurrence_tokens": list(aggregate.recurrence_tokens.shape),
                "nonlinear_tokens": list(aggregate.nonlinear_tokens.shape),
                "all_finite": all(
                    torch.isfinite(tensor).all().item()
                    for tensor in (
                        aggregate.mel_tokens,
                        aggregate.graph_token,
                        aggregate.recurrence_tokens,
                    )
                ),
            },
            "deterministic_eval": determinism,
            "gradient_sanity": gradients,
            "state_dict_roundtrip": roundtrip,
            "parameter_audit": parameters,
            "cpu_timing_seconds": timing,
            "token_statistics": {
                "mel": token_statistics(aggregate.mel_tokens),
                "graph": token_statistics(aggregate.graph_token),
                "recurrence": token_statistics(aggregate.recurrence_tokens),
            },
            "scientific_note": (
                "Token values are descriptive numerical sanity checks only; no classifier, "
                "class-separation analysis, performance metric, or training was run."
            ),
            "module4_started": False,
        }
        _require_passing_summary(summary)
        _write_json_atomic(config.outputs.parameter_audit, parameters)
        _write_json_atomic(config.outputs.summary, summary)
        LOGGER.info("Module 3 smoke PASS for %d signed records", len(dataset))
        return summary
    except (OSError, ValueError, RuntimeError) as exc:
        _write_failures(
            config.outputs.failures,
            [{"logical_id": "<pipeline>", "error_type": type(exc).__name__, "message": str(exc)}],
        )
        LOGGER.exception("Module 3 smoke failed")
        raise


def _selected_ids(
    config: Module3Config, source: SignedRepresentationSource, limit: int
) -> tuple[str, ...]:
    if limit <= 0:
        raise Module3PipelineError("limit must be positive")
    available = source.contract.record_by_id
    configured = tuple(item for item in config.smoke_logical_ids if item in available)
    if len(configured) < limit:
        raise Module3PipelineError("configured production smoke panel is too small")
    return configured[:limit]


def _batches(dataset: EncoderRepresentationDataset, batch_size: int) -> list[EncoderBatch]:
    result: list[EncoderBatch] = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        result.append(collate_encoder_samples([dataset[index] for index in range(start, stop)]))
    if not result:
        raise Module3PipelineError("smoke dataset is empty")
    return result


def _require_passing_summary(summary: dict[str, Any]) -> None:
    deterministic = summary["deterministic_eval"]
    gradients = summary["gradient_sanity"]
    if (
        not summary["outputs"]["all_finite"]
        or not all(deterministic.values())
        or not gradients["frozen_mel_backbone"]["all_requires_grad_false"]
        or not gradients["frozen_mel_backbone"]["all_gradients_absent"]
        or not gradients["mel_projection"]["all_gradients_present"]
        or not gradients["mel_projection"]["all_gradients_finite"]
        or not gradients["recurrence_cnn"]["all_gradients_present"]
        or not gradients["recurrence_cnn"]["all_gradients_finite"]
        or not gradients["visibility_graph_gcn"]["all_gradients_present"]
        or not gradients["visibility_graph_gcn"]["all_gradients_finite"]
        or not summary["state_dict_roundtrip"]["exact_match"]
    ):
        raise Module3PipelineError("one or more Module 3 smoke gates failed")


def _write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FAILURE_FIELDS))
        writer.writeheader()
        writer.writerows(failures)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in _HANDLERS:
        if handler in root.handlers:
            root.removeHandler(handler)
        handler.close()
    _HANDLERS.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.INFO)
    _HANDLERS.extend((file_handler, console_handler))
