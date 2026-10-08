"""Tiny signed Module 2 repositories and Module 3 model fixtures."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from tests.m2_representations.helpers import prepare_tiny_repo, write_config_yaml
from vaaniswitch.m1_vmd.manifest import load_manifest_source
from vaaniswitch.m2_representations import pipeline as module2_pipeline
from vaaniswitch.m2_representations.pipeline import run_module2
from vaaniswitch.m3_encoders.config import (
    GraphEncoderConfig,
    MelEncoderConfig,
    Module3Config,
    OutputConfig,
    RecurrenceEncoderConfig,
    SourceConfig,
)
from vaaniswitch.m3_encoders.schemas import (
    EncoderBatch,
    EncoderSample,
    SampleProvenance,
)


def make_model_config(repo_root: Path) -> Module3Config:
    placeholder = "a" * 64
    return Module3Config(
        schema_version="1.0",
        implementation="vaaniswitch_pytorch_encoders_v1",
        repo_root=repo_root.resolve(),
        source=SourceConfig(
            manifest_path=(repo_root / "data/manifests/torgo_manifest.jsonl").resolve(),
            speaker_folds_path=(repo_root / "data/splits/speaker_folds.json").resolve(),
            module2_config_path=(repo_root / "configs/representations/default.yaml").resolve(),
            module2_signoff_path=(
                repo_root / "artifacts/metrics/module2_full_signoff_audit.json"
            ).resolve(),
            representation_cache_root=(
                repo_root / "data/processed/representations"
            ).resolve(),
            representation_config_hash=placeholder,
            module1_config_hash="b" * 64,
            manifest_fingerprint=placeholder,
            subset="core",
        ),
        token_dimension=128,
        mel=MelEncoderConfig(
            backbone="resnet18",
            input_channels=3,
            pretrained_weights="IMAGENET1K_V1",
            replicate_single_channel=True,
            fixed_db_min=-100.0,
            fixed_db_max=20.0,
            imagenet_normalization=True,
            adaptive_pool=(4, 4),
            tokens=16,
            freeze_backbone_initially=True,
        ),
        graph=GraphEncoderConfig(
            implementation="pure_pytorch_gcn",
            input_dim=2,
            layers=2,
            hidden_dim=96,
            graph_feature_dim=128,
            activation="gelu",
            pooling="mean_plus_max",
            tokens=1,
            add_self_loops=True,
            adjacency_normalization="symmetric",
        ),
        recurrence=RecurrenceEncoderConfig(
            implementation="small_cnn",
            input_channels=1,
            channels=(16, 32, 64),
            normalization="groupnorm",
            activation="gelu",
            adaptive_pool=(4, 4),
            tokens=16,
        ),
        dropout=0.2,
        smoke_logical_ids=(
            "F01/Session1/0001",
            "FC01/Session1/0001",
            "M01/Session1/0001",
            "MC01/Session1/0001",
            "F03/Session1/0179",
            "F03/Session1/0118",
        ),
        outputs=OutputConfig(
            summary=(repo_root / "artifacts/metrics/module3_encoder_summary.json").resolve(),
            failures=(repo_root / "artifacts/logs/module3_encoder_failures.csv").resolve(),
            pipeline_log=(repo_root / "artifacts/logs/module3_encoder_pipeline.log").resolve(),
            parameter_audit=(
                repo_root / "artifacts/metrics/module3_parameter_audit.json"
            ).resolve(),
        ),
    )


def prepare_tiny_m3_repo(repo_root: Path) -> Module3Config:
    module2, _ = prepare_tiny_repo(repo_root)
    write_config_yaml(module2, repo_root / "configs/representations/default.yaml")
    summary = run_module2(module2, limit=4)
    root_logger = logging.getLogger()
    for handler in module2_pipeline._HANDLERS:
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        handler.close()
    module2_pipeline._HANDLERS.clear()
    manifest = load_manifest_source(module2.source.manifest_path)
    ids = tuple(summary["selection"]["logical_ids"])
    speakers = [logical_id.split("/")[0] for logical_id in ids]
    fold_mapping = {speaker: index % 3 + 1 for index, speaker in enumerate(speakers)}
    _write_json(
        repo_root / "data/splits/speaker_folds.json",
        {
            "schema_version": "1.0",
            "source": {"manifest_sha256": manifest.fingerprint},
            "speaker_to_evaluation_fold": fold_mapping,
        },
    )
    _write_json(
        repo_root / "artifacts/metrics/module2_full_signoff_audit.json",
        {
            "status": "PASS",
            "completion": {
                "expected": 4,
                "selected": 4,
                "successful": 4,
                "failed": 0,
                "validated": 4,
            },
            "identity": {
                "representation_config_hash": module2.config_hash,
                "module1_config_hash": module2.source.module1_config_hash,
                "manifest_fingerprint": manifest.fingerprint,
            },
            "validation": {
                "status": "PASS",
                "selected_record_count": 4,
                "valid_cache_entries": 4,
            },
            "module2_signed_off": True,
            "module3_started": False,
        },
    )
    base = make_model_config(repo_root)
    return Module3Config(
        **{
            **asdict(base),
            "repo_root": base.repo_root,
            "source": SourceConfig(
                manifest_path=module2.source.manifest_path,
                speaker_folds_path=(repo_root / "data/splits/speaker_folds.json").resolve(),
                module2_config_path=(
                    repo_root / "configs/representations/default.yaml"
                ).resolve(),
                module2_signoff_path=(
                    repo_root / "artifacts/metrics/module2_full_signoff_audit.json"
                ).resolve(),
                representation_cache_root=module2.cache.root,
                representation_config_hash=module2.config_hash,
                module1_config_hash=module2.source.module1_config_hash,
                manifest_fingerprint=manifest.fingerprint,
                subset="core",
            ),
            "mel": base.mel,
            "graph": base.graph,
            "recurrence": base.recurrence,
            "outputs": base.outputs,
            "smoke_logical_ids": (*ids, "F03/Session1/0179", "F03/Session1/0118"),
        }
    )


def write_module3_yaml(config: Module3Config, path: Path) -> None:
    root = config.repo_root

    def relative(value: Path) -> str:
        return value.relative_to(root).as_posix()

    raw: dict[str, Any] = {
        "schema_version": config.schema_version,
        "implementation": config.implementation,
        "source": {
            "manifest_path": relative(config.source.manifest_path),
            "speaker_folds_path": relative(config.source.speaker_folds_path),
            "module2_config_path": relative(config.source.module2_config_path),
            "module2_signoff_path": relative(config.source.module2_signoff_path),
            "representation_cache_root": relative(config.source.representation_cache_root),
            "representation_config_hash": config.source.representation_config_hash,
            "module1_config_hash": config.source.module1_config_hash,
            "manifest_fingerprint": config.source.manifest_fingerprint,
            "subset": config.source.subset,
        },
        "common_token_dim": config.token_dimension,
        "mel": {**asdict(config.mel), "adaptive_pool": list(config.mel.adaptive_pool)},
        "graph": asdict(config.graph),
        "recurrence": {
            **asdict(config.recurrence),
            "channels": list(config.recurrence.channels),
            "adaptive_pool": list(config.recurrence.adaptive_pool),
        },
        "regularization": {"dropout": config.dropout},
        "execution": {"smoke_logical_ids": list(config.smoke_logical_ids)},
        "outputs": {
            "summary": relative(config.outputs.summary),
            "failures": relative(config.outputs.failures),
            "pipeline_log": relative(config.outputs.pipeline_log),
            "parameter_audit": relative(config.outputs.parameter_audit),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def make_batch(batch_size: int = 1, node_counts: tuple[int, ...] | None = None) -> EncoderBatch:
    counts = node_counts or (3,) * batch_size
    if len(counts) != batch_size:
        raise ValueError("node count list must match batch size")
    maximum = max(counts)
    nodes = torch.zeros((batch_size, maximum, 2), dtype=torch.float32)
    mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    edges: list[torch.Tensor] = []
    for index, count in enumerate(counts):
        nodes[index, :count, 0] = torch.linspace(0, 1, count)
        nodes[index, :count, 1] = torch.linspace(-1, 1, count)
        mask[index, :count] = True
        if count == 1:
            edges.append(torch.empty((2, 0), dtype=torch.int64))
        else:
            left = torch.arange(count - 1, dtype=torch.int64)
            right = left + 1
            edges.append(torch.cat((torch.stack((left, right)), torch.stack((right, left))), dim=1))
    provenance = tuple(
        SampleProvenance(
            logical_id=f"F01/Session1/{index:04d}",
            speaker_id="F01",
            condition="dysarthric",
            gender="female",
            prompt_type="word",
            primary_microphone="head",
            evaluation_fold=1,
        )
        for index in range(batch_size)
    )
    return EncoderBatch(
        mel=torch.linspace(-120, 30, 224 * 224, dtype=torch.float32)
        .reshape(1, 1, 224, 224)
        .repeat(batch_size, 1, 1, 1),
        recurrence=torch.eye(224, dtype=torch.uint8)
        .reshape(1, 1, 224, 224)
        .repeat(batch_size, 1, 1, 1),
        graph_nodes=nodes,
        graph_node_mask=mask,
        graph_edges=tuple(edges),
        provenance=provenance,
    )


def make_sample(node_count: int = 3) -> EncoderSample:
    return EncoderSample(
        mel=torch.zeros((1, 224, 224), dtype=torch.float32),
        recurrence=torch.eye(224, dtype=torch.uint8).unsqueeze(0),
        graph_nodes=torch.zeros((node_count, 2), dtype=torch.float32),
        graph_edges=make_batch(1, (node_count,)).graph_edges[0],
        provenance=make_batch().provenance[0],
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
