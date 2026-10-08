"""Bind the classifier to signed Modules 0-3 before loading any representations."""

from dataclasses import dataclass

from vaaniswitch.m0_data.leakage import speaker_folds_fingerprint
from vaaniswitch.m0_data.splits import validate_speaker_folds
from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.source import SignedRepresentationSource
from vaaniswitch.m4_fusion.config import Module4Config
from vaaniswitch.m4_fusion.reporting import read_json
from vaaniswitch.m4_fusion.splits import SpeakerSplit, create_inner_split


@dataclass(frozen=True, slots=True)
class DevelopmentContext:
    config: Module4Config
    encoder_config: Module3Config
    source: SignedRepresentationSource
    split: SpeakerSplit


def load_encoder_contract(config: Module4Config) -> Module3Config:
    encoder = Module3Config.from_yaml(config.source.encoder_config_path, repo_root=config.repo_root)
    if encoder.config_hash != config.source.encoder_config_hash:
        raise ValueError("Module 3 encoder configuration hash differs from signed contract")
    for key in ("representation_config_hash", "module1_config_hash", "manifest_fingerprint"):
        if getattr(encoder.source, key) != config.source.hashes[key]:
            raise ValueError(f"Module 3 source has wrong {key}")
    audit = read_json(config.source.module3_signoff_path)
    if audit.get("status") != "PASS" or audit.get("module3_signed_off") is not True:
        raise ValueError("Module 3 must be signed off")
    expected_architecture = {
        "encoder_config_hash": encoder.config_hash,
        "token_dimension": 128,
        "gcn_hidden_dimension": 96,
        "recurrence_channels": [16, 32, 64],
    }
    if any(audit.get("configuration", {}).get(k) != v for k, v in expected_architecture.items()):
        raise ValueError("Module 3 signed architecture mismatch")
    expected_source = {
        "signed_core_records": config.source.core_record_count,
        **{k: v for k, v in config.source.hashes.items() if k != "encoder_config_hash"},
    }
    if any(audit.get("source", {}).get(k) != v for k, v in expected_source.items()):
        raise ValueError("Module 3 signed source mismatch")
    return encoder


def prepare_context(config: Module4Config, outer_test_fold: int) -> DevelopmentContext:
    encoder = load_encoder_contract(config)
    source = SignedRepresentationSource(encoder)
    if source.contract.signed_record_count != config.source.core_record_count:
        raise ValueError("signed CORE count mismatch")
    folds = read_json(encoder.source.speaker_folds_path)
    validate_speaker_folds(source.contract.records, folds)
    split = create_inner_split(
        source.contract,
        outer_test_fold=outer_test_fold,
        seed=config.training.seed,
        folds_fingerprint=speaker_folds_fingerprint(folds),
    )
    return DevelopmentContext(config, encoder, source, split)
