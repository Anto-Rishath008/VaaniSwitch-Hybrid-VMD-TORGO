"""Only requested branches are instantiated; signed encoders and C2 attention are reused."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.graph_encoder import VisibilityGraphTokenEncoder
from vaaniswitch.m3_encoders.mel_encoder import MelTokenEncoder
from vaaniswitch.m3_encoders.recurrence_encoder import RecurrenceTokenEncoder
from vaaniswitch.m3_encoders.schemas import EncoderBatch
from vaaniswitch.m4_fusion.model import TokenFusionClassifier
from vaaniswitch.m5_experiments.config import ABLATIONS, Candidate


@dataclass(frozen=True)
class AblationOutput:
    logits: torch.Tensor
    tokens: dict[str, torch.Tensor]
    pooled: torch.Tensor

    @property
    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)


class AblationModel(nn.Module):
    def __init__(
        self, config: Module3Config, ablation: str, candidate: Candidate, *, load_pretrained: bool
    ) -> None:
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError("C3 is design-only; unknown/disabled ablation")
        self.ablation = ablation
        self.branches = nn.ModuleDict()
        # Separate fixed initialization streams preserve shared initial weights across ablations.
        for name, seed in (("mel", 42), ("graph", 43), ("recurrence", 44)):
            needed = {
                "mel": ablation in {"A0", "A1", "C1", "C2"},
                "graph": ablation in {"B1", "B3", "C1", "C2"},
                "recurrence": ablation in {"B2", "B3", "C1", "C2"},
            }[name]
            if not needed:
                continue
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                if name == "mel":
                    branch: nn.Module = MelTokenEncoder(
                        config.mel, token_dimension=128, load_pretrained=load_pretrained
                    )
                elif name == "graph":
                    branch = VisibilityGraphTokenEncoder(
                        config.graph, token_dimension=128, dropout=config.dropout
                    )
                else:
                    branch = RecurrenceTokenEncoder(
                        config.recurrence, token_dimension=128, dropout=config.dropout
                    )
                self.branches[name] = branch
        width = 256 if ablation in {"B3", "C1", "C2"} else 128
        self.fusion: TokenFusionClassifier | None = None
        self.classifier: nn.Sequential | None = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(45)
            if ablation == "C2":
                self.fusion = TokenFusionClassifier()
                self.fusion.classifier[2] = nn.Dropout(candidate.classifier_dropout)
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(width, 128),
                    nn.GELU(),
                    nn.Dropout(candidate.classifier_dropout),
                    nn.Linear(128, 1),
                )

    def verify_frozen_backbone(self) -> None:
        if "mel" in self.branches:
            mel = cast(MelTokenEncoder, self.branches["mel"])
            if (
                not mel.backbone_frozen
                or mel.backbone.training
                or any(p.requires_grad or p.grad is not None for p in mel.backbone.parameters())
            ):
                raise ValueError("Module 5 requires a frozen ImageNet backbone")

    def forward(self, batch: EncoderBatch) -> AblationOutput:
        self.verify_frozen_backbone()
        tokens: dict[str, torch.Tensor] = {}
        for name, branch in self.branches.items():
            tokens[name] = (
                branch(batch.graph_nodes, batch.graph_edges, batch.graph_node_mask)
                if name == "graph"
                else branch(getattr(batch, name))
            )
            expected_count = 1 if name == "graph" else 16
            if tokens[name].shape != (len(batch.provenance), expected_count, 128) or not (
                torch.isfinite(tokens[name]).all()
            ):
                raise ValueError("non-finite or malformed branch tokens")
        if self.ablation in {"C1", "C2"}:
            nonlinear = torch.cat((tokens["graph"], tokens["recurrence"]), dim=1)
            if self.fusion is not None:
                fused = self.fusion(tokens["mel"], nonlinear)
                return AblationOutput(fused.logits, tokens, fused.fusion_vector)
            pooled = torch.cat((tokens["mel"].mean(1), nonlinear.mean(1)), dim=-1)
        else:
            pooled = torch.cat([value.mean(1) for value in tokens.values()], dim=-1)
        assert self.classifier is not None
        logits = cast(torch.Tensor, self.classifier(pooled)).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite ablation logits")
        return AblationOutput(logits, tokens, pooled)

    def parameter_counts(self) -> dict[str, int]:
        head = self.fusion.classifier if self.fusion is not None else self.classifier
        assert head is not None
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "classifier": sum(p.numel() for p in head.parameters()),
        }
