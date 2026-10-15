"""Compact E2 and fresh-in-every-fold B2/HuBERT E3 classifiers."""
import torch
from torch import nn

from .native_b2_contract import RecurrenceEncoderConfig
from .native_b2_encoder import RecurrenceTokenEncoder


class Classifier(nn.Module):
    def __init__(self, model_name, dropout, feature_dimension=1536):
        super().__init__()
        if model_name not in {"E2", "E3"}:
            raise ValueError("Neural classifier must be E2 or E3")
        self.model_name = model_name
        self.speech = nn.Sequential(nn.Linear(feature_dimension, 128), nn.LayerNorm(128),
                                    nn.GELU(), nn.Dropout(dropout))
        self.rp = (RecurrenceTokenEncoder(RecurrenceEncoderConfig(), token_dimension=128,
                                         dropout=0.2) if model_name == "E3" else None)
        self.head = nn.Sequential(nn.Linear(256 if self.rp else 128, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, speech, rp=None):
        projected = self.speech(speech)
        if self.rp is not None:
            if rp is None:
                raise ValueError("E3 requires the unchanged binary RP")
            projected = torch.cat((self.rp(rp).mean(dim=1), projected), dim=1)
        return self.head(projected).squeeze(-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
