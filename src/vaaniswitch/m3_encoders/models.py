"""Combined Module 3 encoders without fusion or classification."""

from __future__ import annotations

from torch import nn

from vaaniswitch.m3_encoders.config import Module3Config
from vaaniswitch.m3_encoders.graph_encoder import VisibilityGraphTokenEncoder
from vaaniswitch.m3_encoders.mel_encoder import MelTokenEncoder
from vaaniswitch.m3_encoders.recurrence_encoder import RecurrenceTokenEncoder
from vaaniswitch.m3_encoders.schemas import EncoderBatch, EncoderTokenOutput


class ParallelTokenEncoders(nn.Module):
    def __init__(self, config: Module3Config, *, load_pretrained: bool) -> None:
        super().__init__()
        self.token_dimension = config.token_dimension
        self.mel_encoder = MelTokenEncoder(
            config.mel,
            token_dimension=config.token_dimension,
            load_pretrained=load_pretrained,
        )
        self.graph_encoder = VisibilityGraphTokenEncoder(
            config.graph,
            token_dimension=config.token_dimension,
            dropout=config.dropout,
        )
        self.recurrence_encoder = RecurrenceTokenEncoder(
            config.recurrence,
            token_dimension=config.token_dimension,
            dropout=config.dropout,
        )

    def forward(self, batch: EncoderBatch) -> EncoderTokenOutput:
        output = EncoderTokenOutput(
            mel_tokens=self.mel_encoder(batch.mel),
            graph_token=self.graph_encoder(
                batch.graph_nodes, batch.graph_edges, batch.graph_node_mask
            ),
            recurrence_tokens=self.recurrence_encoder(batch.recurrence),
        )
        output.validate(self.token_dimension)
        return output
