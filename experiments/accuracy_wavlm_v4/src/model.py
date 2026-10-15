"""Official WavLM with frozen lower modules, compact heads and FP32 masked pooling."""
import torch
from torch import nn
from .common import require
from .numerics import finite
from experiments.accuracy_hubert_v2.src.features import feature_lengths


def frame_mask(hidden, lengths):
    require(hidden.ndim == 3 and lengths.ndim == 1 and len(lengths) == len(hidden), 'Mask dimensions')
    require(lengths.dtype in (torch.int32, torch.int64), 'Mask lengths must be integers')
    require(bool(((lengths > 0) & (lengths <= hidden.shape[1])).all()), 'Invalid valid-frame lengths')
    return torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]


def pool(hidden, lengths, kind='mean'):
    finite(hidden, 'hidden_frames')
    mask = frame_mask(hidden, lengths)
    values = hidden.float()
    mean = values.masked_fill(~mask[..., None], 0).sum(1)/lengths[:, None]
    require(kind in {'mean', 'mean_std'}, 'Unknown pooling')
    if kind == 'mean_std':
        centered = (values-mean[:, None]).masked_fill(~mask[..., None], 0)
        std = (centered.square().sum(1)/lengths[:, None]).clamp_min(0).sqrt()
        mean = torch.cat((mean, std), dim=1)
    return finite(mean, 'pooled_output')


class AttentionPool(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dimension, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))

    def forward(self, hidden, lengths):
        finite(hidden, 'attention_hidden')
        mask = frame_mask(hidden, lengths)
        scores = finite(self.score(hidden.float()).squeeze(-1), 'attention_scores')
        weights = scores.masked_fill(~mask, float('-inf')).softmax(1)
        finite(weights, 'attention_weights')
        return finite((hidden.float()*weights[..., None]).sum(1), 'attention_pool')


def load_backbone(directory):
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    backbone, info = WavLMModel.from_pretrained(directory, local_files_only=True,
        output_loading_info=True)
    require(not info['missing_keys'] and not info['mismatched_keys'] and not info['error_msgs'],
        'Incomplete pretrained WavLM loading: '+str(info))
    # Official SSL checkpoint may contain pretraining-only heads, never a dysarthria classifier.
    allowed = ('quantizer.', 'project_hid.', 'project_q.')
    require(all(k.startswith(allowed) for k in info['unexpected_keys']), 'Unexpected pretrained keys')
    require(backbone.config.hidden_size == 768 and len(backbone.encoder.layers) == 12, 'Wrong WavLM architecture')
    processor = Wav2Vec2FeatureExtractor.from_pretrained(directory, local_files_only=True)
    require(processor.sampling_rate == 16000 and processor.do_normalize is False
        and processor.return_attention_mask is True, 'Unexpected official WavLM processor')
    backbone.requires_grad_(False)
    backbone.feature_extractor._freeze_parameters()
    return backbone, processor, info


def inputs(processor, waveform, device):
    finite(waveform, 'waveform')
    require(len(waveform) >= 400, 'No valid WavLM convolution frame')
    # Batch one preserves GroupNorm semantics; no variable-length padding enters the encoder.
    prepared = processor(waveform, sampling_rate=16000, return_tensors='pt',
        padding=False, return_attention_mask=True)
    values = finite(prepared.input_values.to(device), 'normalized_waveform')
    mask = prepared.attention_mask.to(device)
    require(mask.shape == values.shape and bool((mask == 1).all()), 'Unexpected input padding/mask')
    return values, mask


class Classifier(nn.Module):
    def __init__(self, backbone, depth, pooling, checkpointing=True):
        super().__init__()
        require(depth in (2, 4) and pooling in ('mean', 'attention'), 'Unapproved fine-tuning configuration')
        self.backbone, self.depth, self.pooling = backbone, depth, pooling
        backbone.requires_grad_(False)
        backbone.feature_extractor._freeze_parameters()
        backbone.config.layerdrop = 0.
        backbone.config.apply_spec_augment = False
        for layer in backbone.encoder.layers[-depth:]:
            layer.requires_grad_(True)
        if checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        dimension = backbone.config.hidden_size
        self.attention = AttentionPool(dimension) if pooling == 'attention' else None
        self.head = nn.Sequential(nn.LayerNorm(dimension), nn.Dropout(.1), nn.Linear(dimension, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        # Frozen frontend, positional convolution and lower blocks have dropout disabled.
        for layer in self.backbone.encoder.layers[-self.depth:]:
            layer.train(mode)
        return self

    def forward(self, values, mask):
        finite(values, 'model_input')
        require(values.ndim == 2 and values.shape == mask.shape and bool(((mask == 0)|(mask == 1)).all()), 'Invalid input mask')
        lengths = feature_lengths(mask.sum(1), self.backbone.config.conv_kernel, self.backbone.config.conv_stride)
        hidden = self.backbone(values, attention_mask=mask).last_hidden_state
        pooled = self.attention(hidden, lengths) if self.attention is not None else pool(hidden, lengths)
        return finite(self.head(pooled).squeeze(-1), 'logits')

    def parameter_report(self):
        return {'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'frozen': sum(p.numel() for p in self.parameters() if not p.requires_grad),
            'trainable_names': [n for n, p in self.named_parameters() if p.requires_grad],
            'upper_layers_zero_based': list(range(len(self.backbone.encoder.layers)-self.depth, len(self.backbone.encoder.layers)))}
