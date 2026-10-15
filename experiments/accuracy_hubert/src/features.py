"""Pinned frozen HuBERT CUDA extraction and immutable per-record feature cache."""
from __future__ import annotations

import gc
import io
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .common import ROOT, digest, progress, read, require, sha, write_json, atomic_bytes
from .data import safe_original


def acquire(cfg):
    from huggingface_hub import snapshot_download
    spec = cfg["features"]
    directory = ROOT / "cache/checkpoint" / spec["revision"]
    lock = directory / "verified_model.json"
    expected_files = ["config.json", "preprocessor_config.json", spec["weight_file"]]
    if not lock.exists():
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        snapshot_download(spec["model"], revision=spec["revision"], local_dir=directory,
                          allow_patterns=expected_files, max_workers=2)
        require(sha(directory / spec["weight_file"]) == spec["weight_sha256"],
                "Downloaded official model weight checksum mismatch")
        doc = {"model": spec["model"], "revision": spec["revision"],
               "files": {f: sha(directory / f) for f in expected_files}}
        write_json(lock, doc)
    doc = read(lock)
    require(doc["model"] == spec["model"] and doc["revision"] == spec["revision"],
            "Checkpoint identity changed")
    for filename in expected_files:
        require(sha(directory / filename) == doc["files"][filename], "Checkpoint file changed")
    require(doc["files"][spec["weight_file"]] == spec["weight_sha256"], "Official weight pin changed")
    require(doc["files"]["config.json"] == spec["config_sha256"]
            and doc["files"]["preprocessor_config.json"] == spec["preprocessor_sha256"],
            "Pinned checkpoint configuration checksum mismatch")
    return directory, doc


def audio(record):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    path = safe_original(record.primary_audio_path)
    require(sha(path) == record.primary_audio_sha256, f"Source SHA changed: {record.logical_id}")
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    require(len(values) == record.frame_count and rate == record.sample_rate, "Audio header changed")
    values = values.mean(axis=1, dtype=np.float32)
    if rate != 16000:
        divisor = gcd(rate, 16000)
        values = resample_poly(values, 16000 // divisor, rate // divisor).astype(np.float32)
    require(len(values) >= 400 and np.isfinite(values).all(), "Audio too short/non-finite")
    start = max(0, (len(values) - 128000) // 2)
    end = min(len(values), start + 128000)
    segment = values[start:end].copy()
    return segment, {"source_samples_16k": len(values), "start_sample_16k": start,
                     "end_sample_16k": end, "cropped": len(values) > 128000,
                     "valid_samples": len(segment), "right_padding_samples": 0,
                     "source_rate": rate, "target_rate": 16000}


def feature_lengths(lengths, kernels, strides):
    result = lengths.clone()
    for kernel, stride in zip(kernels, strides, strict=True):
        result = torch.div(result - kernel, stride, rounding_mode="floor") + 1
    require(bool((result > 0).all()), "No valid convolution frames")
    return result


def masked_pool(hidden, lengths):
    require(hidden.ndim == 3 and len(lengths) == len(hidden), "Pooling shape mismatch")
    require(bool(((lengths > 0) & (lengths <= hidden.shape[1])).all()), "Invalid frame lengths")
    values = hidden.float()
    mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
    masked = values.masked_fill(~mask[:, :, None], 0)
    count = lengths[:, None].float()
    mean = masked.sum(1) / count
    deviation = (values - mean[:, None, :]).masked_fill(~mask[:, :, None], 0)
    variance = deviation.square().sum(1) / count
    return torch.cat((mean, variance.clamp_min(0).sqrt()), dim=1)


class FeatureCache:
    def __init__(self, root, identity, layers=(6, 9, 12)):
        self.identity, self.layers = identity, tuple(layers)
        self.root = root / digest(identity)

    def location(self, record):
        require(len(record.logical_id.split("/")) == 3 and ".." not in record.logical_id.split("/"),
                "Unsafe logical ID")
        return self.root / record.logical_id

    def expected(self, record):
        return {"logical_id": record.logical_id, "speaker_id": record.speaker_id,
                "source_identity": {"path": record.primary_audio_path,
                                    "sha256": record.primary_audio_sha256},
                "extraction_identity": self.identity}

    def load(self, record):
        directory = self.location(record)
        metadata = directory / "complete.json"
        if not metadata.exists():
            return None
        doc = read(metadata)
        require(doc["status"] == "COMPLETE" and doc["identity"] == self.expected(record),
                "Completed feature cache identity mismatch; explicit new study required")
        require(PathName(doc["filename"]) == doc["filename"], "Unsafe feature filename")
        path = directory / doc["filename"]
        require(sha(path) == doc["feature_checksum"], "Completed feature cache corruption")
        values = np.load(path, allow_pickle=False)
        require(values.dtype == np.float32 and values.shape == (len(self.layers), 1536)
                and np.isfinite(values).all(), "Feature dtype/shape/finiteness mismatch")
        require(doc["shape"] == list(values.shape) and doc["layers"] == list(self.layers),
                "Feature layer metadata mismatch")
        for index, layer in enumerate(self.layers):
            require(doc["entries"][index] == {"layer_identifier": f"hidden_states[{layer}]",
                    "shape": [1536], "checksum": digest(values[index].tolist()),
                    "completion_status": "COMPLETE"}, "Per-layer cache binding mismatch")
        return values

    def save(self, record, values, segment):
        previous = self.load(record)
        if previous is not None:
            require(np.array_equal(previous, values), "Refusing completed feature replacement")
            return
        values = np.asarray(values, dtype=np.float32)
        require(values.shape == (len(self.layers), 1536) and np.isfinite(values).all(), "Bad feature output")
        directory = self.location(record)
        filename = f"features-{uuid.uuid4().hex}.npy"
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        atomic_bytes(directory / filename, stream.getvalue())
        write_json(directory / "complete.json", {"status": "COMPLETE", "identity": self.expected(record),
                   "filename": filename, "feature_checksum": sha(directory / filename),
                   "shape": list(values.shape), "layers": list(self.layers), "segment": segment,
                   "entries": [{"layer_identifier": f"hidden_states[{layer}]", "shape": [1536],
                                "checksum": digest(values[index].tolist()), "completion_status": "COMPLETE"}
                               for index, layer in enumerate(self.layers)]})


def PathName(name):
    from pathlib import Path
    return Path(name).name


class Extractor:
    def __init__(self, cfg, checkpoint):
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        require(torch.cuda.is_available(), "CUDA is required")
        self.cfg = cfg
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
        self.model = HubertModel.from_pretrained(checkpoint, local_files_only=True,
                                                attn_implementation="sdpa").cuda().eval()
        self.model.requires_grad_(False)
        require(self.model.config.hidden_size == 768 and self.model.config.num_hidden_layers == 12,
                "Unexpected official HuBERT dimensions")
        require(self.model.config.feat_extract_norm == "group", "Reassess padding for different checkpoint")
        require(self.processor.sampling_rate == 16000 and self.processor.do_normalize,
                "Wrong checkpoint preprocessing")
        self.batch_size = cfg["features"]["extraction_batch_cap"]

    @torch.inference_mode()
    def infer(self, waves):
        # GroupNorm normalizes over time. Never add padding to mixed-length utterances.
        require(waves and len({len(w) for w in waves}) == 1, "GroupNorm requires exact-length batches")
        inputs = self.processor(waves, sampling_rate=16000, padding=False,
                                return_attention_mask=True, return_tensors="pt")
        sample_mask = inputs.attention_mask.cuda(non_blocking=True)
        require(bool(sample_mask.bool().all()), "Unexpected padded checkpoint input")
        values = inputs.input_values.pin_memory().cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            # Per checkpoint documentation: group-norm pretraining did not use attention_mask.
            states = self.model(values, output_hidden_states=True).hidden_states
        lengths = feature_lengths(sample_mask.sum(1), self.model.config.conv_kernel,
                                  self.model.config.conv_stride)
        output = torch.stack([masked_pool(states[layer], lengths)
                              for layer in self.cfg["features"]["layers"]], dim=1)
        require(bool(torch.isfinite(output).all()), "Non-finite HuBERT features")
        return output.cpu().numpy().astype(np.float32)

    def benchmark(self, waves):
        """Worst-duration dev segment, duplicated only in this benchmark namespace."""
        wave = max(waves, key=len)
        self.infer([wave])  # warm-up
        timings = []
        for size in (1, 2, 4):
            if size > self.cfg["features"]["extraction_batch_cap"]:
                break
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                self.infer([wave] * size)
                torch.cuda.synchronize()
                timings.append({"batch_size": size, "segment_samples": len(wave),
                                "seconds": time.perf_counter()-started,
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                break
        require(timings, "No CUDA extraction batch succeeded")
        # Choose highest throughput within 80% physical VRAM; adaptive OOM further halves.
        memory = torch.cuda.get_device_properties(0).total_memory
        safe = [t for t in timings if t["peak_reserved_bytes"] <= memory * 0.8]
        require(safe, "Extraction exceeds declared VRAM budget")
        self.batch_size = min(2, max(safe, key=lambda t: t["batch_size"]/t["seconds"])["batch_size"])
        return {"timings": timings, "selected_batch_size": self.batch_size,
                "policy": "exact-length groups; conservative cap two after benchmark"}

    def extract(self, records, cache):
        remaining = [r for r in records if cache.load(r) is None]
        started = time.perf_counter()
        completed = len(records)-len(remaining)
        progress("HuBERT cached recordings", completed, len(records))
        # Bounded lookahead, decode each remaining source once. Exact-duration grouping.
        with ThreadPoolExecutor(max_workers=self.cfg["features"]["decode_workers"]) as pool:
            lookahead = self.cfg["features"]["decode_lookahead"]
            for start in range(0, len(remaining), lookahead):
                selected = remaining[start:start+lookahead]
                decoded = list(pool.map(audio, selected))
                groups = defaultdict(list)
                for record, (wave, segment) in zip(selected, decoded, strict=True):
                    groups[len(wave)].append((record, wave, segment))
                for group in groups.values():
                    position = 0
                    while position < len(group):
                        batch = group[position:position+self.batch_size]
                        try:
                            output = self.infer([row[1] for row in batch])
                        except torch.cuda.OutOfMemoryError:
                            if self.batch_size == 1:
                                raise
                            self.batch_size = max(1, self.batch_size // 2)
                            gc.collect()
                            torch.cuda.empty_cache()
                            progress(f"Extraction OOM; reducing batch to {self.batch_size}")
                            continue
                        for (record, _, segment), features in zip(batch, output, strict=True):
                            cache.save(record, features, segment)
                        position += len(batch)
                        completed += len(batch)
                progress("HuBERT extraction", completed, len(records), started)
        return {"record_count": len(records), "newly_extracted": len(remaining),
                "cache_hits": len(records)-len(remaining), "seconds": time.perf_counter()-started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "frozen_backbone_parameter_count": sum(p.numel() for p in self.model.parameters()),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def extraction_identity(cfg, model_lock):
    import transformers
    return {"config": cfg["features"], "model_lock": model_lock,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "implementation": "frozen_hubert_exact_length_v1"}


def validate_cache(records, cache):
    bindings = {}
    crops = 0
    for record in records:
        require(cache.load(record) is not None, "Missing completed feature cache")
        metadata = cache.location(record) / "complete.json"
        bindings[record.logical_id] = sha(metadata)
        crops += int(read(metadata)["segment"]["cropped"])
    return {"status": "PASS", "record_count": len(records), "cropped_recordings": crops,
            "bindings": bindings, "cache_identity": cache.identity}
