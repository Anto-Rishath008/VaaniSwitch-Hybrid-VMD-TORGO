"""V4-only publication, identities, dependencies and immutable checkpoint files."""
import importlib.metadata
import io
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
from experiments.accuracy_hubert_v2.src.common import (
    atomic_bytes as inherited_atomic, digest, read, require, sha, ProcessLock)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parents[1]
TITLE = 'V4 POST-HOC EXPLORATORY STUDY'


def inside(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Writes must stay inside V4')
    return path


def atomic(path, value, replace=False):
    inherited_atomic(inside(path), value, replace=replace)


def write(path, value, replace=False):
    import json
    atomic(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode(), replace)


def publish(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Sealed artifact differs: '+str(path))
    else:
        write(path, value)
    return Path(path)


def versions():
    return {p: importlib.metadata.version(p) for p in
        ('torch', 'transformers', 'huggingface-hub', 'numpy', 'scipy', 'scikit-learn', 'soundfile', 'matplotlib')}


def config():
    cfg = read(ROOT/'configs/study.json')
    require(cfg['seed'] == 42 and cfg['threshold'] == .5, 'Seed/threshold policy')
    require(cfg['layers'] == [9, 12] and cfg['C'] == [.01, .1], 'Candidate budget')
    require(cfg['precision'] == 'fp32' and cfg['physical_batch_size'] == 1, 'Verified numerical policy')
    require(cfg['attention_backend'] == 'math' and cfg['deterministic_algorithms'] == 'strict', 'Deterministic attention policy')
    require(cfg['fine_tune_seconds'] == 4 and cfg['frozen_seconds'] == 8, 'Frozen segment policy')
    return cfg


def sources():
    paths = list((ROOT/'src').glob('*.py')) + [ROOT/p for p in
        ('run_all.ps1', 'configs/study.json', 'V3_LINEAGE.json', 'EXPERIMENT_PROTOCOL.md', 'MODEL_SELECTION_PROTOCOL.md')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def event(root, kind, **details):
    write(Path(root)/'events'/f'{time.time_ns()}-{uuid.uuid4().hex}.json', {'kind': kind, **details})


def checkpoint(root, payload):
    import torch
    path = inside(Path(root)/f"state-{payload.get('step', 0):07d}-{uuid.uuid4().hex}.pt")
    stream = io.BytesIO()
    torch.save(payload, stream)
    atomic(path, stream.getvalue())
    reference = {'path': str(path), 'sha256': sha(path), 'identity': payload['identity']}
    write(Path(root)/'latest.json', reference, replace=True)
    return reference


def load_checkpoint(reference, identity):
    import torch
    path = inside(reference['path'])
    require(reference['identity'] == identity and sha(path) == reference['sha256'], 'Checkpoint identity/checksum')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(payload['identity'] == identity, 'Checkpoint payload identity')
    return payload


def environment():
    import torch
    from .numerics import configure_numerics
    require(Path(sys.prefix).resolve() == (STUDY/'.venv-accuracy').resolve(), 'Use .venv-accuracy')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    require('4070 Laptop' in torch.cuda.get_device_name(0), 'Unexpected GPU')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    configure_numerics(42)
    return {'versions': versions(), 'python': sys.version, 'executable': sys.executable,
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'total_vram': torch.cuda.get_device_properties(0).total_memory, 'precision': 'fp32',
        'attention_backend': 'math', 'deterministic_algorithms': 'strict'}
