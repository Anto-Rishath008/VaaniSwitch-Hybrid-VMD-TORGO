"""V3 identities and publication. Imported V2 utilities are read-only dependencies."""
from pathlib import Path
import importlib.metadata
import os
import sys

sys.dont_write_bytecode = True
from experiments.accuracy_hubert_v2.src.common import (
    BASELINE, ORIGINAL, STUDY, ProcessLock, atomic_bytes, digest, git, progress,
    read, require, sha, write_json, preservation_snapshot)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.with_name('accuracy_hubert_v2')
V1 = ROOT.with_name('accuracy_hubert')
TITLE = 'V3 POST-HOC EXPLORATORY STUDY'


def publish(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, f'Frozen artifact differs: {path}')
    else:
        write_json(path, value)
    return path


def within_v3(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Output must stay inside V3')
    return path


def config():
    import yaml
    cfg = yaml.safe_load((ROOT/'configs/study.yaml').read_text())
    require(cfg['threshold'] == .5 and cfg['seed'] == 42, 'Frozen threshold/seed changed')
    require(cfg['layers'] == [6, 9, 12] and cfg['C'] == [.01, .1, 1.], 'Candidate grid changed')
    require(cfg['augmentation_per_speaker'] == 16 and cfg['cv_folds'] == 3, 'Budget changed')
    require(cfg['logistic_max_iter'] == 3000 and cfg['cpu_threads'] == 2, 'LR policy changed')
    require(cfg['vram_fraction'] <= .8 and cfg['calibration'] in (True, False), 'Resource policy')
    return cfg


def source_identity():
    paths = [ROOT/'configs/study.yaml', ROOT/'run_all.ps1', ROOT/'V2_LINEAGE.json']
    paths += [ROOT/n for n in ('EXPERIMENT_PROTOCOL.md', 'MODEL_SELECTION_PROTOCOL.md',
                              'AUGMENTATION_PROTOCOL.md')]
    paths += sorted((ROOT/'src').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def versions():
    return {p: importlib.metadata.version(p) for p in
            ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers', 'soundfile')}


def snapshot():
    result = preservation_snapshot()
    for version in ('accuracy_hubert', 'accuracy_hubert_v2'):
        base = ROOT.with_name(version)
        result[version] = {p.relative_to(base).as_posix():
                          {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha(p)}
                          for p in sorted(base.rglob('*')) if p.is_file()}
    return result


def verify_preservation():
    before = read(ROOT/'validation/protected_before.json')
    after = snapshot()
    write_json(ROOT/'validation/protected_after.json', after, replace=True)
    changes = {}
    for group in before:
        changed = [k for k in set(before[group]) | set(after[group])
                   if before[group].get(k) != after[group].get(k)]
        if changed:
            changes[group] = changed
    report = {'status': 'FAIL' if changes else 'PASS', 'changes': changes,
              'counts': {k: len(v) for k, v in after.items()},
              'original_hashed_files': sum('sha256' in x for x in after['original'].values()),
              'corpus_scope': 'Raw/processed original corpus: path/size/mtime only; primary audio hashes separately checked',
              'predecessor_scope': 'All V1/V2 files: SHA256 + size + mtime, including existing bytecode'}
    write_json(ROOT/'validation/preservation.json', report, replace=True)
    require(not changes, 'Protected data differ; STOP, see validation/preservation.json')
    return report


def environment(cfg):
    import torch
    import shutil
    from experiments.accuracy_hubert_v2.src.training import seed_all
    require(Path(sys.prefix).resolve() == (STUDY/'.venv-accuracy').resolve(), 'Use existing .venv-accuracy')
    require(torch.cuda.is_available() and '4070 Laptop' in torch.cuda.get_device_name(0), 'Expected RTX 4070 Laptop CUDA')
    require(git(ORIGINAL, 'rev-parse', 'HEAD') == git(STUDY, 'rev-parse', 'HEAD') == BASELINE, 'Protected HEAD changed')
    require(shutil.disk_usage(ROOT).free > 2 * 1024**3, 'Need 2 GiB free for bounded V3 outputs')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(cfg['vram_fraction'])
    seed_all()
    return {'versions': versions(), 'python': sys.version, 'executable': sys.executable,
            'gpu': torch.cuda.get_device_name(0), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'free_disk_bytes': shutil.disk_usage(ROOT).free, 'cuda': torch.version.cuda}
