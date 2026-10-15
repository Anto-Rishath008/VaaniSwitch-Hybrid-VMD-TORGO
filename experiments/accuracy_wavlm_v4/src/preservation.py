"""Content/metadata inventory; writes only inside V4, never repairs predecessors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parents[1]
ORIGINAL = STUDY.with_name('VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args],
        env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).decode().strip()


def inventory(base):
    result = {}
    for directory, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != '.git' and
            not (Path(directory) == STUDY/'experiments' and d == ROOT.name))
        for name in sorted(files):
            p = Path(directory)/name
            rel = p.relative_to(base).as_posix()
            if rel == '.git':
                continue
            s = p.stat()
            entry = {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}
            metadata = rel.startswith(('data/raw/', 'data/processed/')) or any(
                part.startswith('.venv') for part in p.relative_to(base).parts)
            if not metadata:
                entry['sha256'] = sha(p)
            result[rel] = entry
    return result


def snapshot():
    result = {}
    for name, base in [('original', ORIGINAL), ('study', STUDY)]:
        print('Inventory: '+name, flush=True)
        result[name] = inventory(base)
        result[name+'_git'] = {'head': git(base, 'rev-parse', 'HEAD'),
            'branch': git(base, 'branch', '--show-current'),
            'refs': git(base, 'show-ref')}
    return result


def run(mode):
    directory = ROOT/'validation'
    directory.mkdir(parents=True, exist_ok=True)
    before_path = directory/'protected_before.json'
    if mode == 'before' and before_path.exists():
        raise ValueError('Baseline already exists; never replace it')
    current = snapshot()
    destination = before_path if mode == 'before' else directory/'protected_after.json'
    destination.write_text(json.dumps(current, sort_keys=True), encoding='utf-8')
    if mode == 'after':
        before = json.loads(before_path.read_text(encoding='utf-8'))
        differences = {group: [key for key in set(before[group]) | set(current[group])
            if before[group].get(key) != current[group].get(key)] for group in before}
        report = {'status': 'FAIL' if any(differences.values()) else 'PASS',
            'changes': differences, 'counts': {g: {'total': len(current[g]),
                'byte_hashed': sum('sha256' in e for e in current[g].values()),
                'metadata_only': sum('sha256' not in e for e in current[g].values())}
                for g in ('original', 'study')},
            'scope': 'Every file except Git internals and V4. Raw/processed and existing venvs: path,size,mtime only. All other files including V1/V2/V3: SHA256,size,mtime. Git HEAD, branch and refs checked.'}
        (directory/'preservation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
        if report['status'] != 'PASS':
            raise ValueError('Protected inventory changed; inspect preservation.json')
    else:
        print({g: len(current[g]) for g in current}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['before', 'after'])
    run(parser.parse_args().mode)
