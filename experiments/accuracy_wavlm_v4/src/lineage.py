"""Pin inherited implementation bytes, not historical predictions for selection."""
from .common import ROOT, STUDY, publish, read, require, sha


def prepare():
    destination = ROOT/'V3_LINEAGE.json'
    if destination.exists():
        return verify()
    files = {}
    for name in ('accuracy_hubert_v2', 'accuracy_hubert_v3'):
        base = STUDY/'experiments'/name
        for p in sorted((base/'src').glob('*.py')):
            files[p.relative_to(STUDY).as_posix()] = sha(p)
        for p in sorted(base.glob('*PROTOCOL.md')):
            files[p.relative_to(STUDY).as_posix()] = sha(p)
    doc = {'files': files, 'release_commit': '2dee81f7656e5b5cae800d9a3c8692e5e8107bf9',
        'behavior': 'Read-only inherited Corpus/Record/primary microphone and outer split contracts; V3 three-fold speaker CV assignment, metrics and prediction row definitions; V2 RNG and atomic utilities. No inherited launcher or writer runs.',
        'results_policy': 'Historical results excluded from selection APIs. Later report-only comparison after all V4 outer seals.',
        'historical_exposure': 'Original TORGO outer results already inspected; V4 comparisons are POST-HOC EXPLORATORY.'}
    publish(destination, doc)
    return doc


def verify():
    doc = read(ROOT/'V3_LINEAGE.json')
    for rel, expected in doc['files'].items():
        require(sha(STUDY/rel) == expected, 'Inherited source changed: '+rel)
    return doc
