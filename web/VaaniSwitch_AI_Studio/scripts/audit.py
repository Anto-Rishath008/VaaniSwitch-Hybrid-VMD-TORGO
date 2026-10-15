"""Read-only discovery, sealed report verification and before/after preservation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, hashlib, os, subprocess
from backend.settings import ROOT, ORIGINAL, STUDY, REGISTRY

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

def inventory(root):
    out = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in files:
            p = Path(directory)/name
            st = p.stat()
            out[p.relative_to(root).as_posix()] = [st.st_size, st.st_mtime_ns]
    return out

def seal():
    p = STUDY/'experiments/accuracy_wavlm_v4/results/REPORT_SEAL.json'
    d = read(p)
    checked = {str(Path(k).relative_to(STUDY)): sha(k) == v for k, v in d['files'].items()}
    if not all(checked.values()): raise ValueError('V4 sealed evidence changed')
    return {'status': 'PASS', 'files': checked, 'seal_sha256': sha(p), 'identity': d['identity']}

def discover():
    v4 = STUDY/'experiments/accuracy_wavlm_v4'
    v3 = STUDY/'experiments/accuracy_hubert_v3'
    v2 = STUDY/'experiments/accuracy_hubert_v2'
    d4, d3, d2 = [read(p/'results/final_results.json') for p in (v4,v3,v2)]
    base = [
        ('c2','Hybrid CNN–GCN Cross-Attention','Original','Shared VMD → 16 Mel + 17 nonlinear tokens → cross-attention',d4['baselines']['Original C2']),
        ('b2','Recurrence CNN Baseline','Original','Shared VMD → recurrence plot → CNN',d4['baselines']['Original B2']),
        ('hubert-v2','HuBERT Feature Classifier','HuBERT V2','Frozen HuBERT → mean + population std → logistic regression',d4['baselines']['V2 E1']),
        ('hubert-v3','Improved HuBERT Classifier','HuBERT V3','Frozen layer 9 → context-specific normalization / weighting → logistic regression',d4['baselines']['V3 Selected']),
        ('wavlm-selected','WavLM Adaptive Fine-Tuning','WavLM V4','Last 2 / 4 layers → mean / attention pooling → linear head',d4['models']['selected']['fold_aggregation']),
        ('wavlm-reference','WavLM Frozen Reference','WavLM V4','Frozen layer 12 → mean + population std → logistic regression C=0.01',d4['models']['frozen_reference']['fold_aggregation']),
    ]
    models = []
    for ident, name, family, architecture, metrics in base:
        entry = dict(id=ident, display_name=name, family=family, architecture=architecture,
            checkpoint_reference={}, source_project='original' if family=='Original' else 'study',
            input_contract={'sample_rate':16000,'channels':1,'max_source_seconds':60},
            output_contract={'labels':{'0':'Control','1':'Dysarthric'},'threshold':.5,'score':'uncalibrated model probability'},
            fold_context=[1,2,3], availability='research_only', research_role='historical evaluated model',
            evaluation_metrics={k:metrics[k] for k in ('accuracy','macro_f1')}, limitations=[
                'Exploratory TORGO evaluation; no independent clinical validation.',
                'Fold-context weights are not a universal deployed classifier.',
                'External audio is outside the validated evaluation distribution.'],
            inference_adapter=ident, integrity_status='verified local assets', runtime_validated=False)
        if ident in ('b2','c2'):
            entry.update(availability='historical_only', inference_adapter=None,
                blocker='Frozen corpus-bound VMD/representation normalization and model adapters have not been validated for arbitrary uploads.')
            entry['preprocessing'] = 'Shared all-mode VMD; frozen representation settings and train-only normalization.'
            entry['checkpoint_reference'] = [str(p) for p in ORIGINAL.glob(f'artifacts/module5/**/fold*_{ident.upper()}*/final/*.pt')]
        elif ident=='hubert-v2':
            entry['checkpoint_reference'] = {str(i): d2['frozen_decisions']['selections'][f'fold{i}_E1']['final_checkpoint'] for i in (1,2,3)}
            entry['configurations'] = {str(i): {'layer':d2['frozen_decisions']['selections'][f'fold{i}_E1']['layer']} for i in (1,2,3)}
        elif ident=='hubert-v3':
            entry['checkpoint_reference'] = {str(f['outer_fold']):f['identity']['checkpoint'] for f in d3['models']['selected']['folds']}
            entry['configurations'] = read(v3/'results/selected_configurations.json')
            entry['speaker_accuracy'] = d3['models']['selected']['speaker_descriptive']['metrics']['accuracy']
        else:
            role = 'selected' if ident.endswith('selected') else 'frozen_reference'
            entry['checkpoint_reference'] = {str(f['fold']):f['identity']['fit'] for f in d4['models'][role]['folds']}
            if role=='selected':
                entry['outer_fit_reference'] = dict(entry['checkpoint_reference'])
                entry['checkpoint_reference'] = {str(i):read(v4/f'experiments/context{i}/final/selected/complete.json')['best_checkpoint'] for i in (1,2,3)}
            entry['configurations'] = {str(f['fold']):f['candidate'] for f in d4['models'][role]['folds']}
            entry['speaker_accuracy'] = d4['models'][role]['speaker_descriptive']['metrics']['accuracy']
            entry['research_role'] = 'Development-Selected V4' if role=='selected' else 'Best Observed Reference'
        if ident not in ('b2','c2'):
            seconds = 4 if ident=='wavlm-selected' else 8
            entry['input_contract']['segment_seconds'] = seconds
            entry['preprocessing'] = f'Mono arithmetic mean; polyphase resample to 16 kHz; center crop at {seconds}s; no VMD; no right padding.'
        models.append(entry)
    models += [dict(id='asr', display_name='Whisper small.en', family='Frozen utility', availability='available',
        research_role='English recognition', checkpoint_reference=str(ORIGINAL/'artifacts/module6/models/whisper/small.en.pt'),
        expected_sha256='f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872',
        input_contract={'sample_rate':16000,'max_source_seconds':60}, preprocessing='Mono, 16 kHz, unchanged amplitude; frozen Module 6 decoding.',
        limitations=['Batch recognition; review text before synthesis.'], integrity_status='verified local assets', runtime_validated=False),
        dict(id='tts', display_name='Microsoft Zira', family='Frozen utility', availability='available',
        research_role='Clear synthesized English speech', checkpoint_reference='Windows SAPI5 installed voice',
        input_contract={'max_characters':2000}, preprocessing='Module 6 Zira; 175 words/minute; volume 1.0.',
        limitations=['Standard synthetic voice; no original voice preservation.'], integrity_status='installed runtime; probe required', runtime_validated=False)]
    hubert_lock = d2['frozen_decisions']['model_lock']
    backbone = {'hubert':{'directory':str(STUDY/'experiments/accuracy_hubert/cache/checkpoint'/hubert_lock['revision']),**hubert_lock},
                'wavlm':{'directory':str(v4/'cache/model/4c66d4806a428f2e922ccfa1a962776e232d487b'),**read(v4/'configs/model_lock.json')}}
    runtime_versions={**read(v4/'results/FROZEN_DECISIONS.json')['versions'],'openai-whisper':'20250625','pyttsx3':'2.99'}
    registry = {'schema_version':1,'models':models,'backbones':backbone,'runtime_versions':runtime_versions,'source_roots':{'original':str(ORIGINAL),'study':str(STUDY)}}
    save(REGISTRY, registry)
    research = {'models':[{k:v for k,v in m.items() if k not in ('checkpoint_reference',)} for m in models[:6]],
                'folds':{k:[{'fold':f['fold'],'metrics':f['metrics'],'candidate':f['candidate']} for f in v['folds']] for k,v in d4['models'].items()},
                'speaker_confusions':{k:v['speaker_descriptive']['metrics']['confusion_matrix'] for k,v in d4['models'].items()},
                'limitations':d4['limitations'],'dataset':{'speakers':15,'recordings':8673,'folds':3}, 'report_seal':seal()}
    save(ROOT/'backend/research_data.json', research)
    return registry

def strong_paths(reg):
    paths = set()
    for root in (ORIGINAL, STUDY):
        for folder in ('src','configs','docs'):
            paths.update(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for folder in ('accuracy_hubert','accuracy_hubert_v2','accuracy_hubert_v3','accuracy_wavlm_v4'):
        root=STUDY/'experiments'/folder
        for part in ('src','results','configs'):
            paths.update(p for p in (root/part).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for m in reg['models']:
        refs=m['checkpoint_reference']
        if isinstance(refs,dict): paths.update(Path(v['path']) for v in refs.values())
        elif isinstance(refs,list): paths.update(map(Path,refs))
        elif Path(refs).is_file(): paths.add(Path(refs))
    for b in reg['backbones'].values(): paths.update(Path(b['directory'])/f for f in b['files'])
    paths.update(Path(k) for k in read(STUDY/'experiments/accuracy_wavlm_v4/results/REPORT_SEAL.json')['files'])
    return sorted(paths)

if __name__=='__main__':
    mode = sys.argv[1] if len(sys.argv)>1 else 'before'
    if mode=='registry':
        reg=discover()
        save(ROOT/'docs/integration_hashes_before.json',{str(p):sha(p) for p in strong_paths(reg)})
        print('Registry and integration hashes refreshed before first inference')
    elif mode=='before':
        if (ROOT/'docs/preservation_before.json').exists(): raise RuntimeError('Baseline already exists; do not overwrite')
        reg=discover()
        data={'inventories':{str(r):inventory(r) for r in (ORIGINAL,STUDY)},
              'sha256':{str(p):sha(p) for p in strong_paths(reg)}, 'v4_seal':seal(),
              'git_status':{str(r):subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True) for r in (ORIGINAL,STUDY)}}
        save(ROOT/'docs/preservation_before.json',data)
        print('Baseline', {r:len(v) for r,v in data['inventories'].items()}, 'SHA256 files',len(data['sha256']))
    else:
        before=read(ROOT/'docs/preservation_before.json')
        changed=[]
        for r, old in before['inventories'].items():
            now=inventory(Path(r))
            changed += [r+'/'+p for p in old.keys()|now.keys() if old.get(p)!=now.get(p)]
        hashes=[p for p,h in before['sha256'].items() if not Path(p).is_file() or sha(p)!=h]
        extra=ROOT/'docs/integration_hashes_before.json'
        if extra.exists(): hashes += [p for p,h in read(extra).items() if not Path(p).is_file() or sha(p)!=h]
        git_changed=[r for r,s in before['git_status'].items() if subprocess.check_output(['git','-C',r,'status','--porcelain'],text=True)!=s]
        result={'status':'PASS' if not changed and not hashes and not git_changed else 'FAIL',
            'inventory_changes':changed, 'sha256_changes':hashes,'git_status_changes':git_changed,
            'files_inventoried':sum(map(len,before['inventories'].values())), 'files_hashed':len(before['sha256']), 'v4_seal':seal()}
        supplemental=read(extra) if extra.exists() else {}
        result['additional_assets_hashed']=len(set(supplemental)-set(before['sha256']))
        result['unique_files_hashed']=len(set(supplemental)|set(before['sha256']))
        save(ROOT/'docs/preservation_after.json',result)
        print(json.dumps(result,indent=2)); sys.exit(0 if result['status']=='PASS' else 1)
