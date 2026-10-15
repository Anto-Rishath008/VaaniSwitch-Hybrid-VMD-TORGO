"""Focused engineering smoke with newly synthesized speech; never TORGO evaluation."""
import sys,json,subprocess,time,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.settings import prepare_environment,REGISTRY
prepare_environment()

def execute(spec,name):
    directory=ROOT/'runtime/smoke';directory.mkdir(parents=True,exist_ok=True)
    request=directory/f'{name}.request.json';result=directory/f'{name}.result.json'
    request.write_text(json.dumps({**spec,'result_path':str(result)}),encoding='utf-8')
    started=time.perf_counter()
    p=subprocess.run([sys.executable,'-B','-m','backend.worker',str(request)],cwd=ROOT,capture_output=True,text=True,timeout=310)
    d=json.loads(result.read_text(encoding='utf-8')) if result.exists() else {'ok':False,'private_detail':p.stderr[-2000:]}
    print(name,'PASS' if d['ok'] else 'FAIL',round(time.perf_counter()-started,2),'s',flush=True)
    if not d['ok']:print(d.get('private_detail','worker failed'),flush=True)
    return d

if __name__=='__main__':
    directory=ROOT/'runtime/smoke';directory.mkdir(parents=True,exist_ok=True);sample=directory/'sample.wav'
    report={'scope':'Engineering smoke on new Microsoft Zira synthesized speech. No TORGO outer evaluation or accuracy measurement.',
        'source_text':'Every voice deserves to be understood. This is a local speech research demonstration.',
        'tests':{},'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','scikit-learn','openai-whisper','pyttsx3','fastapi','soundfile')}}
    report['tests']['tts']=execute({'operation':'synthesize','text':report['source_text'],'output_wav':str(sample)},'tts')
    if report['tests']['tts']['ok']:
        report['tests']['asr']=execute({'operation':'transcribe','audio_path':str(sample)},'asr')
        for ident in ('hubert-v2','hubert-v3','wavlm-reference','wavlm-selected'):
            for context in (1,2,3):
                name=f'{ident}-context{context}'
                d=execute({'operation':'classify','audio_path':str(sample),'model_id':ident,'context':context},name)
                if d['ok']:
                    p=d['result']['class_probabilities']
                    assert abs(sum(p.values())-1)<1e-7 and all(0<=v<=1 for v in p.values())
                    assert d['result']['prediction_label']==('Dysarthric' if p['Dysarthric']>=.5 else 'Control')
                report['tests'][name]=d
    report['status']='PASS' if len(report['tests'])==14 and all(d['ok'] for d in report['tests'].values()) else 'FAIL'
    (ROOT/'docs/inference_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
    for m in registry['models']:
        key=m['id']
        m['runtime_validated']=all(report['tests'].get(f'{key}-context{i}',{}).get('ok',False) for i in (1,2,3)) if m['availability']=='research_only' else report['tests'].get(key,{}).get('ok',False)
    REGISTRY.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    print(report['status']);sys.exit(0 if report['status']=='PASS' else 1)
