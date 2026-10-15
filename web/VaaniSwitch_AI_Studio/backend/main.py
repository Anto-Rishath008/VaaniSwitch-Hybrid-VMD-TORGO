import asyncio, contextlib, csv, io, json, shutil, time, uuid
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, ConfigDict
from .settings import ROOT, ORIGINAL, STUDY, RUNTIME, REGISTRY, MAX_BYTES, SESSION_TTL, prepare_environment
prepare_environment()
from .audio import decode,visualize,AudioError
from .jobs import JobManager
from .schemas import AudioAsset,JobView
import soundfile as sf

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
registry=read(REGISTRY)
sessions={}

def session(ident):
    if ident not in sessions:raise HTTPException(404,'Audio session not found or expired.')
    if time.time()-sessions[ident]['created']>SESSION_TTL:
        delete_session(ident);raise HTTPException(404,'Audio session expired.')
    return sessions[ident]

def asset(ident):
    for sid,s in list(sessions.items()):
        if ident in s['assets']:
            session(sid)
            return s,s['assets'][ident]
    raise HTTPException(404,'Audio asset not found or expired.')

def register_audio(sid,path,name, *, synthesized=False):
    values,rate,metadata=decode(path,max_seconds=180 if synthesized else 60)
    aid=uuid.uuid4().hex
    canonical=path.parent/f'{aid}.wav'
    sf.write(canonical,values,rate,subtype='FLOAT')
    # Remove the original upload after validation; retain only the validated mono audio.
    if path!=canonical:path.unlink(missing_ok=True)
    record={'id':aid,'session_id':sid,'name':name,'metadata':metadata,'visualization':visualize(values,rate),
        'audio_url':f'/api/audio/{aid}','download_url':f'/api/audio/{aid}?download=true','path':str(canonical)}
    sessions[sid]['assets'][aid]=record
    return {k:v for k,v in record.items() if k!='path'}

def completed(sid,spec,result):
    if spec['operation']=='synthesize':
        result['audio']=register_audio(sid,Path(spec['output_wav']),'Clear speech · Microsoft Zira.wav',synthesized=True)
    return result

manager=JobManager(completed)

def delete_session(sid):
    manager.remove_session(sid)
    if sid in sessions:
        directory=RUNTIME/'sessions'/sid
        if directory.resolve().parent!=(RUNTIME/'sessions').resolve():raise RuntimeError('Unsafe cleanup target')
        shutil.rmtree(directory,ignore_errors=True);del sessions[sid]

async def cleanup_loop():
    while True:
        await asyncio.sleep(30)
        for sid,s in list(sessions.items()):
            if time.time()-s['created']>SESSION_TTL:delete_session(sid)

@asynccontextmanager
async def lifespan(app):
    # Verify the preserved V4 seal before serving research values.
    from scripts.audit import seal
    seal()
    base=RUNTIME/'sessions';base.mkdir(parents=True,exist_ok=True)
    # No session survives an application restart.
    for p in base.iterdir():
        if p.is_dir() and len(p.name)==32 and all(c in '0123456789abcdef' for c in p.name):shutil.rmtree(p)
    task=asyncio.create_task(cleanup_loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):await task
    manager.close()
    for sid in list(sessions):delete_session(sid)

app=FastAPI(title='VaaniSwitch AI Studio',version='1.0.0',lifespan=lifespan,docs_url=None,redoc_url=None)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])

@app.middleware('http')
async def privacy_headers(request,call_next):
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin and origin not in ('http://127.0.0.1:8000','http://localhost:8000','http://127.0.0.1:5173','http://localhost:5173'):
        return JSONResponse({'detail':'Cross-origin writes are not permitted.'},status_code=403)
    length=request.headers.get('content-length')
    if length and int(length)>MAX_BYTES+1024*1024:return JSONResponse({'detail':'Maximum upload size is 25 MB.'},status_code=413)
    response=await call_next(request)
    response.headers.update({'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer',
        'Cache-Control':'no-store' if request.url.path.startswith('/api') else 'no-cache',
        'Permissions-Policy':'microphone=(self), camera=()',
        'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"})
    return response

class Input(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id:str=Field(pattern='^[a-f0-9]{32}$')
    asset_id:str=Field(pattern='^[a-f0-9]{32}$')

class Analysis(Input):
    model_id:str
    context:int=Field(ge=1,le=3)

class Comparison(Input):
    model_ids:list[str]=Field(min_length=1,max_length=4)
    context:int=Field(ge=1,le=3)

class Synthesis(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id:str=Field(pattern='^[a-f0-9]{32}$')
    text:str=Field(min_length=1,max_length=2000)

def public_model(m):
    allowed=('id','display_name','family','architecture','source_project','input_contract','output_contract','fold_context','availability','research_role','evaluation_metrics','limitations','inference_adapter','integrity_status','preprocessing','configurations','blocker','speaker_accuracy','runtime_validated')
    out={k:m[k] for k in allowed if k in m}
    if isinstance(m.get('checkpoint_reference'),dict):out['checkpoint_identities']={k:v['sha256'] for k,v in m['checkpoint_reference'].items()}
    return out

@app.get('/api/health')
def health():return {'status':'ok','processing':'local','queue':'serial subprocess','retention_seconds':SESSION_TTL,'classification':'research fold-context only'}

@app.get('/api/models')
def models():return [public_model(m) for m in registry['models']]

@app.get('/api/models/{model_id}')
def model(model_id):
    found=next((m for m in models() if m['id']==model_id),None)
    if not found:raise HTTPException(404,'Model not found')
    return found

@app.post('/api/audio/upload',response_model=AudioAsset)
@app.post('/api/audio/recording',response_model=AudioAsset)
async def upload(file:UploadFile=File(...)):
    if len(sessions)>=30:raise HTTPException(429,'Local session capacity reached. Remove an audio session.')
    sid=uuid.uuid4().hex;directory=RUNTIME/'sessions'/sid;directory.mkdir(parents=True)
    path=directory/'incoming.audio';size=0
    try:
        with path.open('wb') as f:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_BYTES:raise HTTPException(413,'Maximum upload size is 25 MB.')
                f.write(chunk)
        sessions[sid]={'created':time.time(),'assets':{}}
        # Filename is a display label only, never a filesystem path.
        name=(file.filename or 'Recording').replace('\\','/').split('/')[-1][:150]
        result=await asyncio.to_thread(register_audio,sid,path,name)
        return result
    except (AudioError,ValueError) as exc:
        delete_session(sid);raise HTTPException(422,str(exc)) from None
    except Exception:
        delete_session(sid)
        if path.parent.exists():shutil.rmtree(path.parent)
        raise
    finally:await file.close()

@app.get('/api/audio/{asset_id}')
def get_audio(asset_id:str,download:bool=False):
    _,a=asset(asset_id)
    return FileResponse(a['path'],media_type='audio/wav',filename='vaaniswitch-audio.wav' if download else None)

def input_spec(data):
    s=session(data.session_id)
    a=s['assets'].get(data.asset_id)
    if not a:raise HTTPException(404,'Audio does not belong to this session.')
    if a['metadata']['rms']<1e-5:raise HTTPException(422,'Recording is silent. Record audible speech.')
    return {'audio_path':a['path']}

def enqueue(sid,specs):
    try:return manager.submit(sid,specs)
    except ValueError as exc:raise HTTPException(429,str(exc)) from None

def check_classifier(ident):
    m=next((m for m in registry['models'] if m['id']==ident),None)
    if not m or m['availability']!='research_only':raise HTTPException(409,'This classifier is historical-only or unavailable. Choose a verified research preview.')

@app.post('/api/audio/analyze',response_model=JobView,response_model_exclude_none=True)
def analyze(data:Analysis):
    check_classifier(data.model_id)
    return enqueue(data.session_id,[{**input_spec(data),'operation':'classify','model_id':data.model_id,'context':data.context}])

@app.post('/api/audio/compare',response_model=JobView,response_model_exclude_none=True)
def compare(data:Comparison):
    if len(data.model_ids)!=len(set(data.model_ids)):raise HTTPException(422,'Choose distinct models.')
    for ident in data.model_ids:check_classifier(ident)
    source=input_spec(data)
    return enqueue(data.session_id,[{**source,'operation':'classify','model_id':ident,'context':data.context} for ident in data.model_ids])

@app.post('/api/audio/transcribe',response_model=JobView,response_model_exclude_none=True)
@app.post('/api/audio/full-pipeline',response_model=JobView,response_model_exclude_none=True)
def transcribe(data:Input):
    # The full conversion deliberately pauses at transcript review.
    return enqueue(data.session_id,[{**input_spec(data),'operation':'transcribe'}])

@app.post('/api/audio/synthesize',response_model=JobView,response_model_exclude_none=True)
def synthesize(data:Synthesis):
    session(data.session_id)
    if not data.text.strip():raise HTTPException(422,'Enter some English text.')
    directory=RUNTIME/'sessions'/data.session_id
    return enqueue(data.session_id,[{'operation':'synthesize','text':data.text,'output_wav':str(directory/f'{uuid.uuid4().hex}.tts.wav')}])

@app.get('/api/jobs/{job_id}',response_model=JobView,response_model_exclude_none=True)
def job(job_id):
    try:return manager.public(job_id)
    except KeyError:raise HTTPException(404,'Job not found or expired') from None

@app.post('/api/jobs/{job_id}/cancel',response_model=JobView,response_model_exclude_none=True)
def cancel(job_id):
    try:return manager.cancel(job_id)
    except KeyError:raise HTTPException(404,'Job not found or expired') from None

@app.get('/api/jobs/{job_id}/events')
async def events(job_id):
    job(job_id)
    async def stream():
        while True:
            try:d=job(job_id)
            except HTTPException:return
            yield 'data: '+json.dumps(d)+'\n\n'
            if d['status'] not in ('queued','running'):break
            await asyncio.sleep(.8)
    return StreamingResponse(stream(),media_type='text/event-stream')

@app.delete('/api/sessions/{session_id}')
def remove(session_id):
    session(session_id);delete_session(session_id);return {'status':'deleted'}

@app.get('/api/research/metrics')
def metrics():
    data=read(ROOT/'backend/research_data.json')
    data['models']=models()[:6]
    m6=read(ORIGINAL/'artifacts/module6/final_evaluation/165831812be61cdf1161921967e4fb134decef70e1e809f2f9402d2901dfda8f/metrics_raw.json')
    data['asr']={k:m6[k] for k in ('condition','corpus','execution_counts','scope','completion_status')}
    decisions=read(STUDY/'experiments/accuracy_wavlm_v4/results/FROZEN_DECISIONS.json')
    data['development']=[{'context':s['outer_context'],'frozen':s['reference']['mean'],'selected':s['selected']['mean'],
                          'split':s['split_identity']} for s in decisions['selections']]
    # Only aggregate evaluator metrics are returned; no reference transcripts.
    return data

@app.get('/api/research/folds')
def folds():return read(ROOT/'backend/research_data.json')['folds']

@app.get('/api/research/speakers')
def speakers():
    path=STUDY/'experiments/accuracy_wavlm_v4/results/speaker_predictions.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    return [{k:float(v) if k not in ('model','speaker_id') and v!='' else v for k,v in r.items()} for r in rows]

FIGURES={'accuracy':'accuracy_comparison.png','confusion':'confusion_matrix.png','speakers':'speaker_accuracy_analysis.png','training':'training_curves.png','memory':'gpu_memory_analysis.png'}

@app.get('/api/research/figures')
def figures():return [{'id':k,'name':v,'url':f'/api/research/figures/{k}'} for k,v in FIGURES.items()]

@app.get('/api/research/representation-example')
def representation_example():
    import numpy as np
    from scripts.audit import sha
    directory=ORIGINAL/'data/processed/representations/d8a36c6ab8c48b15e61ecd63194ffa30ede4e7e7e7ab0a30c197002652f9d65d/M03/Session2/0013'
    if not directory.exists():raise HTTPException(503,'Preserved representation example unavailable.')
    metadata=read(directory/'metadata.json')
    arrays={}
    for key,spec in metadata['artifacts'].items():
        path=directory/spec['filename']
        if sha(path)!=spec['sha256']:raise HTTPException(503,'Representation integrity failed.')
        arrays[key]=np.load(path,allow_pickle=False).tolist()
    return {'logical_id':'M03/Session2/0013','kind':'Preserved historical development representation; no new processing',
            'contract':'Original shared all-mode VMD; frozen Module 2 representations',
            'graph_layout':'x = normalized time; y = standardized block-RMS envelope',**arrays}

@app.get('/api/research/figures/{figure_id}')
def figure(figure_id):
    if figure_id not in FIGURES:raise HTTPException(404,'Figure not found')
    return FileResponse(STUDY/'experiments/accuracy_wavlm_v4/figures'/FIGURES[figure_id],media_type='image/png')

@app.get('/api/research/download/{name}')
def download(name):
    allowed={'folds':'fold_results.csv','speakers':'speaker_predictions.csv','records':'record_predictions.csv','report':'FINAL_EXPERIMENT_REPORT.md'}
    if name not in allowed:raise HTTPException(404,'Artifact not found')
    return FileResponse(STUDY/'experiments/accuracy_wavlm_v4/results'/allowed[name],filename=allowed[name])

@app.get('/{path:path}',include_in_schema=False)
def frontend(path):
    if path.startswith('api/'):raise HTTPException(404,'Endpoint not found')
    dist=ROOT/'frontend/dist';candidate=(dist/path).resolve()
    if candidate.is_relative_to(dist.resolve()) and candidate.is_file():return FileResponse(candidate)
    if (dist/'index.html').is_file():return FileResponse(dist/'index.html')
    return JSONResponse({'detail':'Frontend is not built. Run the setup script.'},status_code=503)
