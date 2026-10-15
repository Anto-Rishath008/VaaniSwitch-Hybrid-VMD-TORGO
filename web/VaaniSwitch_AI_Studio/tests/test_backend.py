import io, json, struct, time
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from backend.audio import decode,at_16k,visualize,AudioError
from backend.settings import ROOT,REGISTRY
from backend.main import app,sessions

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:yield c

def wav(samples=None,rate=16000):
    if samples is None:samples=np.sin(np.arange(rate)*2*np.pi*300/rate).astype('float32')*.2
    f=io.BytesIO();sf.write(f,samples,rate,format='WAV',subtype='FLOAT');return f.getvalue()

@pytest.fixture
def audio_file(tmp_path):
    path=tmp_path/'test.wav';path.write_bytes(wav());return path

def test_audio_resampling_preserves_amplitude_and_duration():
    values=np.ones(48000,dtype='float32')*.25
    out=at_16k(values,48000)
    assert len(out)==16000
    assert abs(float(np.mean(out[50:-50]))-.25)<1e-5

def test_stereo_downmix_not_peak_normalized(tmp_path):
    path=tmp_path/'stereo.wav'
    path.write_bytes(wav(np.stack([np.ones(16000)*.2,np.ones(16000)*.4],axis=1)))
    values,rate,meta=decode(path)
    assert np.allclose(values,.3)
    assert meta['channels']==2 and rate==16000

@pytest.mark.parametrize('content',[b'not audio',b'',b'ID3'+b'bad'*100])
def test_invalid_audio_rejected(tmp_path,content):
    path=tmp_path/'invalid';path.write_bytes(content)
    with pytest.raises(AudioError):decode(path)

def test_overdeclared_riff_rejected(tmp_path):
    content=bytearray(wav());content[4:8]=struct.pack('<I',len(content)+100)
    path=tmp_path/'corrupt.wav';path.write_bytes(content)
    with pytest.raises(AudioError,match='header'):decode(path)

def test_nonfinite_audio_rejected(tmp_path):
    x=np.zeros(3200,dtype='float32');x[5]=np.nan
    path=tmp_path/'nan.wav';path.write_bytes(wav(x))
    with pytest.raises(AudioError,match='non-finite'):decode(path)

def test_silence_explicit_warning(tmp_path):
    p=tmp_path/'silence.wav';p.write_bytes(wav(np.zeros(16000)))
    _,_,m=decode(p);assert m['rms']==0 and m['warnings']

def test_too_short_rejected(tmp_path):
    p=tmp_path/'short.wav';p.write_bytes(wav(np.zeros(100)))
    with pytest.raises(AudioError,match='0.1'):decode(p)

def test_long_audio_rejected_without_truncation(tmp_path):
    p=tmp_path/'long.wav';p.write_bytes(wav(np.zeros(16000*61)))
    with pytest.raises(AudioError,match='60'):decode(p)

def test_synthesized_audio_has_separate_duration_budget(tmp_path):
    p=tmp_path/'synthesis.wav';p.write_bytes(wav(np.ones(16000*61,dtype='float32')*.1))
    values,rate,_=decode(p,max_seconds=180)
    assert len(values)==16000*61 and rate==16000

@pytest.mark.parametrize('extension',['wav','mp3','m4a','flac','ogg','webm'])
def test_common_audio_formats_really_decode(tmp_path,extension):
    import imageio_ffmpeg,subprocess
    source=tmp_path/'source.wav';source.write_bytes(wav())
    target=tmp_path/f'encoded.{extension}'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-i',str(source),str(target)],check=True,capture_output=True,timeout=15)
    values,rate,metadata=decode(target)
    assert .9<len(values)/rate<1.2 and metadata['rms']>.1

def test_display_transform_is_real_audio(audio_file):
    x,rate,_=decode(audio_file);v=visualize(x,rate)
    assert len(v['peaks'])==700 and len(v['spectrogram'])==96
    assert max(b[1] for b in v['peaks'])==pytest.approx(.2,abs=.001)
    spec=np.asarray(v['spectrogram']);strongest=spec.mean(1).argmax()
    assert 200<strongest/95*8000<400

def test_health_and_security_headers(client):
    r=client.get('/api/health');assert r.status_code==200
    assert r.json()['processing']=='local'
    assert r.headers['cache-control']=='no-store'
    assert "frame-ancestors 'none'" in r.headers['content-security-policy']

def test_cross_origin_writes_denied(client):
    assert client.post('/api/audio/upload',headers={'Origin':'https://untrusted.example'}).status_code==403

def test_dns_rebinding_host_denied(client):
    assert client.get('/api/health',headers={'Host':'untrusted.example'}).status_code==400

def test_registry_paths_never_public(client):
    r=client.get('/api/models');assert r.status_code==200
    assert 'C:\\' not in r.text and 'OneDrive' not in r.text and 'checkpoint_reference' not in r.text
    models=r.json();assert len(models)==8
    assert [m['availability'] for m in models[:2]]==['historical_only']*2

def test_v4_best_delta_matches_original_inference_contract():
    r=json.loads(REGISTRY.read_text(encoding='utf-8'))
    selected=next(m for m in r['models'] if m['id']=='wavlm-selected')
    for context,reference in selected['checkpoint_reference'].items():
        p=Path(r['source_roots']['study'])/f'experiments/accuracy_wavlm_v4/experiments/context{context}/final/selected/complete.json'
        d=json.loads(p.read_text(encoding='utf-8'))
        assert reference==d['best_checkpoint']
        assert selected['outer_fit_reference'][context]==d['checkpoint']

def test_v2_scaler_transform_import_and_label_mapping():
    from backend.settings import prepare_environment
    prepare_environment()
    from experiments.accuracy_hubert_v2.src.training import transform
    r=json.loads(REGISTRY.read_text(encoding='utf-8'))
    ref=next(m for m in r['models'] if m['id']=='hubert-v2')['checkpoint_reference']['1']
    import torch
    p=torch.load(ref['path'],map_location='cpu',weights_only=False)
    values=transform(np.zeros((1,1536),dtype='float32'),p['scaler'])
    assert np.isfinite(values).all() and p['model'].classes_.tolist()==[0.,1.]

def test_windows_tts_bindings_available():
    import pywintypes
    import pythoncom
    assert pywintypes and pythoncom

def test_upload_play_download_and_delete(client):
    r=client.post('/api/audio/upload',files={'file':('../../private.wav',wav(),'audio/wav')})
    assert r.status_code==200,r.text
    a=r.json();assert a['name']=='private.wav' and 'path' not in a
    assert client.get(a['audio_url']).status_code==200
    d=client.get(a['download_url']);assert 'attachment' in d.headers['content-disposition']
    assert client.delete('/api/sessions/'+a['session_id']).status_code==200
    assert client.get(a['audio_url']).status_code==404
    assert not (ROOT/'runtime/sessions'/a['session_id']).exists()

def test_synthesis_display_name_keeps_unicode(client):
    from backend.main import completed
    a=client.post('/api/audio/upload',files={'file':('source.wav',wav(),'audio/wav')}).json()
    target=ROOT/'runtime/sessions'/a['session_id']/'generated.wav';target.write_bytes(wav())
    result=completed(a['session_id'],{'operation':'synthesize','output_wav':str(target)},{})
    assert result['audio']['name']=='Clear speech \u00b7 Microsoft Zira.wav'
    client.delete('/api/sessions/'+a['session_id'])

def test_invalid_upload_cleans_directory(client):
    before=set(sessions)
    r=client.post('/api/audio/upload',files={'file':('bad.mp3',b'junk','audio/mpeg')})
    assert r.status_code==422 and set(sessions)==before

def test_expired_session_removes_audio_on_access(client):
    a=client.post('/api/audio/upload',files={'file':('expiry.wav',wav(),'audio/wav')}).json()
    sessions[a['session_id']]['created']=time.time()-3601
    assert client.get(a['audio_url']).status_code==404
    assert not (ROOT/'runtime/sessions'/a['session_id']).exists()

def test_oversized_request_rejected_before_decoding(client):
    assert client.post('/api/audio/upload',headers={'Content-Length':str(27*1024*1024)}).status_code==413

def test_silent_inference_rejected(client):
    a=client.post('/api/audio/upload',files={'file':('silence.wav',wav(np.zeros(16000)),'audio/wav')}).json()
    r=client.post('/api/audio/transcribe',json={'session_id':a['session_id'],'asset_id':a['id']})
    assert r.status_code==422
    client.delete('/api/sessions/'+a['session_id'])

def test_historical_model_not_silently_substituted(client):
    r=client.post('/api/audio/analyze',json={'session_id':'a'*32,'asset_id':'b'*32,'model_id':'c2','context':1})
    assert r.status_code==409

@pytest.mark.parametrize('context',[None,0,4])
def test_context_is_explicit_and_bounded(client,context):
    p={'session_id':'a'*32,'asset_id':'b'*32,'model_id':'wavlm-reference'}
    if context is not None:p['context']=context
    assert client.post('/api/audio/analyze',json=p).status_code==422

def test_unknown_asset_and_traversal(client):
    assert client.get('/api/audio/'+'c'*32).status_code==404
    assert client.get('/api/research/figures/checkpoint.pt').status_code==404
    assert client.get('/api/research/download/REPORT_SEAL.json').status_code==404
    assert client.get('/api/nonexistent').status_code==404

def test_research_values_and_reference_role_preserved(client):
    r=client.get('/api/research/metrics');assert r.status_code==200
    d=r.json();models={m['id']:m for m in d['models']}
    assert models['wavlm-reference']['evaluation_metrics']['accuracy']['mean']==pytest.approx(.8018336365926424)
    assert models['wavlm-selected']['research_role']=='Development-Selected V4'
    assert len(d['development'])==3 and d['report_seal']['status']=='PASS'
    assert 'per_record' not in d['asr'] and 'C:\\' not in r.text

def test_speaker_rows_and_allowed_downloads(client):
    rows=client.get('/api/research/speakers').json();assert len(rows)==30
    selected=[s for s in rows if s['model']=='selected']
    assert sorted(s['speaker_id'] for s in selected if s['prediction']!=s['label'])==['F03','F04','FC03','M03']
    assert client.get('/api/research/download/folds').status_code==200

def test_preserved_representations_are_real_and_bounded(client):
    r=client.get('/api/research/representation-example');assert r.status_code==200
    d=r.json();assert np.asarray(d['recurrence']).shape==(224,224)
    assert set(np.asarray(d['recurrence']).flat)=={0,1}
    assert np.asarray(d['graph_nodes']).shape==(256,2)
    assert np.asarray(d['graph_edges']).shape==(2,4286)
    assert 'C:\\' not in r.text

def test_job_failure_and_cancellation_are_terminal(tmp_path):
    from backend.jobs import JobManager
    directory=ROOT/'runtime/sessions'/('d'*32);directory.mkdir(parents=True,exist_ok=True)
    m=JobManager()
    try:
        j=m.submit('d'*32,[{'operation':'classify','audio_path':str(tmp_path/'missing.wav'),'model_id':'wavlm-reference','context':1}])
        deadline=time.time()+20
        while m.public(j['id'])['status'] in ('queued','running') and time.time()<deadline:time.sleep(.1)
        assert m.public(j['id'])['status']=='failed'
        assert 'C:\\' not in json.dumps(m.public(j['id']))
        second=m.submit('d'*32,[{'operation':'classify'}]);m.cancel(second['id'])
        assert m.public(second['id'])['status']=='cancelled'
        m.remove_session('d'*32);assert not m.jobs
    finally:
        m.close()
        import shutil
        shutil.rmtree(directory,ignore_errors=True)
