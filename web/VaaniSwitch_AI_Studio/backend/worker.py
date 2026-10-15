"""One bounded inference process, read-only research imports, no training paths."""
import sys, json, time
from pathlib import Path
from .settings import ROOT, ORIGINAL, REGISTRY, prepare_environment
prepare_environment()
import numpy as np
from .audio import at_16k, decode

def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def verify_runtime_versions(versions):
    import importlib.metadata
    for name,expected in versions.items():
        if importlib.metadata.version(name)!=expected:raise ValueError('Runtime dependency drift: '+name)

def verified_payload(reference):
    import torch
    p=Path(reference['path']).resolve()
    if not (p.is_relative_to(ORIGINAL) or p.is_relative_to(Path(registry['source_roots']['study']))): raise ValueError('Checkpoint outside allowed roots')
    if sha(p)!=reference['sha256']: raise ValueError('Checkpoint integrity check failed')
    # Only audited, local, checksum-pinned research pickle payloads are accepted.
    payload=torch.load(p,map_location='cpu',weights_only=False)
    if payload['identity']!=reference['identity']: raise ValueError('Checkpoint identity mismatch')
    return payload

def verify_backbone(kind):
    b=registry['backbones'][kind]
    for filename,expected in b['files'].items():
        if sha(Path(b['directory'])/filename)!=expected: raise ValueError('Backbone integrity check failed')
    return Path(b['directory'])

def classify(spec,values,rate):
    import torch
    from experiments.accuracy_wavlm_v4.src.data import segment
    model_id=spec['model_id']; context=str(spec['context'])
    entry=next(m for m in registry['models'] if m['id']==model_id)
    if entry['availability']!='research_only': raise ValueError('Model inference unavailable')
    if not torch.cuda.is_available(): raise ValueError('CUDA is required for the frozen classifier extraction contract')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    from experiments.accuracy_wavlm_v4.src.numerics import configure_numerics
    configure_numerics(42)
    reference=entry['checkpoint_reference'][context]
    payload=verified_payload(reference)
    waveform,pre=segment(values,rate,entry['input_contract']['segment_seconds'])
    if model_id.startswith('hubert'):
        from experiments.accuracy_hubert_v2.src.features import Extractor
        extractor=Extractor({'features':{'layers':[6,9,12],'extraction_batch_cap':1}},verify_backbone('hubert'))
        load_seconds=time.perf_counter()-started
        infer_start=time.perf_counter()
        with torch.inference_mode(): features=extractor.infer([waveform])
        layer=payload['layer'] if model_id=='hubert-v2' else payload['candidate']['layer']
        x=features[:,[6,9,12].index(layer),:]
        if model_id=='hubert-v2':
            from experiments.accuracy_hubert_v2.src.training import transform
            if payload['model'].classes_.tolist()!=[0.,1.]: raise ValueError('Label mapping differs')
            probability=float(payload['model'].predict_proba(transform(x,payload['scaler']))[0,1])
        else:
            from experiments.accuracy_hubert_v3.src.training import predict
            if payload['model'].classes_.tolist()!=[0.,1.]: raise ValueError('Label mapping differs')
            probability=float(predict(payload,x)[0])
        pre.update(hidden_layer=layer,extraction_precision='FP16 autocast, FP32 pooled features',vmd=False)
        configuration=entry['configurations'][context]
    else:
        from experiments.accuracy_wavlm_v4.src.model import load_backbone,inputs,Classifier,pool,feature_lengths
        from experiments.accuracy_wavlm_v4.src.training import restore_delta,predict_lr
        backbone,processor,_=load_backbone(verify_backbone('wavlm'))
        configuration=entry['configurations'][context]
        if model_id=='wavlm-selected':
            model=Classifier(backbone,configuration['depth'],configuration['pooling'],checkpointing=False)
            restore_delta(model,payload['model'])
        else: model=backbone
        model.cuda().eval().requires_grad_(False)
        x,mask=inputs(processor,waveform,'cuda')
        torch.cuda.synchronize();load_seconds=time.perf_counter()-started;infer_start=time.perf_counter()
        with torch.inference_mode():
            if model_id=='wavlm-selected': probability=float(model(x,mask).sigmoid()[0])
            else:
                states=model(x,attention_mask=mask,output_hidden_states=True).hidden_states
                lengths=feature_lengths(mask.sum(1),model.config.conv_kernel,model.config.conv_stride)
                features=pool(states[12],lengths,'mean_std').cpu().numpy()
                if payload['model'].classes_.tolist()!=[0.,1.]: raise ValueError('Label mapping differs')
                probability=float(predict_lr(payload,features)[0])
        pre.update(extraction_precision='FP32',vmd=False)
    torch.cuda.synchronize()
    if not np.isfinite(probability) or not 0<=probability<=1: raise ValueError('Invalid model output')
    return {'model_id':model_id,'model_version':entry['family'],'research_role':entry['research_role'],
        'fold_context':int(context),'configuration':configuration,'checkpoint_identity':reference['sha256'],
        'prediction_label':'Dysarthric' if probability>=.5 else 'Control',
        'class_probabilities':{'Control':1-probability,'Dysarthric':probability},'threshold':.5,
        'preprocessing_metadata':pre,'model_load_seconds':load_seconds,'inference_duration_ms':(time.perf_counter()-infer_start)*1000,
        'gpu':{'name':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()},'warnings':entry['limitations'],
        'execution_status':'succeeded','provenance':'Exact frozen preprocessing and saved fitted head; explicit research fold context; no ensemble.'}

def recognize(values,rate):
    import torch,whisper
    from vaaniswitch.m6_reconstruction.asr import transcribe_audio
    entry=next(m for m in registry['models'] if m['id']=='asr')
    checkpoint=Path(entry['checkpoint_reference'])
    if sha(checkpoint)!=entry['expected_sha256']: raise ValueError('Whisper checkpoint integrity failed')
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable; frozen ASR runtime requires CUDA on this machine')
    torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.8);torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();model=whisper.load_model(str(checkpoint),device='cuda').eval().requires_grad_(False)
    load=time.perf_counter()-started
    result=transcribe_audio(model,at_16k(values,rate),device='cuda')
    return {'recognized_text':result['raw_text'],'processing_duration_seconds':result['elapsed_seconds'],
        'model_load_seconds':load,'model_identity':entry['expected_sha256'],'model':'Whisper small.en',
        'reference_context_used':False,'raw_transcript_immutable':True,'decoding':result['decoding'],
        'input_sha256':result['input_float32_sha256'],'gpu':{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved()},'warnings':['Batch English ASR. Review recognition errors before synthesis.']}

def synthesize(spec):
    import pyttsx3
    from vaaniswitch.m6_reconstruction.tts import select_required_voice
    from vaaniswitch.m6_reconstruction.config import VOICE_ID
    import soundfile as sf
    text=spec['text'].strip()
    if not text or len(text)>2000: raise ValueError('Enter 1–2000 characters')
    output=Path(spec['output_wav']).resolve()
    if not output.is_relative_to(ROOT/'runtime'): raise ValueError('Output outside runtime')
    started=time.perf_counter();engine=pyttsx3.init(driverName='sapi5')
    voice=select_required_voice(engine.getProperty('voices'))
    engine.setProperty('voice',VOICE_ID);engine.setProperty('rate',175);engine.setProperty('volume',1.0)
    engine.startLoop(False)
    try: engine.iterate()
    finally: engine.endLoop()
    engine.save_to_file(text,str(output));engine.runAndWait();engine.stop()
    audio,rate=sf.read(output,dtype='float32')
    if not len(audio) or not np.isfinite(audio).all() or float(abs(audio).max())==0: raise ValueError('Synthesis produced no audible samples')
    return {'voice':voice.name,'engine':'pyttsx3 2.99 / SAPI5','rate':175,'sample_rate':rate,
        'duration_seconds':len(audio)/rate,'processing_duration_seconds':time.perf_counter()-started,
        'synthesis_text':text,'warnings':['Standard synthetic voice; no voice cloning or original voice recovery.']}

if __name__=='__main__':
    registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
    spec=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        verify_runtime_versions(registry.get('runtime_versions',{}))
        if spec['operation']=='synthesize': result=synthesize(spec)
        else:
            values,rate,metadata=decode(spec['audio_path'])
            if metadata['rms']<1e-5: raise ValueError('Recording is silent; record audible speech')
            if spec['operation']=='transcribe': result=recognize(values,rate)
            else: result=classify(spec,values,rate)
            result['input_audio_metadata']=metadata
        Path(spec['result_path']).write_text(json.dumps({'ok':True,'result':result},allow_nan=False),encoding='utf-8')
    except Exception as exc:
        # No paths, raw transcript or third-party exception strings enter the public error.
        Path(spec['result_path']).write_text(json.dumps({'ok':False,'error_type':type(exc).__name__, 'private_detail':str(exc)}),encoding='utf-8')
        sys.exit(1)
