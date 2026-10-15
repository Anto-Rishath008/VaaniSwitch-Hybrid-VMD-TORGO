"""Bounded real audio decoding and display transforms; no model normalization."""
from pathlib import Path
import hashlib, math, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft
from .settings import MAX_BYTES, MAX_SECONDS

class AudioError(ValueError): pass

def validate_riff(path):
    with Path(path).open('rb') as f: head=f.read(12)
    if head[:4]==b'RIFF' and head[8:12]==b'WAVE':
        if int.from_bytes(head[4:8],'little')+8 > Path(path).stat().st_size:
            raise AudioError('The WAV header declares missing data. Export a standard WAV and try again.')

def decode(path, *, max_seconds=MAX_SECONDS):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES: raise AudioError('Maximum upload size is 25 MB.')
    validate_riff(path)
    codec=None
    try:
        info=sf.info(path)
        if not 8000<=info.samplerate<=192000 or not 1<=info.channels<=8: raise AudioError('Unsupported sample rate or channel count.')
        if info.duration>max_seconds: raise AudioError(f'Maximum duration is {max_seconds} seconds. Trim the recording before uploading.')
        values,rate=sf.read(path,dtype='float32',always_2d=True,frames=max_seconds*info.samplerate+1)
        codec=f'{info.format} / {info.subtype}'
    except AudioError: raise
    except (sf.LibsndfileError,RuntimeError):
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        try:
            # PCM output capped by both duration and file size. No unbounded pipe allocation.
            target=path.with_suffix('.decoded.wav')
            subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-i',str(path),'-t',str(max_seconds+1),
                            '-ac','1','-ar','16000','-c:a','pcm_f32le','-fs','5000000',str(target)],
                           capture_output=True,timeout=25,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            values,rate=sf.read(target,dtype='float32',always_2d=True)
            codec='FFmpeg decoded audio (mono 16 kHz); source channel/rate metadata unavailable'
        except (subprocess.SubprocessError,sf.LibsndfileError,RuntimeError):
            raise AudioError('This file is not decodable audio. Try WAV, MP3, M4A, FLAC, OGG or WEBM.') from None
        finally:
            path.with_suffix('.decoded.wav').unlink(missing_ok=True)
    if len(values)<rate*.1: raise AudioError('Record at least 0.1 seconds of audio.')
    if len(values)/rate>max_seconds: raise AudioError(f'Maximum duration is {max_seconds} seconds. Trim the recording before uploading.')
    if not np.isfinite(values).all(): raise AudioError('Audio contains non-finite samples.')
    if abs(values).max()>1.01: raise AudioError('Floating-point audio exceeds the supported amplitude range. Export PCM without amplification.')
    channels=values.shape[1]
    mono=values.mean(axis=1,dtype=np.float32)
    rms=float(np.sqrt(np.mean(mono.astype('float64')**2)))
    clipping=float(np.mean(abs(values)>=.999))
    warnings=[]
    if rms<1e-5: warnings.append('Silent or nearly silent recording; inference is disabled.')
    if clipping>.001: warnings.append('Clipping detected; the recording may affect model outputs.')
    return mono,rate,{'duration_seconds':len(mono)/rate,'sample_rate':rate,'channels':channels,'codec':codec,
        'bytes':path.stat().st_size,'rms':rms,'clipped_fraction':clipping,'warnings':warnings,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'downmix':'arithmetic mean; amplitude not normalized'}

def at_16k(values,rate):
    if rate==16000: return values.copy()
    divisor=math.gcd(int(rate),16000)
    return resample_poly(values,16000//divisor,int(rate)//divisor).astype('float32')

def visualize(values,rate):
    blocks=np.array_split(values,min(700,len(values)))
    peaks=[[float(b.min()),float(b.max())] for b in blocks]
    # A display-only STFT, explicitly distinct from frozen model representations.
    display=at_16k(values,rate)
    _,times,z=stft(display,fs=16000,nperseg=min(512,len(display)),noverlap=min(384,len(display)//2))
    db=20*np.log10(np.maximum(abs(z),1e-6))
    fi=np.linspace(0,db.shape[0]-1,96).astype(int)
    ti=np.linspace(0,db.shape[1]-1,min(220,db.shape[1])).astype(int)
    return {'peaks':peaks,'spectrogram':np.clip(db[fi][:,ti],-100,0).tolist(),
        'spectrogram_contract':'Display STFT: linear frequency 0–8 kHz; 512-sample Hann window; dB amplitude. Not a model Mel input.',
        'duration_seconds':len(values)/rate,'frequency_max_hz':8000}
