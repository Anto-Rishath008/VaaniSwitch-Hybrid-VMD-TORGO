"""Seeded mild channel perturbations on private waveform copies, 16 kHz only."""
import numpy as np
from scipy.signal import butter, sosfilt
from .common import digest, require

VERSION = 'v3-mild-channel-1'


def parameters(record):
    seed = int(digest([VERSION, 42, record.logical_id, record.primary_audio_sha256])[:16], 16)
    rng = np.random.default_rng(seed)
    profile = ('band_noise', 'tilt_echo')[int(rng.integers(2))]
    return {'version': VERSION, 'seed': seed, 'profile': profile,
            'gain_db': float(rng.uniform(-3, 3)), 'highpass_hz': float(rng.uniform(60, 120)),
            'lowpass_hz': float(rng.uniform(4000, 6500)), 'tilt_db': float(rng.uniform(-3, 3)),
            'tilt_cutoff_hz': 1000., 'snr_db': float(rng.uniform(25, 35)),
            'echo_delay_ms': float(rng.uniform(10, 30)), 'echo_mix': float(rng.uniform(.05, .12)),
            'sample_rate': 16000, 'peak_limit': .99}


def augment(wave, params):
    require(wave.ndim == 1 and len(wave) >= 400 and np.isfinite(wave).all(), 'Invalid waveform')
    p = params
    require(p['version'] == VERSION and p['sample_rate'] == 16000, 'Augmentation version/rate')
    require(-3 <= p['gain_db'] <= 3 and 25 <= p['snr_db'] <= 35
            and 60 <= p['highpass_hz'] <= 120 and 4000 <= p['lowpass_hz'] <= 6500
            and -3 <= p['tilt_db'] <= 3 and p['tilt_cutoff_hz'] == 1000.
            and 10 <= p['echo_delay_ms'] <= 30 and .05 <= p['echo_mix'] <= .12
            and p['peak_limit'] == .99, 'Perturbation outside frozen mild bounds')
    x = wave.astype(np.float64, copy=True) * 10**(p['gain_db']/20)
    if p['profile'] == 'band_noise':
        x = sosfilt(butter(2, p['highpass_hz'], btype='highpass', fs=16000, output='sos'), x)
        x = sosfilt(butter(2, p['lowpass_hz'], btype='lowpass', fs=16000, output='sos'), x)
    elif p['profile'] == 'tilt_echo':
        low = sosfilt(butter(1, p['tilt_cutoff_hz'], fs=16000, output='sos'), x)
        x += (10**(p['tilt_db']/20)-1)*low
        delay = int(round(p['echo_delay_ms']*16))
        dry = x.copy()
        x[delay:] += p['echo_mix']*dry[:-delay]
    else:
        raise ValueError('Unknown augmentation profile')
    rng = np.random.default_rng(p['seed'])
    noise = rng.standard_normal(len(x))
    noise /= np.sqrt(np.mean(noise**2))
    signal_rms = float(np.sqrt(np.mean(x**2)))
    noise *= signal_rms / 10**(p['snr_db']/20)
    x += noise
    peak = float(np.max(np.abs(x)))
    factor = min(1., p['peak_limit']/peak) if peak else 1.
    x = np.asarray(x*factor, dtype=np.float32)
    require(len(x) == len(wave) and np.isfinite(x).all() and np.max(np.abs(x)) <= 1.,
            'Augmentation numerical corruption')
    return x, {'pre_noise_rms': signal_rms, 'noise_rms': float(np.sqrt(np.mean(noise**2))),
               'peak_rescale': factor, 'output_peak': float(np.max(np.abs(x))),
               'silent_input': signal_rms == 0.}


def choose_records(partition, budget=16):
    partition.check(fitting=True)
    require(1 <= budget <= 16, 'Augmentation budget outside fixed cap')
    records = []
    for speaker in sorted({r.speaker_id for r in partition.records}):
        available = [r for r in partition.records if r.speaker_id == speaker]
        records.extend(sorted(available, key=lambda r: (digest(['augment-select', 42, r.logical_id]),
                                                        r.logical_id))[:budget])
    return tuple(records)
