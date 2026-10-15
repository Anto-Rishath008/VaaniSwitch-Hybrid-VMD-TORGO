"""Official WavLM download pinned before use; all download/cache writes stay in V4."""
import hashlib
import os
from .common import ROOT, publish, read, require, sha, versions

MODEL = 'microsoft/wavlm-base-plus'
REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
WEIGHT_SHA = '3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'
FILES = ('config.json', 'preprocessor_config.json', 'pytorch_model.bin', 'README.md')


def acquire(download=False):
    directory = ROOT/'cache/model'/REVISION
    lock = ROOT/'configs/model_lock.json'
    if not lock.exists():
        require(download, 'Model not yet verified; run Prepare')
        os.environ['HF_HOME'] = str(ROOT/'cache/huggingface')
        os.environ['HF_HUB_CACHE'] = str(ROOT/'cache/huggingface/hub')
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        from huggingface_hub import HfApi
        info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
        require(info.sha == REVISION, 'Official revision mismatch')
        remote = {s.rfilename: s for s in info.siblings}
        require(remote['pytorch_model.bin'].lfs.sha256 == WEIGHT_SHA, 'Official LFS identity mismatch')
        import requests
        import uuid
        directory.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            path = directory/name
            if not path.exists():
                partial = directory/(name+'.'+uuid.uuid4().hex+'.partial')
                url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
                with requests.get(url, stream=True, timeout=(10, 30)) as response:
                    response.raise_for_status()
                    with partial.open('xb') as stream:
                        count = 0
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk)
                            count += len(chunk)
                            if count % (32*1024*1024) == 0:
                                print(f'{name}: {count/1024**2:.0f} MiB', flush=True)
                        stream.flush()
                        os.fsync(stream.fileno())
                require(partial.stat().st_size == remote[name].size, 'Incomplete official download')
                if remote[name].lfs:
                    require(sha(partial) == remote[name].lfs.sha256, 'Downloaded weight checksum')
                os.rename(partial, path)
        for name in FILES:
            content = (directory/name).read_bytes()
            spec = remote[name]
            require(len(content) == spec.size, 'Official model file size mismatch')
            if spec.lfs:
                require(sha(directory/name) == spec.lfs.sha256, 'Official LFS checksum')
            else:
                blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
                require(blob == spec.blob_id, 'Official Git blob mismatch: '+name)
        publish(lock, {'repository': MODEL, 'revision': REVISION,
            'files': {name: sha(directory/name) for name in FILES},
            'processor_configuration': read(directory/'preprocessor_config.json'),
            'versions_at_acquisition': versions(), 'verification': 'Pinned official Git blobs and LFS SHA256'})
    doc = read(lock)
    require(doc['repository'] == MODEL and doc['revision'] == REVISION, 'Model lock identity changed')
    require(doc['files']['pytorch_model.bin'] == WEIGHT_SHA, 'Weight pin changed')
    for name in FILES:
        require(sha(directory/name) == doc['files'][name], 'Model file corruption: '+name)
    require(read(directory/'preprocessor_config.json') == doc['processor_configuration'], 'Processor changed')
    return directory, doc


if __name__ == '__main__':
    directory, doc = acquire(download=True)
    print({'directory': str(directory), **doc}, flush=True)
