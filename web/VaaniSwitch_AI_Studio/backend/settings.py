from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get('VAANI_ORIGINAL', ROOT.parent / 'VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project')).resolve()
STUDY = Path(os.environ.get('VAANI_STUDY', ROOT.parent / 'VaaniSwitch_Accuracy_Study')).resolve()
RUNTIME = ROOT / 'runtime'
REGISTRY = ROOT / 'backend/model_registry.json'
MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 60
SESSION_TTL = 3600

def prepare_environment():
    # Imports and third-party caches may never write into the research checkouts.
    import sys
    sys.dont_write_bytecode = True
    for key, value in {'HF_HOME': RUNTIME/'hf', 'HF_HUB_CACHE': RUNTIME/'hf/hub',
                       'NUMBA_CACHE_DIR': RUNTIME/'numba', 'MPLCONFIGDIR': RUNTIME/'mpl',
                       'TORCH_HOME': RUNTIME/'torch', 'TEMP': RUNTIME/'tmp', 'TMP': RUNTIME/'tmp'}.items():
        os.environ[key] = str(value)
        Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
    sys.path[:0] = [str(STUDY), str(ORIGINAL/'src')]
