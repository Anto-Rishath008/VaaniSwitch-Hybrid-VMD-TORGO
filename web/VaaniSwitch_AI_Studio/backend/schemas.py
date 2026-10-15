from typing import Any,Literal
from pydantic import BaseModel,ConfigDict

class AudioMetadata(BaseModel):
    duration_seconds:float
    sample_rate:int
    channels:int
    codec:str
    bytes:int
    rms:float
    clipped_fraction:float
    warnings:list[str]
    sha256:str
    downmix:str

class AudioAsset(BaseModel):
    id:str
    session_id:str
    name:str
    metadata:AudioMetadata
    visualization:dict[str,Any]
    audio_url:str
    download_url:str

class InferenceResult(BaseModel):
    model_config=ConfigDict(extra='allow')
    request_id:str
    model_id:str|None=None
    prediction_label:Literal['Control','Dysarthric']|None=None
    class_probabilities:dict[str,float]|None=None
    threshold:float|None=None
    fold_context:int|None=None
    recognized_text:str|None=None
    audio:AudioAsset|None=None

class JobView(BaseModel):
    id:str
    status:Literal['queued','running','succeeded','failed','cancelled']
    stage:str
    created:float
    started:float|None=None
    finished:float|None=None
    elapsed_seconds:float|None=None
    results:list[InferenceResult]
    error:str|None=None
