import {useEffect,useRef,useState} from 'react';

export function useRecorder(onRecorded:(file:File)=>void,onError:(message:string)=>void){
  const [status,setStatus]=useState<'idle'|'recording'|'paused'>('idle'),[elapsed,setElapsed]=useState(0),[level,setLevel]=useState(0);
  const recorder=useRef<MediaRecorder|null>(null),stream=useRef<MediaStream|null>(null),context=useRef<AudioContext|null>(null),raf=useRef(0),chunks=useRef<Blob[]>([]),discarded=useRef(false),ticks=useRef(0),canvas=useRef<HTMLCanvasElement>(null),starting=useRef(false);
  const callback=useRef(onRecorded);callback.current=onRecorded;
  const live=useRef(true);
  const errorCallback=useRef(onError);errorCallback.current=onError;
  const release=()=>{cancelAnimationFrame(raf.current);stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;void context.current?.close();context.current=null;setLevel(0);};
  const stop=(discard=false)=>{discarded.current=discard;if(recorder.current&&recorder.current.state!=='inactive')recorder.current.stop();release();setStatus('idle');};
  useEffect(()=>{live.current=true;return()=>{live.current=false;discarded.current=true;if(recorder.current?.state!=='inactive')recorder.current?.stop();cancelAnimationFrame(raf.current);stream.current?.getTracks().forEach(t=>t.stop());void context.current?.close();};},[]);
  useEffect(()=>{if(status!=='recording')return;const timer=setInterval(()=>{ticks.current+=.25;setElapsed(ticks.current);if(ticks.current>=60)stop();},250);return()=>clearInterval(timer);},[status]);
  const start=async()=>{
    if(starting.current||status!=='idle')return;starting.current=true;
    try{
      if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder)throw new Error('Recording requires a supported browser on localhost or HTTPS.');
      const source=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
      if(!live.current){source.getTracks().forEach(t=>t.stop());return;}
      stream.current=source;discarded.current=false;chunks.current=[];ticks.current=0;setElapsed(0);
      const mime=['audio/webm;codecs=opus','audio/ogg;codecs=opus','audio/mp4'].find(m=>MediaRecorder.isTypeSupported(m));
      const rec=new MediaRecorder(source,mime?{mimeType:mime}:undefined);recorder.current=rec;
      rec.ondataavailable=e=>{if(e.data.size)chunks.current.push(e.data);};
      rec.onstop=()=>{if(discarded.current)return;const blob=new Blob(chunks.current,{type:rec.mimeType});if(!blob.size){errorCallback.current('The recording is empty. Please try again.');return;}callback.current(new File([blob],`Microphone recording.${rec.mimeType.includes('mp4')?'m4a':rec.mimeType.includes('ogg')?'ogg':'webm'}`,{type:rec.mimeType}));};
      rec.onerror=()=>{errorCallback.current('The recording was interrupted. Please start a new recording.');stop(true);};
      source.getTracks()[0].onended=()=>{errorCallback.current('The microphone disconnected. Check your device and try again.');stop(true);};
      context.current=new AudioContext();const analyser=context.current.createAnalyser();analyser.fftSize=1024;context.current.createMediaStreamSource(source).connect(analyser);
      const samples=new Uint8Array(analyser.fftSize);let last=0;
      const draw=(timestamp:number)=>{
        analyser.getByteTimeDomainData(samples);
        if(timestamp-last>80){const amplitude=Math.sqrt(samples.reduce((s,v)=>s+((v-128)/128)**2,0)/samples.length);setLevel(Math.min(1,amplitude*5));last=timestamp;}
        const c=canvas.current,ctx=c?.getContext('2d');if(c&&ctx){c.width=c.clientWidth*2;c.height=160;ctx.strokeStyle='#70e4d2';ctx.lineWidth=2;ctx.beginPath();samples.forEach((v,i)=>{const x=i*c.width/samples.length,y=v/255*c.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
        raf.current=requestAnimationFrame(draw);
      };raf.current=requestAnimationFrame(draw);rec.start(250);setStatus('recording');
    }catch(error){release();setStatus('idle');const name=(error as DOMException).name;onError(name==='NotAllowedError'?'Microphone access was denied. Allow microphone access in your browser settings.':name==='NotFoundError'?'No microphone was found. Connect a microphone or upload an audio file.':(error as Error).message||'Unable to record audio.');}
    finally{starting.current=false;}
  };
  const pause=()=>{recorder.current?.pause();setStatus('paused');};
  const resume=()=>{recorder.current?.resume();setStatus('recording');};
  return {status,elapsed,level,canvas,start,stop,pause,resume};
}
