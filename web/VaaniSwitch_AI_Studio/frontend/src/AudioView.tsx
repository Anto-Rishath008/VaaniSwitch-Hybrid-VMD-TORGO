import {useEffect,useRef,useState} from 'react';
import {Download,AudioLines,Activity} from 'lucide-react';
import type {Asset} from './types';

export function SignalCanvas({asset,kind='waveform',cursor=0}:{asset:Asset;kind?:string;cursor?:number}){
  const ref=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{
    const canvas=ref.current;if(!canvas)return;
    const draw=()=>{
      const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
      canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
      const ctx=canvas.getContext('2d');if(!ctx)return;ctx.scale(dpr,dpr);
      const w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);
      if(kind==='spectrogram'){
        const rows=asset.visualization.spectrogram,cols=rows[0]?.length||0;
        rows.forEach((r,y)=>r.forEach((db,x)=>{const v=(db+100)/100;
          ctx.fillStyle=`hsl(${245-v*90} ${25+v*60}% ${8+v*57}%)`;
          ctx.fillRect(x*w/cols,h-(y+1)*h/rows.length,w/cols+1,h/rows.length+1);
        }));
      }else{
        ctx.strokeStyle='#1a2938';ctx.lineWidth=1;
        for(let y=0;y<5;y++){ctx.beginPath();ctx.moveTo(0,y*h/4);ctx.lineTo(w,y*h/4);ctx.stroke();}
        for(let x=0;x<9;x++){ctx.beginPath();ctx.moveTo(x*w/8,0);ctx.lineTo(x*w/8,h);ctx.stroke();}
        ctx.strokeStyle='#70e4d2';ctx.lineWidth=Math.max(1,w/asset.visualization.peaks.length*.65);
        ctx.beginPath();asset.visualization.peaks.forEach(([lo,hi],i)=>{const x=i*w/asset.visualization.peaks.length;ctx.moveTo(x,h/2-hi*h*.44);ctx.lineTo(x,h/2-lo*h*.44);});ctx.stroke();
      }
      if(cursor>0){ctx.strokeStyle='#ffffff';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(cursor*w,0);ctx.lineTo(cursor*w,h);ctx.stroke();}
    };
    const observer=new ResizeObserver(draw);observer.observe(canvas);draw();return()=>observer.disconnect();
  },[asset,kind,cursor]);
  return <canvas ref={ref} className="signal-canvas" role="img" aria-label={kind==='spectrogram'?'Time-frequency spectrogram of the supplied audio':'Waveform of the supplied audio'}/>;
}

export function AudioPanel({asset,title='Original speech'}:{asset:Asset;title?:string}){
  const [kind,setKind]=useState('waveform'),[cursor,setCursor]=useState(0);
  const [browserMetadata,setBrowserMetadata]=useState('');
  useEffect(()=>{
    let cancelled=false;const context=new AudioContext();
    fetch(asset.audio_url).then(r=>r.arrayBuffer()).then(b=>context.decodeAudioData(b)).then(audio=>{
      if(!cancelled)setBrowserMetadata(`Browser decoded ${audio.length.toLocaleString()} samples · ${(audio.sampleRate/1000).toFixed(1)} kHz playback`);
    }).catch(()=>{if(!cancelled)setBrowserMetadata('Validated waveform supplied by local audio decoder.');}).finally(()=>{void context.close();});
    return()=>{cancelled=true;};
  },[asset]);
  return <div className="audio-panel"><div className="panel-heading"><div className="eyebrow"><AudioLines size={15}/>{title}</div><span className="mono">{asset.metadata.duration_seconds.toFixed(2)} s</span></div>
    <div className="segmented compact" aria-label="Signal visualization"><button aria-pressed={kind==='waveform'} onClick={()=>setKind('waveform')}>Waveform</button><button aria-pressed={kind==='spectrogram'} onClick={()=>setKind('spectrogram')}>Spectrogram</button></div>
    <SignalCanvas asset={asset} kind={kind} cursor={cursor}/><div className="axis"><span>0.0 s</span><span>{kind==='spectrogram'?'0–8 kHz · dB amplitude':'Actual audio samples'}</span><span>{asset.metadata.duration_seconds.toFixed(1)} s</span></div>
    <audio controls src={asset.audio_url} onTimeUpdate={e=>setCursor(e.currentTarget.currentTime/asset.metadata.duration_seconds)} aria-label={`${title} playback`}/>
    <div className="audio-caption"><span className="truncate" title={asset.name}>{asset.name}</span><a href={asset.download_url} download aria-label={`Download ${title}`}><Download size={16}/></a></div>
    <p className="small muted browser-audio-meta">{browserMetadata}</p>
    {kind==='spectrogram'&&<p className="small muted">{asset.visualization.spectrogram_contract}</p>}
  </div>;
}

export function EmptySignal(){return <div className="empty-signal"><div className="signal-grid"/><Activity size={42} strokeWidth={1}/><h3>Your voice, visualized.</h3><p>Upload or record audio to reveal its real waveform and frequency structure.</p><span className="eyebrow">NO AUDIO LOADED</span></div>;}
