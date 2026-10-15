import {useEffect,useRef,useState} from 'react';
import {Activity,Info} from 'lucide-react';
import {api} from './api';
import type {Research} from './types';

export function ASREvidence({research}:{research:Research|null}){
  const asr=research?.asr;if(!asr?.corpus)return null;
  return <section className="section panel asr-evidence"><div className="eyebrow">HISTORICAL MODULE 6 EVALUATION</div><h2>Recognition has its own evidence.</h2><div className="asr-metrics">{[[asr.corpus.wer.percentage,'Micro word error rate'],[asr.corpus.cer.percentage,'Character error rate'],[asr.condition.control.wer.percentage,'Control speech WER'],[asr.condition.dysarthric.wer.percentage,'Dysarthric speech WER']].map(([v,l])=><div key={l}><b>{v.toFixed(2)}%</b><span>{l}</span></div>)}</div><p>{asr.execution_counts.successfully_transcribed} / {asr.execution_counts.requested} historical cases completed. These are corpus evaluation metrics; a new recording does not have a known accuracy.</p><details><summary>Understanding word error rate</summary><p>WER counts substitutions (wrong words), deletions (missing words), and insertions (extra words), divided by the reference word count. CER applies the same idea to characters. A raw 1 − WER value is not a universal recognition accuracy.</p><p>The synthesis evidence was a limited listening check of two demonstrations. No formal MOS, clinical intelligibility improvement or original-voice fidelity was established.</p></details></section>;
}

type Example={logical_id:string;kind:string;contract:string;graph_layout:string;mel:number[][];recurrence:number[][];graph_nodes:number[][];graph_edges:number[][]};
export function CachedRepresentation(){
  const [example,setExample]=useState<Example|null>(null),[error,setError]=useState(''),[tab,setTab]=useState('Recurrence plot'),ref=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{api<Example>('/research/representation-example').then(setExample).catch(e=>setError(e.message));},[]);
  useEffect(()=>{
    const c=ref.current;if(!c||!example)return;
    const paint=()=>{
      const dpr=window.devicePixelRatio||1,rect=c.getBoundingClientRect();c.width=rect.width*dpr;c.height=rect.height*dpr;
      const ctx=c.getContext('2d');if(!ctx)return;ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#0a1520';ctx.fillRect(0,0,w,h);
      if(tab==='Visibility graph'){
        const nodes=example.graph_nodes,min=Math.min(...nodes.map(n=>n[1])),max=Math.max(...nodes.map(n=>n[1]));
        const xy=(i:number)=>[15+nodes[i][0]*(w-30),h-18-(nodes[i][1]-min)/(max-min||1)*(h-36)];
        ctx.strokeStyle='#4b958344';ctx.lineWidth=.6;ctx.beginPath();
        example.graph_edges[0].forEach((source,i)=>{const target=example.graph_edges[1][i];if(source>=target)return;const [x,y]=xy(source),[a,b]=xy(target);ctx.moveTo(x,y);ctx.lineTo(a,b);});ctx.stroke();
        ctx.fillStyle='#91ebd3';nodes.forEach((_,i)=>{const [x,y]=xy(i);ctx.beginPath();ctx.arc(x,y,1.8,0,Math.PI*2);ctx.fill();});
      }else{
        const matrix=tab==='Recurrence plot'?example.recurrence:example.mel,rows=matrix.length,cols=matrix[0].length;
        let min=Infinity,max=-Infinity;for(const r of matrix)for(const v of r){min=Math.min(min,v);max=Math.max(max,v);}
        const image=document.createElement('canvas');image.width=cols;image.height=rows;const ic=image.getContext('2d')!;
        for(let y=0;y<rows;y++)for(let x=0;x<cols;x++){const v=(matrix[y][x]-min)/(max-min||1);ic.fillStyle=tab==='Recurrence plot'?(v?'#8bdcca':'#0d1d29'):`hsl(${240-v*70} ${25+v*40}% ${10+v*65}%)`;ic.fillRect(x,tab==='Recurrence plot'?y:rows-1-y,1,1);}
        ctx.imageSmoothingEnabled=false;ctx.drawImage(image,0,0,w,h);
      }
    };paint();const resize=new ResizeObserver(paint);resize.observe(c);return()=>resize.disconnect();
  },[example,tab]);
  return <section className="panel section cached-representation"><div className="section-heading"><div><div className="eyebrow">PRESERVED REPRESENTATION EXAMPLE</div><h2>Real structure. Frozen provenance.</h2></div><Activity size={21} className="cyan"/></div><div className="segmented compact">{['Recurrence plot','Log-Mel','Visibility graph'].map(t=><button key={t} aria-pressed={tab===t} onClick={()=>setTab(t)}>{t}</button>)}</div>{error?<div className="warning"><Info size={16}/>{error}</div>:example?<><canvas ref={ref} role="img" aria-label={`Preserved ${tab} from TORGO development record ${example.logical_id}`} className="cached-canvas"/><p className="mono">M03 / SESSION2 / 0013 · {tab==='Visibility graph'?`${example.graph_nodes.length} nodes · ${example.graph_edges[0].length/2} undirected edges`:'224 × 224 preserved values'}</p><p className="small muted">{example.kind}. {example.contract}. {tab==='Visibility graph'?example.graph_layout:'Display colors map preserved values; no random matrix or new representation generation.'} Dataset audio is not served.</p></>:<p className="muted small">Loading checksum-verified representation…</p>}</section>;
}
