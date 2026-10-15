import json, queue, subprocess, sys, threading, time, uuid
from pathlib import Path
from .settings import ROOT, RUNTIME

class JobManager:
    def __init__(self, completed=None):
        self.jobs={}; self.queue=queue.Queue(maxsize=8); self.lock=threading.RLock()
        self.process=None; self.running=None; self.completed=completed
        self.closed=False
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()

    def submit(self,session,specs):
        with self.lock:
            if self.queue.full(): raise ValueError('The inference queue is full. Please try again shortly.')
            if sum(j['status'] in ('queued','running') and j['session']==session for j in self.jobs.values())>=2:
                raise ValueError('Wait for the current session jobs to finish.')
            ident=uuid.uuid4().hex
            self.jobs[ident]={'id':ident,'session':session,'status':'queued','stage':'Queued for local processing',
                'created':time.time(),'results':[],'error':None,'specs':specs}
            self.queue.put_nowait(ident)
            return self.public(ident)

    def public(self,ident):
        with self.lock:
            if ident not in self.jobs: raise KeyError(ident)
            return {k:v for k,v in self.jobs[ident].items() if k not in ('specs','session')}

    def cancel(self,ident):
        with self.lock:
            if ident not in self.jobs: raise KeyError(ident)
            j=self.jobs[ident]
            if j['status'] in ('queued','running'):
                j.update(status='cancelled',stage='Cancelled; worker released')
                if self.running==ident and self.process is not None:
                    self.process.terminate()
                    try:self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
            return self.public(ident)

    def remove_session(self,session):
        with self.lock:
            for ident in [i for i,j in self.jobs.items() if j['session']==session]:
                self.cancel(ident);del self.jobs[ident]

    def close(self):
        self.closed=True
        for ident in list(self.jobs):self.cancel(ident)

    def _loop(self):
        while not self.closed:
            try:ident=self.queue.get(timeout=.5)
            except queue.Empty:continue
            try:self._run(ident)
            finally:self.queue.task_done()

    def _run(self,ident):
        with self.lock:
            if ident not in self.jobs or self.jobs[ident]['status']=='cancelled':return
            j=self.jobs[ident];j.update(status='running',started=time.time());self.running=ident
            specs=j['specs']
        for n,spec in enumerate(specs):
            with self.lock:
                if ident not in self.jobs or j['status']=='cancelled':return
                directory=RUNTIME/'sessions'/j['session']
                prefix=directory/f'job-{ident}-{n}'
                result_path=prefix.with_suffix('.result.json')
                spec_path=prefix.with_suffix('.request.json')
                spec={**spec,'result_path':str(result_path)}
                spec_path.write_text(json.dumps(spec),encoding='utf-8')
                target=spec.get('model_id',spec['operation'])
                j['stage']=f'Loading and running {target} ({n+1}/{len(specs)})'
                self.process=subprocess.Popen([sys.executable,'-B','-m','backend.worker',str(spec_path)],cwd=ROOT,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                process=self.process
            try:
                process.wait(timeout=60 if spec['operation']=='synthesize' else 300)
                with self.lock:
                    if ident not in self.jobs or j['status']=='cancelled':return
                    data=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'ok':False}
                    if not data['ok']:raise RuntimeError(data.get('error_type','WorkerFailure'))
                    result=data['result'];result.update(request_id=ident)
                    if self.completed:result=self.completed(j['session'],spec,result)
                    j['results'].append(result)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)
                with self.lock:j['error']='Processing timed out. Try a shorter recording.';j['status']='failed';j['stage']='Timed out'
                return
            except Exception:
                with self.lock:
                    if j['status']=='cancelled':return
                    j['error']='The local model could not complete this operation. Check the runtime diagnostic and retry.'
                    j['status']='failed';j['stage']='Processing failed'
                return
            finally:
                with self.lock:self.process=None
        with self.lock:
            j.update(status='succeeded',stage='Complete',finished=time.time(),elapsed_seconds=time.time()-j['started'])
            self.running=None
