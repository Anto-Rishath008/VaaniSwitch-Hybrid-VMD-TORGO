import {it,expect,vi,afterEach} from 'vitest';
import {renderHook,act,cleanup} from '@testing-library/react';
import {useRecorder} from './useRecorder';
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('releases a microphone permission result arriving after navigation',async()=>{
  const stop=vi.fn();let resolve!:(stream:MediaStream)=>void;
  const getUserMedia=vi.fn(()=>new Promise<MediaStream>(r=>{resolve=r;}));
  vi.stubGlobal('navigator',{mediaDevices:{getUserMedia}});
  vi.stubGlobal('MediaRecorder',class {});
  const onRecorded=vi.fn(),onError=vi.fn();
  const hook=renderHook(()=>useRecorder(onRecorded,onError));
  let request!:Promise<void>;
  act(()=>{request=hook.result.current.start();});
  hook.unmount();
  await act(async()=>{resolve({getTracks:()=>[{stop}]} as unknown as MediaStream);await request;});
  expect(stop).toHaveBeenCalledOnce();expect(onRecorded).not.toHaveBeenCalled();expect(onError).not.toHaveBeenCalled();
});
