export async function api<T>(path:string,body?:unknown,method?:string):Promise<T>{
  const response=await fetch(`/api${path}`,{method:method||(body?'POST':'GET'),headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined});
  if(!response.ok){const e=await response.json().catch(()=>({}));throw new Error(typeof e.detail==='string'?e.detail:'The request could not be completed. Check your input and try again.');}
  return response.json();
}
export async function uploadAudio(file:File){
  if(file.size>25*1024*1024)throw new Error('Maximum upload size is 25 MB.');
  const form=new FormData();form.append('file',file);
  const response=await fetch('/api/audio/upload',{method:'POST',body:form});
  if(!response.ok){const e=await response.json();throw new Error(typeof e.detail==='string'?e.detail:'The audio could not be decoded.');}
  return response.json();
}
