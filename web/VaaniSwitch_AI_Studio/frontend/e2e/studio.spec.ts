import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import path from 'node:path';
import fs from 'node:fs';
const shots=path.resolve('../docs/screenshots');fs.mkdirSync(shots,{recursive:true});
const sample=path.resolve('../runtime/smoke/sample.wav');
const routes=['/','/studio','/clear-speech','/models','/pipeline','/research','/speakers','/limitations','/about','/presentation'];

async function screenshot(page:any,name:string){
  await page.evaluate(()=>{(document.activeElement as HTMLElement)?.blur();window.scrollTo(0,0);});
  await page.waitForTimeout(250);
  await page.screenshot({path:path.join(shots,name),fullPage:true});
}

test('desktop routes, accurate metrics, browser errors and visual evidence',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  for(const route of routes){await page.goto(route);await expect(page.locator('main h1').first()).toBeVisible();await expect(page.getByText('Local runtime connected')).toBeVisible();
    await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1)).toBe(true);
    if(['/', '/studio','/research','/pipeline'].includes(route))await page.screenshot({path:path.join(shots,`${route==='/'?'overview':route.slice(1)}-desktop.png`),fullPage:true});
  }
  expect(errors).toEqual([]);
  await page.goto('/research');await expect(page.getByText('80.18%',{exact:true})).toBeVisible();await expect(page.getByText('79.22%',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Macro-F1',exact:true}).click();await expect(page.getByText('77.93%',{exact:true})).toBeVisible();
});

test('mobile routes have no overflow and navigation works',async({page})=>{
  await page.setViewportSize({width:390,height:844});
  for(const route of routes){await page.goto(route);await expect(page.locator('main h1').first()).toBeVisible();await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1)).toBe(true);}
  await page.goto('/');await page.screenshot({path:path.join(shots,'overview-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Open navigation',exact:true}).click();await page.getByRole('link',{name:'Speech studio',exact:true}).click();await expect(page.getByRole('heading',{name:'Speech studio',exact:true})).toBeVisible();
  await page.screenshot({path:path.join(shots,'studio-mobile.png'),fullPage:true});
});

test('upload, real model comparison, ASR, text editing, synthesis, playback and download',async({page})=>{
  await page.goto('/studio');await page.getByLabel('Upload speech audio').setInputFiles(sample);
  await expect(page.getByText('Browser decoded', {exact:false})).toBeVisible();
  await page.getByLabel('Explicit fold context').selectOption('1');await page.getByLabel('I understand this is a research preview.').check();
  await page.getByRole('button',{name:'Compare available models',exact:true}).click();
  await expect(page.getByRole('heading',{name:'One recording. Multiple perspectives.'})).toBeVisible({timeout:150000});
  await expect(page.locator('.result-card')).toHaveCount(4);await expect(page.locator('.result-card').first()).toContainText('Dysarthric probability');
  await page.getByRole('button',{name:'Recognize speech',exact:true}).click();
  const transcript=page.getByLabel('Review or edit the words to be spoken');
  await expect(transcript).toHaveValue(/Every voice deserves to be understood/i,{timeout:60000});
  await transcript.fill('This is an edited message from VaaniSwitch.');await page.getByRole('button',{name:'Generate clear speech',exact:true}).click();
  await expect(page.getByLabel('Synthesized speech playback')).toBeVisible({timeout:30000});
  const output=page.getByLabel('Synthesized speech playback');
  await output.evaluate(async(node:HTMLAudioElement)=>{await node.play();});
  await expect.poll(()=>output.evaluate((node:HTMLAudioElement)=>node.currentTime)).toBeGreaterThan(0);
  const download=page.waitForEvent('download');await page.getByRole('link',{name:'Download Synthesized speech',exact:true}).click();const file=await download;expect(await file.failure()).toBeNull();
  await page.getByText('Original recognized text · preserved').click();await expect(page.locator('.raw-text p')).toContainText('Every voice deserves to be understood');
  await screenshot(page,'studio-real-results.png');
  await page.getByRole('button',{name:'Remove audio session',exact:true}).click();await expect(page.getByRole('heading',{name:'Your voice, visualized.'})).toBeVisible();
});

test('browser recording pause, resume, stop and analyze a virtual microphone input',async({page,context})=>{
  await context.grantPermissions(['microphone']);await page.goto('/studio');await page.getByRole('tab',{name:'Record audio',exact:true}).click();
  await page.getByRole('button',{name:'Start recording',exact:true}).click();await expect(page.getByText('● RECORDING',{exact:true})).toBeVisible();
  await page.waitForTimeout(1600);await page.getByRole('button',{name:'Pause',exact:true}).click();await expect(page.getByText('RECORDING PAUSED',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Resume',exact:true}).click();await page.waitForTimeout(3000);await page.getByRole('button',{name:'Stop recording',exact:true}).click();
  await expect(page.getByText('Microphone recording.webm',{exact:true})).toBeVisible({timeout:30000});
  await page.getByLabel('Explicit fold context').selectOption('2');await page.getByLabel('I understand this is a research preview.').check();await page.getByRole('button',{name:'Analyze speech',exact:true}).click();
  await expect(page.locator('.result-card')).toHaveCount(1,{timeout:60000});
  await page.screenshot({path:path.join(shots,'recorded-audio-analysis.png'),fullPage:true});
  await page.getByRole('button',{name:'Remove audio session',exact:true}).click();
});

test('permission denial and invalid uploads produce useful errors',async({browser})=>{
  const context=await browser.newContext({permissions:[]});
  await context.addInitScript(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Permission denied','NotAllowedError');};});
  const page=await context.newPage();await page.goto('/studio');await page.getByRole('tab',{name:'Record audio',exact:true}).click();await page.getByRole('button',{name:'Start recording',exact:true}).click();await expect(page.getByRole('alert')).toContainText('Microphone access was denied');
  await page.getByRole('tab',{name:'Upload audio',exact:true}).click();await page.getByLabel('Upload speech audio').setInputFiles({name:'invalid.wav',mimeType:'audio/wav',buffer:Buffer.from('This is not audio')});await expect(page.getByRole('alert')).toContainText('not decodable audio');await context.close();
});

test('accessibility scans and measured browser load time',async({page})=>{
  const checks:Record<string,unknown>={};
  for(const route of ['/', '/studio','/clear-speech','/models','/pipeline','/research','/speakers','/limitations','/about']){
    await page.goto(route);await expect(page.getByText('Local runtime connected')).toBeVisible();
    const report=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    checks[route]={violations:report.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))})),load:await page.evaluate(()=>{const n=performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;return {domContentLoaded_ms:n.domContentLoadedEventEnd,load_ms:n.loadEventEnd};})};
  }
  fs.writeFileSync(path.resolve('../docs/accessibility_browser.json'),JSON.stringify(checks,null,2));
  const severe=Object.entries(checks).flatMap(([route,r]:[string,any])=>r.violations.filter((v:any)=>['serious','critical'].includes(v.impact)).map((v:any)=>`${route}: ${v.id}`));expect(severe).toEqual([]);
});
