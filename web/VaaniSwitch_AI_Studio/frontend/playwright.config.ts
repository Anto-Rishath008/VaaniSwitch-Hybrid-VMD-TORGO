import {defineConfig} from '@playwright/test';
import {existsSync,readdirSync} from 'node:fs';
import path from 'node:path';
const cache=path.join(process.env.LOCALAPPDATA||'','ms-playwright');
const candidates=existsSync(cache)?readdirSync(cache).filter(n=>/^chromium-\d+$/.test(n)).sort((a,b)=>Number(b.split('-')[1])-Number(a.split('-')[1])):[];
const executable=candidates.map(n=>path.join(cache,n,'chrome-win64','chrome.exe')).find(existsSync);
const capture=path.resolve('../runtime/smoke/microphone.wav');
export default defineConfig({testDir:'./e2e',workers:1,timeout:180000,expect:{timeout:10000},retries:0,
  reporter:[['list'],['json',{outputFile:'../docs/browser_results.json'}]],outputDir:'../docs/browser-artifacts',
  use:{baseURL:'http://127.0.0.1:8000',viewport:{width:1440,height:1000},launchOptions:{executablePath:executable,args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',`--use-file-for-fake-audio-capture=${capture}`]},screenshot:'only-on-failure',trace:'retain-on-failure'}});
