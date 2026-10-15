import {afterEach,beforeEach,describe,it,expect,vi} from 'vitest';
import {cleanup,render,screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {MemoryRouter} from 'react-router-dom';
import App from './App';
import researchData from '../../backend/research_data.json';

beforeEach(()=>{vi.stubGlobal('fetch',vi.fn(async(url:string)=>({ok:!url.endsWith('/representation-example'),json:async()=>url.endsWith('/representation-example')?{detail:'Preserved representation unavailable in component test'}:url.endsWith('/models')?researchData.models:url.endsWith('/speakers')?[]:{...researchData,development:[],asr:{}}})));});
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
function show(path='/'){return render(<MemoryRouter initialEntries={[path]}><App/></MemoryRouter>);}
describe('Research studio interface',()=>{
  it('loads verified values and separates the reference',async()=>{show();expect(await screen.findByText('80.18%')).toBeInTheDocument();expect(screen.getByRole('heading',{name:/Every Voice/})).toBeInTheDocument();expect(screen.getByText(/frozen reference, not the development-selected/)).toBeInTheDocument();});
  it('navigates to the working studio',async()=>{show();await userEvent.click(screen.getByRole('link',{name:/Launch speech studio/}));expect(await screen.findByRole('heading',{name:'Speech studio'})).toBeInTheDocument();expect(screen.getByRole('button',{name:/Analyze speech/})).toBeDisabled();});
  it('requires audio, explicit context and acknowledgement',async()=>{show('/studio');expect(screen.getByLabelText('Explicit fold context')).toHaveValue('');expect(screen.getByRole('button',{name:/Compare available models/})).toBeDisabled();await userEvent.selectOptions(screen.getByLabelText('Explicit fold context'),'2');expect(screen.getByRole('button',{name:/Analyze speech/})).toBeDisabled();});
  it('marks original classifiers historical-only',async()=>{show('/models');expect((await screen.findAllByText('HISTORICAL ONLY')).length).toBe(2);expect(screen.getByText('Development-Selected V4')).toBeInTheDocument();});
  it('edits synthesis text without fake recognition',async()=>{show('/clear-speech');const box=await screen.findByLabelText(/Review or edit/);await userEvent.type(box,'A clear message.');expect(box).toHaveValue('A clear message.');expect(screen.getByRole('button',{name:'Generate clear speech'})).toBeDisabled();});
  it('reports missing microphone support',async()=>{show('/studio');await userEvent.click(screen.getByRole('tab',{name:'Record audio'}));await userEvent.click(screen.getByRole('button',{name:'Start recording'}));expect(await screen.findByRole('alert')).toHaveTextContent('Recording requires a supported browser');});
  it('changes architecture stages',async()=>{show('/pipeline');await userEvent.click(screen.getByRole('button',{name:'WavLM'}));expect(screen.getByRole('heading',{name:'Frozen representations or upper-layer adaptation.'})).toBeInTheDocument();await userEvent.click(screen.getByRole('button',{name:/04 Pooling/}));expect(screen.getByText(/reference produces 1,536 features/)).toBeInTheDocument();});
  it('keeps backend errors honest',async()=>{vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('Runtime offline')));show('/');expect(await screen.findByRole('alert')).toHaveTextContent('Runtime offline');expect(screen.queryByText('80.18%')).not.toBeInTheDocument();});
  it('explains the actual V3 B2 weighting without claiming equal class weights',async()=>{show('/pipeline');await userEvent.click(screen.getByRole('button',{name:/^HuBERT$/}));await userEvent.click(screen.getByRole('button',{name:/04 Fitted normalization/}));expect(screen.getByText(/clipped square-root speaker-count weights/)).toHaveTextContent('no class weighting');});
});
