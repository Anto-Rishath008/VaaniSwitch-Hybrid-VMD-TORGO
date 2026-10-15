import '@testing-library/jest-dom/vitest';
import {vi} from 'vitest';
Object.defineProperty(window,'scrollTo',{value:vi.fn()});
class Observer{observe(){}unobserve(){}disconnect(){}}
Object.defineProperty(window,'ResizeObserver',{value:Observer});
Object.defineProperty(window,'matchMedia',{value:vi.fn().mockImplementation(()=>({matches:false,addListener(){},removeListener(){},addEventListener(){},removeEventListener(){},dispatchEvent(){}}))});
