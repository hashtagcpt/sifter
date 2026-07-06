const { TextEncoder, TextDecoder } = require('util');
Object.assign(global, { TextEncoder, TextDecoder });

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

describe('Frontend initialization and fetch interceptor', () => {
    let dom;
    let window;
    let document;
    let originalFetch;

    beforeEach((done) => {
        const html = fs.readFileSync(path.resolve(__dirname, '../static/index.html'), 'utf8');
        
        dom = new JSDOM(html, { 
            runScripts: "dangerously",
            beforeParse(window) {
                // Setup mock fetch BEFORE scripts run
                window.fetch = jest.fn().mockImplementation((url) => {
                    if (url === '/api/universe') {
                        return Promise.resolve({
                            json: () => Promise.resolve({ universe: ['AAPL', 'MSFT'] })
                        });
                    }
                    return Promise.resolve({
                        json: () => Promise.resolve({})
                    });
                });
                originalFetch = window.fetch;
            }
        });
        window = dom.window;
        document = window.document;
        
        done();
    });

    test('init() fetches ONLY the stock universe', async () => {
        // Clear mock calls
        originalFetch.mockClear();
        
        // Clear DOM for init
        document.getElementById('analysis-grid').innerHTML = 'TEST';
        
        // Call init manually to see what it does
        await window.init();
        
        // Check that originalFetch was called exactly once
        expect(originalFetch).toHaveBeenCalledTimes(1);
        
        // Check that it was called with /api/universe
        expect(originalFetch.mock.calls[0][0]).toBe('/api/universe');
        
        // Check that state was updated
        expect(window.state.universe).toEqual(['AAPL', 'MSFT']);
    });
    
    test('fetch interceptor toggles status UI correctly', async () => {
        // Use the intercepted window.fetch
        const interceptor = window.fetch;
        
        const statusEl = document.getElementById('backend-status');
        
        expect(statusEl.style.display).toBe('none');
        
        // Trigger a fetch
        const fetchPromise = interceptor('/api/dummy');
        
        // Instantly, display should be flex
        expect(statusEl.style.display).toBe('flex');
        
        // After promise resolves, display should be none
        await fetchPromise;
        
        expect(statusEl.style.display).toBe('none');
    });
});
