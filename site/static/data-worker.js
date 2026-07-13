import { StaticDataEngine } from './data-engine.js';

const engine = new StaticDataEngine(
  new URL('../data/', import.meta.url),
  message => self.postMessage({ type: 'progress', message })
);

const methods = {
  metadata: () => engine.metadata(),
  formulaEvent: payload => engine.formulaEvent(payload),
  formulaBatch: payload => engine.formulaBatch(payload)
};

self.onmessage = async event => {
  const { id, method, payload } = event.data || {};
  try {
    if (!(method in methods)) throw new Error(`Unknown client data method: ${method}`);
    const result = await methods[method](payload);
    self.postMessage({ type: 'result', id, ok: true, result });
  } catch (error) {
    self.postMessage({
      type: 'result',
      id,
      ok: false,
      error: {
        message: error?.message || String(error),
        stack: error?.stack || ''
      }
    });
  }
};
