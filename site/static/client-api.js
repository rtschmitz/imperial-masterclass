class ClientDataAPI {
  constructor() {
    this.worker = new Worker(new URL('./data-worker.js', import.meta.url), { type: 'module' });
    this.nextId = 1;
    this.pending = new Map();
    this.progressListeners = new Set();
    this.worker.onmessage = event => this.handleMessage(event.data);
    this.worker.onerror = event => {
      const error = new Error(event.message || 'The client-side data worker failed');
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    };
  }

  handleMessage(message) {
    if (message?.type === 'progress') {
      for (const listener of this.progressListeners) listener(message.message);
      return;
    }
    if (message?.type !== 'result') return;
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.ok) pending.resolve(message.result);
    else {
      const error = new Error(message.error?.message || 'Client-side calculation failed');
      error.workerStack = message.error?.stack;
      pending.reject(error);
    }
  }

  call(method, payload = null) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, method, payload });
    });
  }

  onProgress(listener) {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  metadata() {
    return this.call('metadata');
  }

  formulaEvent(payload) {
    return this.call('formulaEvent', payload);
  }

  formulaBatch(payload) {
    return this.call('formulaBatch', payload);
  }
}

export const clientAPI = new ClientDataAPI();
