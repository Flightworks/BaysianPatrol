import type { ScenarioConfig } from '../types/simulation';
import { runMonteCarloSuite } from './simulator';

self.onmessage = (e: MessageEvent) => {
  const { config } = e.data as { config: ScenarioConfig };

  try {
    const result = runMonteCarloSuite(config, (progressPercent: number) => {
      self.postMessage({
        type: 'PROGRESS',
        progress: progressPercent,
      });
    });

    self.postMessage({
      type: 'COMPLETE',
      result,
    });
  } catch (err: any) {
    self.postMessage({
      type: 'ERROR',
      error: err?.message || String(err),
    });
  }
};
