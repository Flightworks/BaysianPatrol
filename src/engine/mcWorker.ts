import type { ScenarioConfig, GlobalSimulationResult } from '../types/simulation';
import { runMonteCarloSuite } from './simulator';

// Web Worker message handler
self.onmessage = (e: MessageEvent<{ config: ScenarioConfig }>) => {
  const { config } = e.data;
  
  try {
    const result: GlobalSimulationResult = runMonteCarloSuite(config, (progressPercent) => {
      self.postMessage({ type: 'PROGRESS', progress: progressPercent });
    });

    self.postMessage({ type: 'COMPLETE', result });
  } catch (error: any) {
    self.postMessage({ type: 'ERROR', error: error?.message || 'Simulation error' });
  }
};
