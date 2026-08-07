import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createHistoryEntry,
  parseHistory,
  prependHistory,
  serializeHistory,
} from '../src/engine/runHistory.ts';
import type { GlobalSimulationResult, ScenarioConfig, StrategyStats } from '../src/types/simulation.ts';

const config: ScenarioConfig = {
  searchAreaWidth: 60,
  searchAreaHeight: 60,
  searchAreaCenterX: 0,
  searchAreaCenterY: 0,
  datumX: -15,
  datumY: -15,
  sigmaDatumX: 3,
  sigmaDatumY: 3,
  meanHeading: 45,
  meanSpeed: 16,
  sigmaT: 10,
  sigmaHeading: 15,
  sigmaSpeed: 4,
  windDirection: 270,
  windSpeed: 15,
  sigmaWindSpeed: 4,
  sigmaWindDirection: 15,
  frigateX: 20,
  frigateY: -15,
  sigmaFrigatePosition: 3,
  helicoMaxSpeed: 130,
  sigmaHelicoSpeed: 5,
  helicoEndurance: 180,
  bingoFuelBuffer: 20,
  sigmaRouteDrift: 3,
  sigmaSpeedDrift: 1,
  radarBaseRange: 12,
  gridCellSize: 0.5,
  dt: 1,
  numIterations: 250,
  monteCarloSeed: 2026,
  strategy: 'TRIO',
};

const stats = (successRate: number, time: number): StrategyStats => ({
  totalRuns: 250,
  successCount: Math.round(successRate * 2.5),
  successRate,
  meanInterceptionTime: time,
  meanFuelConsumed: time,
  bingoCount: 0,
  bingoRate: 0,
  safeReturnCount: 0,
  safeReturnRate: 0,
  timeHistogram: [],
  cumulativeProb: [],
});

const result: GlobalSimulationResult = {
  config,
  sigmaStats: stats(93.6, 16.4),
  naiveStats: stats(100, 16.3),
  rlStats: stats(100, 12),
  sigmaRuns: [],
  naiveRuns: [],
  rlRuns: [],
  executionTimeMs: 1234,
};

test('history entry keeps parameters and compact strategy summaries', () => {
  const entry = createHistoryEntry(result, 'Scénario standard', '2026-08-06T20:00:00.000Z', 'run-1');

  assert.equal(entry.id, 'run-1');
  assert.equal(entry.scenarioName, 'Scénario standard');
  assert.equal(entry.config.numIterations, 250);
  assert.equal(entry.strategies.hybrid?.successRate, 100);
  assert.equal(entry.strategies.bayesian?.meanInterceptionTime, 16.4);
  assert.equal(entry.strategies.naive?.successRate, 100);
  assert.equal('sigmaRuns' in entry, false);
});

test('history prepends newest entries and keeps only the latest twenty', () => {
  const entries = Array.from({ length: 20 }, (_, index) =>
    createHistoryEntry(result, `Scénario ${index}`, `2026-08-06T20:${String(index).padStart(2, '0')}:00.000Z`, `run-${index}`),
  );
  const newest = createHistoryEntry(result, 'Nouveau', '2026-08-06T21:00:00.000Z', 'run-new');

  const updated = prependHistory(entries, newest);

  assert.equal(updated.length, 20);
  assert.equal(updated[0].id, 'run-new');
  assert.equal(updated.some((entry) => entry.id === 'run-19'), false);
});

test('history serialization round-trips and corrupt storage recovers empty', () => {
  const entry = createHistoryEntry(result, 'Standard', '2026-08-06T20:00:00.000Z', 'run-1');

  assert.deepEqual(parseHistory(serializeHistory([entry])), [entry]);
  assert.deepEqual(parseHistory('{not-json'), []);
  assert.deepEqual(parseHistory(null), []);
});
