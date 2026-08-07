import type { GlobalSimulationResult, ScenarioConfig, StrategyStats } from '../types/simulation';

export const RUN_HISTORY_STORAGE_KEY = 'baysian-patrol:monte-carlo-history:v1';
export const MAX_HISTORY_ENTRIES = 20;

export interface HistoryStrategySummary {
  successRate: number;
  meanInterceptionTime: number;
  meanFuelConsumed: number;
  safeReturnRate: number;
  bingoRate: number;
}

export interface RunHistoryEntry {
  id: string;
  createdAt: string;
  scenarioName: string;
  config: ScenarioConfig;
  executionTimeMs: number;
  strategies: {
    hybrid?: HistoryStrategySummary;
    bayesian?: HistoryStrategySummary;
    naive?: HistoryStrategySummary;
  };
}

const summarize = (stats?: StrategyStats): HistoryStrategySummary | undefined =>
  stats
    ? {
        successRate: stats.successRate,
        meanInterceptionTime: stats.meanInterceptionTime,
        meanFuelConsumed: stats.meanFuelConsumed,
        safeReturnRate: stats.safeReturnRate,
        bingoRate: stats.bingoRate,
      }
    : undefined;

export function createHistoryEntry(
  result: GlobalSimulationResult,
  scenarioName: string,
  createdAt = new Date().toISOString(),
  id = `${Date.now()}`,
): RunHistoryEntry {
  return {
    id,
    createdAt,
    scenarioName,
    config: { ...result.config },
    executionTimeMs: result.executionTimeMs,
    strategies: {
      hybrid: summarize(result.rlStats),
      bayesian: summarize(result.sigmaStats),
      naive: summarize(result.naiveStats),
    },
  };
}

export function prependHistory(
  history: RunHistoryEntry[],
  entry: RunHistoryEntry,
): RunHistoryEntry[] {
  return [entry, ...history.filter((item) => item.id !== entry.id)].slice(0, MAX_HISTORY_ENTRIES);
}

export function serializeHistory(history: RunHistoryEntry[]): string {
  return JSON.stringify(history.slice(0, MAX_HISTORY_ENTRIES));
}

function isHistoryEntry(value: unknown): value is RunHistoryEntry {
  if (!value || typeof value !== 'object') return false;
  const entry = value as Partial<RunHistoryEntry>;
  return (
    typeof entry.id === 'string' &&
    typeof entry.createdAt === 'string' &&
    typeof entry.scenarioName === 'string' &&
    typeof entry.executionTimeMs === 'number' &&
    !!entry.config &&
    typeof entry.config === 'object' &&
    !!entry.strategies &&
    typeof entry.strategies === 'object'
  );
}

export function parseHistory(raw: string | null): RunHistoryEntry[] {
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isHistoryEntry).slice(0, MAX_HISTORY_ENTRIES);
  } catch {
    return [];
  }
}
