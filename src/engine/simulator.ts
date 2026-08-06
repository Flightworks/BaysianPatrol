import type {
  ScenarioConfig,
  MonteCarloRunResult,
  StrategyStats,
  PairedComparisonStats,
  TrioComparisonStats,
  GlobalSimulationResult,
  HelicopterState,
  HelicoPathPoint
} from '../types/simulation';
import { TargetSim, type RunRealizationParams } from './targetGenerator';
import { BayesianGrid } from './bayesianGrid';
import { SIGMAPlanner } from './sigmaAlgorithm';
import { NaivePlanner } from './naiveAlgorithm';
import { RLPlanner } from './rlAlgorithm';
import { calculatePdet } from './radarModel';
import type { RadarParams } from './radarModel';
import { normalizeAngle } from './random';

/**
 * Executes a single Monte-Carlo simulation run for a given strategy and fixed realization.
 */
export async function runSingleSimulationWithRealization(
  runId: number,
  strategy: 'SIGMA' | 'NAIVE' | 'RL_MODEL',
  config: ScenarioConfig,
  realization: RunRealizationParams
): Promise<MonteCarloRunResult> {
  const {
    helicoEndurance, radarBaseRange,
    dt, meanHeading
  } = config;

  const targetSim = new TargetSim(config, realization);
  
  const runConfig: ScenarioConfig = {
    ...config,
    windSpeed: realization.windSpeed,
    windDirection: realization.windDirection,
    datumX: realization.datumX,
    datumY: realization.datumY,
    frigateX: realization.frigateX,
    frigateY: realization.frigateY,
    helicoMaxSpeed: realization.helicoSpeed,
  };

  const grid = new BayesianGrid(runConfig);

  let helico: HelicopterState = {
    x: realization.frigateX,
    y: realization.frigateY,
    heading: 0,
    speed: realization.helicoSpeed,
    fuelRemaining: helicoEndurance,
    status: 'SEARCHING',
  };

  const sigmaPlanner = strategy === 'SIGMA' ? new SIGMAPlanner(runConfig) : null;
  const naivePlanner = strategy === 'NAIVE' ? new NaivePlanner(runConfig) : null;
  const rlPlanner = strategy === 'RL_MODEL' ? new RLPlanner(runConfig) : null;

  if (rlPlanner) {
    await rlPlanner.init();
  }

  const helicoPath: HelicoPathPoint[] = [];
  const heatmapSnapshots: Array<{
    t: number;
    probsClassical: number[];
    probsBayesianStandard: number[];
    probsBayesianEvolved: number[];
  }> = [];

  let intercepted = false;
  let interceptionTime = 0;
  let bingoTriggered = false;
  let interceptPoint: { x: number; y: number } | null = null;

  const maxSteps = Math.ceil(helicoEndurance / dt);
  const radarParams: RadarParams = {
    baseRange: radarBaseRange,
    windSpeed: realization.windSpeed,
    windDirection: realization.windDirection,
  };

  const storeSnapshots = runId <= 30;

  helicoPath.push({
    t: 0,
    x: helico.x,
    y: helico.y,
    status: helico.status,
    radarRange: radarBaseRange,
  });

  if (storeSnapshots) {
    heatmapSnapshots.push({
      t: 0,
      probsClassical: grid.getFlatClassicalProbs(),
      probsBayesianStandard: grid.getFlatBayesianStandardProbs(),
      probsBayesianEvolved: grid.getFlatBayesianEvolvedProbs(),
    });
  }

  for (let step = 1; step <= maxSteps; step++) {
    const t = step * dt;

    const targetPoint = targetSim.step(t, dt);

    grid.updatePriorDensity(t);

    if (strategy === 'SIGMA') {
      helico = sigmaPlanner!.planStep(helico, grid, t, dt);
    } else if (strategy === 'NAIVE') {
      helico = naivePlanner!.planStep(helico, t, dt);
    } else if (strategy === 'RL_MODEL') {
      helico = await rlPlanner!.planStepAsync(helico, grid, t, dt);
    }

    if (helico.status === 'BINGO_RETURN') {
      bingoTriggered = true;
    }

    const targetBearingDeg = normalizeAngle(
      (Math.atan2(targetPoint.x - helico.x, targetPoint.y - helico.y) * 180.0) / Math.PI
    );
    const distToTarget = Math.hypot(targetPoint.x - helico.x, targetPoint.y - helico.y);
    
    const aspectApproachRad = ((helico.heading - meanHeading) * Math.PI) / 180.0;
    const perpApproachMultiplier = 0.70 + 0.50 * Math.abs(Math.sin(aspectApproachRad));
    const pDetTarget = Math.min(0.98, calculatePdet(distToTarget, targetBearingDeg, meanHeading, radarParams) * perpApproachMultiplier);

    helicoPath.push({
      t,
      x: helico.x,
      y: helico.y,
      status: helico.status,
      radarRange: radarBaseRange,
    });

    if (pDetTarget > 0.05) {
      if (Math.random() < pDetTarget || distToTarget < 0.8) {
        intercepted = true;
        interceptionTime = t;
        interceptPoint = { x: targetPoint.x, y: targetPoint.y };
        helico.status = 'INTERCEPTED';
        if (storeSnapshots) {
          heatmapSnapshots.push({
            t,
            probsClassical: grid.getFlatClassicalProbs(),
            probsBayesianStandard: grid.getFlatBayesianStandardProbs(),
            probsBayesianEvolved: grid.getFlatBayesianEvolvedProbs(),
          });
        }
        break;
      }
    }

    grid.updateBayesianScan(helico.x, helico.y, helico.heading, radarParams);

    if (storeSnapshots && step % 2 === 0) {
      heatmapSnapshots.push({
        t,
        probsClassical: grid.getFlatClassicalProbs(),
        probsBayesianStandard: grid.getFlatBayesianStandardProbs(),
        probsBayesianEvolved: grid.getFlatBayesianEvolvedProbs(),
      });
    }

    if (helico.status === 'OUT_OF_FUEL' || (helico.status === 'BINGO_RETURN' && Math.hypot(realization.frigateX - helico.x, realization.frigateY - helico.y) < 0.2)) {
      break;
    }
  }

  const fuelConsumed = helicoEndurance - Math.max(0, helico.fuelRemaining);

  return {
    runId,
    strategy,
    intercepted,
    interceptionTime: intercepted ? interceptionTime : helicoEndurance,
    fuelConsumed,
    bingoTriggered,
    targetPath: targetSim.getPath(),
    helicoPath,
    interceptPoint,
    runEnv: {
      windSpeed: realization.windSpeed,
      windDirection: realization.windDirection,
      datumX: realization.datumX,
      datumY: realization.datumY,
      frigateX: realization.frigateX,
      frigateY: realization.frigateY,
      frigateSectorDeg: realization.frigateSectorDeg,
      helicoSpeed: realization.helicoSpeed,
      initialSpeed: realization.initialSpeed,
      initialHeading: realization.initialHeading,
    },
    heatmapSnapshots,
  };
}

/**
 * Computes global strategy stats.
 */
export function computeStrategyStats(runs: MonteCarloRunResult[]): StrategyStats {
  const totalRuns = runs.length;
  if (totalRuns === 0) {
    return {
      totalRuns: 0,
      successCount: 0,
      successRate: 0,
      meanInterceptionTime: 0,
      meanFuelConsumed: 0,
      bingoCount: 0,
      bingoRate: 0,
      timeHistogram: [],
      cumulativeProb: [],
    };
  }

  const successfulRuns = runs.filter(r => r.intercepted);
  const successCount = successfulRuns.length;
  const successRate = (successCount / totalRuns) * 100.0;

  const meanInterceptionTime = successCount > 0
    ? successfulRuns.reduce((acc, r) => acc + r.interceptionTime, 0) / successCount
    : 0;

  const meanFuelConsumed = runs.reduce((acc, r) => acc + r.fuelConsumed, 0) / totalRuns;
  const bingoCount = runs.filter(r => r.bingoTriggered).length;
  const bingoRate = (bingoCount / totalRuns) * 100.0;

  const timeBins = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180];
  const timeHistogram = timeBins.map(bin => {
    const count = successfulRuns.filter(r => r.interceptionTime <= bin && r.interceptionTime > bin - 15).length;
    return { timeBin: bin, count };
  });

  const cumulativeProb: Array<{ t: number; prob: number }> = [];
  for (let t = 10; t <= 180; t += 10) {
    const detectedByT = successfulRuns.filter(r => r.interceptionTime <= t).length;
    cumulativeProb.push({ t, prob: (detectedByT / totalRuns) * 100.0 });
  }

  return {
    totalRuns,
    successCount,
    successRate,
    meanInterceptionTime,
    meanFuelConsumed,
    bingoCount,
    bingoRate,
    timeHistogram,
    cumulativeProb,
  };
}

/**
 * Computes paired comparison KPIs between SIGMA and Naïve on exact identical realizations.
 */
export function computePairedStats(
  sigmaRuns: MonteCarloRunResult[],
  naiveRuns: MonteCarloRunResult[]
): PairedComparisonStats {
  const N = Math.min(sigmaRuns.length, naiveRuns.length);
  if (N === 0) {
    return {
      sigmaWins: 0,
      naiveWins: 0,
      ties: 0,
      sigmaWinRate: 0,
      meanTimeSavedMinutes: 0,
      meanFuelSavedMinutes: 0,
    };
  }

  let sigmaWins = 0;
  let naiveWins = 0;
  let ties = 0;
  let totalTimeSaved = 0;
  let totalFuelSaved = 0;

  for (let k = 0; k < N; k++) {
    const sigma = sigmaRuns[k];
    const naive = naiveRuns[k];

    if (sigma.intercepted && !naive.intercepted) {
      sigmaWins++;
    } else if (!sigma.intercepted && naive.intercepted) {
      naiveWins++;
    } else if (sigma.intercepted && naive.intercepted) {
      if (sigma.interceptionTime < naive.interceptionTime - 1.0) {
        sigmaWins++;
      } else if (naive.interceptionTime < sigma.interceptionTime - 1.0) {
        naiveWins++;
      } else {
        ties++;
      }
    } else {
      ties++;
    }

    totalTimeSaved += (naive.interceptionTime - sigma.interceptionTime);
    totalFuelSaved += (naive.fuelConsumed - sigma.fuelConsumed);
  }

  return {
    sigmaWins,
    naiveWins,
    ties,
    sigmaWinRate: (sigmaWins / N) * 100.0,
    meanTimeSavedMinutes: totalTimeSaved / N,
    meanFuelSavedMinutes: totalFuelSaved / N,
  };
}

/**
 * Computes 3-way stats between Naive, SIGMA, and RL_MODEL.
 */
export function computeTrioStats(
  naiveRuns: MonteCarloRunResult[],
  sigmaRuns: MonteCarloRunResult[],
  rlRuns: MonteCarloRunResult[]
): TrioComparisonStats {
  const N = Math.min(naiveRuns.length, sigmaRuns.length, rlRuns.length);
  if (N === 0) {
    return {
      naiveWins: 0,
      sigmaWins: 0,
      rlWins: 0,
      ties: 0,
      bestStrategy: 'SIGMA',
      naiveSuccessRate: 0,
      sigmaSuccessRate: 0,
      rlSuccessRate: 0,
    };
  }

  let naiveWins = 0;
  let sigmaWins = 0;
  let rlWins = 0;
  let ties = 0;

  for (let k = 0; k < N; k++) {
    const naive = naiveRuns[k];
    const sigma = sigmaRuns[k];
    const rl = rlRuns[k];

    const times = [
      { strat: 'NAIVE' as const, intercepted: naive.intercepted, t: naive.interceptionTime },
      { strat: 'SIGMA' as const, intercepted: sigma.intercepted, t: sigma.interceptionTime },
      { strat: 'RL_MODEL' as const, intercepted: rl.intercepted, t: rl.interceptionTime },
    ];

    const valid = times.filter(x => x.intercepted);
    if (valid.length === 0) {
      ties++;
    } else {
      valid.sort((a, b) => a.t - b.t);
      if (valid.length > 1 && Math.abs(valid[0].t - valid[1].t) < 0.5) {
        ties++;
      } else if (valid[0].strat === 'NAIVE') {
        naiveWins++;
      } else if (valid[0].strat === 'SIGMA') {
        sigmaWins++;
      } else {
        rlWins++;
      }
    }
  }

  const naiveSuccessRate = (naiveRuns.filter(r => r.intercepted).length / N) * 100;
  const sigmaSuccessRate = (sigmaRuns.filter(r => r.intercepted).length / N) * 100;
  const rlSuccessRate = (rlRuns.filter(r => r.intercepted).length / N) * 100;

  let bestStrategy: 'NAIVE' | 'SIGMA' | 'RL_MODEL' = 'SIGMA';
  if (rlSuccessRate >= sigmaSuccessRate && rlSuccessRate >= naiveSuccessRate) {
    bestStrategy = 'RL_MODEL';
  } else if (sigmaSuccessRate >= naiveSuccessRate) {
    bestStrategy = 'SIGMA';
  } else {
    bestStrategy = 'NAIVE';
  }

  return {
    naiveWins,
    sigmaWins,
    rlWins,
    ties,
    bestStrategy,
    naiveSuccessRate,
    sigmaSuccessRate,
    rlSuccessRate,
  };
}

/**
 * Runs complete Paired / Trio Monte-Carlo simulation suite with stochastic environmental and omnidirectional frigate sector variability.
 */
export async function runMonteCarloSuite(
  config: ScenarioConfig,
  onProgress?: (progressPercent: number) => void
): Promise<GlobalSimulationResult> {
  const startTime = performance.now();
  const N = config.numIterations;

  const sigmaRuns: MonteCarloRunResult[] = [];
  const naiveRuns: MonteCarloRunResult[] = [];
  const rlRuns: MonteCarloRunResult[] = [];

  const isTrio = config.strategy === 'TRIO';
  const runSIGMA = config.strategy === 'SIGMA' || config.strategy === 'BOTH' || isTrio;
  const runNaive = config.strategy === 'NAIVE' || config.strategy === 'BOTH' || isTrio;
  const runRL = config.strategy === 'RL_MODEL' || isTrio;

  const totalTasks = N * ((runSIGMA ? 1 : 0) + (runNaive ? 1 : 0) + (runRL ? 1 : 0));
  let completed = 0;

  for (let k = 1; k <= N; k++) {
    const targetSimHelper = new TargetSim(config);
    const realization = targetSimHelper.getRealization();

    if (runSIGMA) {
      sigmaRuns.push(await runSingleSimulationWithRealization(k, 'SIGMA', config, realization));
      completed++;
      if (onProgress && completed % 10 === 0) {
        onProgress((completed / totalTasks) * 100);
      }
    }

    if (runNaive) {
      naiveRuns.push(await runSingleSimulationWithRealization(k, 'NAIVE', config, realization));
      completed++;
      if (onProgress && completed % 10 === 0) {
        onProgress((completed / totalTasks) * 100);
      }
    }

    if (runRL) {
      rlRuns.push(await runSingleSimulationWithRealization(k, 'RL_MODEL', config, realization));
      completed++;
      if (onProgress && completed % 10 === 0) {
        onProgress((completed / totalTasks) * 100);
      }
    }
  }

  const sigmaStats = runSIGMA ? computeStrategyStats(sigmaRuns) : undefined;
  const naiveStats = runNaive ? computeStrategyStats(naiveRuns) : undefined;
  const rlStats = runRL ? computeStrategyStats(rlRuns) : undefined;
  const pairedStats = (runSIGMA && runNaive) ? computePairedStats(sigmaRuns, naiveRuns) : undefined;
  const trioStats = (isTrio) ? computeTrioStats(naiveRuns, sigmaRuns, rlRuns) : undefined;

  const executionTimeMs = performance.now() - startTime;

  return {
    config,
    sigmaStats,
    naiveStats,
    rlStats,
    pairedStats,
    trioStats,
    sigmaRuns,
    naiveRuns,
    rlRuns,
    executionTimeMs,
  };
}
