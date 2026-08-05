import type {
  ScenarioConfig,
  MonteCarloRunResult,
  StrategyStats,
  PairedComparisonStats,
  GlobalSimulationResult,
  HelicopterState,
  HelicoPathPoint
} from '../types/simulation';
import { TargetSim, type RunRealizationParams } from './targetGenerator';
import { BayesianGrid } from './bayesianGrid';
import { AMIPlanner } from './amiAlgorithm';
import { NaivePlanner } from './naiveAlgorithm';
import { calculatePdet } from './radarModel';
import type { RadarParams } from './radarModel';
import { normalizeAngle } from './random';

/**
 * Executes a single Monte-Carlo simulation run for a given strategy and fixed realization.
 */
export function runSingleSimulationWithRealization(
  runId: number,
  strategy: 'AMI' | 'NAIVE',
  config: ScenarioConfig,
  realization: RunRealizationParams
): MonteCarloRunResult {
  const {
    helicoEndurance, radarBaseRange,
    dt, meanHeading
  } = config;

  const targetSim = new TargetSim(config, realization);
  
  // Custom scenario config for this specific run's environment & frigate departure point
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
    fuelRemaining: helicoEndurance,
    status: 'SEARCHING',
  };

  const amiPlanner = strategy === 'AMI' ? new AMIPlanner(runConfig) : null;
  const naivePlanner = strategy === 'NAIVE' ? new NaivePlanner(runConfig) : null;

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

    if (strategy === 'AMI') {
      helico = amiPlanner!.planStep(helico, grid, t, dt);
    } else {
      helico = naivePlanner!.planStep(helico, t, dt);
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
 * Computes paired comparison KPIs between AMI and Naïve on exact identical realizations.
 */
export function computePairedStats(
  amiRuns: MonteCarloRunResult[],
  naiveRuns: MonteCarloRunResult[]
): PairedComparisonStats {
  const N = Math.min(amiRuns.length, naiveRuns.length);
  if (N === 0) {
    return {
      amiWins: 0,
      naiveWins: 0,
      ties: 0,
      amiWinRate: 0,
      meanTimeSavedMinutes: 0,
      meanFuelSavedMinutes: 0,
    };
  }

  let amiWins = 0;
  let naiveWins = 0;
  let ties = 0;
  let totalTimeSaved = 0;
  let totalFuelSaved = 0;

  for (let k = 0; k < N; k++) {
    const ami = amiRuns[k];
    const naive = naiveRuns[k];

    if (ami.intercepted && !naive.intercepted) {
      amiWins++;
    } else if (!ami.intercepted && naive.intercepted) {
      naiveWins++;
    } else if (ami.intercepted && naive.intercepted) {
      if (ami.interceptionTime < naive.interceptionTime - 1.0) {
        amiWins++;
      } else if (naive.interceptionTime < ami.interceptionTime - 1.0) {
        naiveWins++;
      } else {
        ties++;
      }
    } else {
      ties++;
    }

    totalTimeSaved += (naive.interceptionTime - ami.interceptionTime);
    totalFuelSaved += (naive.fuelConsumed - ami.fuelConsumed);
  }

  return {
    amiWins,
    naiveWins,
    ties,
    amiWinRate: (amiWins / N) * 100.0,
    meanTimeSavedMinutes: totalTimeSaved / N,
    meanFuelSavedMinutes: totalFuelSaved / N,
  };
}

/**
 * Runs complete Paired Monte-Carlo simulation suite with stochastic environmental and omnidirectional frigate sector variability.
 */
export function runMonteCarloSuite(
  config: ScenarioConfig,
  onProgress?: (progressPercent: number) => void
): GlobalSimulationResult {
  const startTime = performance.now();
  const N = config.numIterations;

  const amiRuns: MonteCarloRunResult[] = [];
  const naiveRuns: MonteCarloRunResult[] = [];

  const runAMI = config.strategy === 'AMI' || config.strategy === 'BOTH';
  const runNaive = config.strategy === 'NAIVE' || config.strategy === 'BOTH';

  const totalTasks = N * ((runAMI ? 1 : 0) + (runNaive ? 1 : 0));
  let completed = 0;

  for (let k = 1; k <= N; k++) {
    const targetSimHelper = new TargetSim(config);
    const realization = targetSimHelper.getRealization();

    if (runAMI) {
      amiRuns.push(runSingleSimulationWithRealization(k, 'AMI', config, realization));
      completed++;
      if (onProgress && completed % 10 === 0) {
        onProgress((completed / totalTasks) * 100);
      }
    }

    if (runNaive) {
      naiveRuns.push(runSingleSimulationWithRealization(k, 'NAIVE', config, realization));
      completed++;
      if (onProgress && completed % 10 === 0) {
        onProgress((completed / totalTasks) * 100);
      }
    }
  }

  const amiStats = runAMI ? computeStrategyStats(amiRuns) : undefined;
  const naiveStats = runNaive ? computeStrategyStats(naiveRuns) : undefined;
  const pairedStats = (runAMI && runNaive) ? computePairedStats(amiRuns, naiveRuns) : undefined;

  const executionTimeMs = performance.now() - startTime;

  return {
    config,
    amiStats,
    naiveStats,
    pairedStats,
    amiRuns,
    naiveRuns,
    executionTimeMs,
  };
}
