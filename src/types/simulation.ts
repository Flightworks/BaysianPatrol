export interface ScenarioConfig {
  // Search Area Bounds
  searchAreaWidth: number;   // NM
  searchAreaHeight: number;  // NM
  searchAreaCenterX: number; // NM
  searchAreaCenterY: number; // NM
  
  // Initial Target Datum (With spatial + temporal uncertainty)
  datumX: number;            // NM
  datumY: number;            // NM
  sigmaDatumX: number;       // Spatial position std X (NM)
  sigmaDatumY: number;       // Spatial position std Y (NM)
  
  // Theoretical Target Kinematics
  meanHeading: number;       // degrees (0 = North, 90 = East)
  meanSpeed: number;         // knots
  
  // Monte-Carlo Variability per Run
  sigmaT: number;            // Departure time std (minutes)
  sigmaHeading: number;      // Initial heading std (degrees)
  sigmaSpeed: number;        // Initial speed std (knots)
  
  // Environmental Variability per Run
  windDirection: number;     // degrees (direction wind is coming from)
  windSpeed: number;         // knots
  sigmaWindSpeed: number;    // Wind speed variability per run (knots)
  sigmaWindDirection: number;// Wind direction variability per run (degrees)
  
  // Frigate & Helicopter Departure Point Stochastic Variability per Run
  frigateX: number;          // Nominal Frigate X (NM)
  frigateY: number;          // Nominal Frigate Y (NM)
  sigmaFrigatePosition: number; // Frigate position variability per run (NM)
  helicoMaxSpeed: number;    // Max speed (knots)
  sigmaHelicoSpeed: number;  // Helicopter speed variability per run (knots)
  helicoEndurance: number;   // Total endurance (minutes)
  bingoFuelBuffer: number;   // Reserve flight buffer (minutes)
  
  // Stochastic Noise on Route
  sigmaRouteDrift: number;   // Heading drift std per dt (degrees)
  sigmaSpeedDrift: number;   // Speed variation std per dt (knots)
  
  // Radar Specs
  radarBaseRange: number;    // Nominal radar range R_0 (NM)
  
  // Simulation Controls
  gridCellSize: number;      // Maille size in NM (default 0.5 NM)
  dt: number;                // Time step in minutes (default 1.0 min)
  numIterations: number;     // Monte-Carlo N
  monteCarloSeed?: number;   // deterministic paired-run seed (default 2026)
  strategy: 'SIGMA' | 'NAIVE' | 'RL_MODEL' | 'BOTH' | 'TRIO';
}

export interface TargetState {
  x: number;
  y: number;
  speed: number;
  heading: number;
}

export type HelicopterStatus = 'SEARCHING' | 'INTERCEPTED' | 'BINGO_RETURN' | 'SAFE_RTB' | 'OUT_OF_FUEL';

export interface HelicopterState {
  x: number;
  y: number;
  heading: number;
  speed: number; // current speed in knots
  fuelRemaining: number; // minutes of flight left
  status: HelicopterStatus;
}

export interface PathPoint {
  t: number;
  x: number;
  y: number;
}

export interface HelicoPathPoint extends PathPoint {
  status: HelicopterStatus;
  radarRange: number;
}

export type GridMode = 'CLASSICAL' | 'BAYESIAN_STANDARD' | 'BAYESIAN_EVOLVED';

export interface MonteCarloRunResult {
  runId: number;
  strategy: 'SIGMA' | 'NAIVE' | 'RL_MODEL';
  intercepted: boolean;
  interceptionTime: number; // minutes
  fuelConsumed: number;     // minutes
  bingoTriggered: boolean;  // true only for an unsafe fuel violation
  safeReturn: boolean;
  outOfBounds: boolean;
  outcome: 'INTERCEPTED' | 'SAFE_RTB' | 'BINGO_VIOLATION' | 'OUT_OF_BOUNDS' | 'OUT_OF_FUEL' | 'TIME_LIMIT';
  targetPath: PathPoint[];
  helicoPath: HelicoPathPoint[];
  interceptPoint: { x: number; y: number } | null;
  
  // Specific realization parameters for this run
  runEnv: {
    windSpeed: number;
    windDirection: number;
    datumX: number;
    datumY: number;
    frigateX: number;
    frigateY: number;
    frigateSectorDeg: number;
    helicoSpeed: number;
    initialSpeed: number;
    initialHeading: number;
  };

  heatmapSnapshots: Array<{
    t: number;
    probsClassical: number[];
    probsBayesianStandard: number[];
    probsBayesianEvolved: number[];
  }>;
}

export interface StrategyStats {
  totalRuns: number;
  successCount: number;
  successRate: number;        // percentage
  meanInterceptionTime: number; // minutes
  meanFuelConsumed: number;   // minutes
  bingoCount: number;
  bingoRate: number;          // unsafe violations only
  safeReturnCount: number;
  safeReturnRate: number;
  timeHistogram: Array<{ timeBin: number; count: number }>;
  cumulativeProb: Array<{ t: number; prob: number }>;
}

export interface PairedComparisonStats {
  sigmaWins: number;
  naiveWins: number;
  ties: number;
  sigmaWinRate: number; // percentage
  meanTimeSavedMinutes: number;
  meanFuelSavedMinutes: number;
}

export interface TrioComparisonStats {
  naiveWins: number;
  sigmaWins: number;
  rlWins: number;
  ties: number;
  bestStrategy: 'NAIVE' | 'SIGMA' | 'RL_MODEL';
  naiveSuccessRate: number;
  sigmaSuccessRate: number;
  rlSuccessRate: number;
}

export interface GlobalSimulationResult {
  config: ScenarioConfig;
  sigmaStats?: StrategyStats;
  naiveStats?: StrategyStats;
  rlStats?: StrategyStats;
  pairedStats?: PairedComparisonStats;
  trioStats?: TrioComparisonStats;
  sigmaRuns: MonteCarloRunResult[];
  naiveRuns: MonteCarloRunResult[];
  rlRuns: MonteCarloRunResult[];
  executionTimeMs: number;
}

export interface GridCell {
  i: number;
  j: number;
  x: number;
  y: number;
  pPresence: number;
  pClassical: number;
  pBayesianStandard: number;
  pBayesianEvolved: number;
  pDet: number;
  scanned: boolean;
  scanMemory: number;
}
