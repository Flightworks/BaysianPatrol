export interface BeliefAction {
  id: string;
  detectionLikelihoods: readonly number[];
  travelCost: number;
}

export interface BeliefOutcome {
  detectionProbability: number;
  noDetectionProbability: number;
  noDetectionPosterior: number[];
  priorEntropy: number;
  noDetectionEntropy: number;
  expectedPosteriorEntropy: number;
  expectedInformationGain: number;
}

export interface BeliefActionChoice {
  action: BeliefAction;
  outcome: BeliefOutcome;
  score: number;
}

export interface BeliefActionWeights {
  detectionReward: number;
  informationReward: number;
  travelPenalty: number;
}

const DEFAULT_WEIGHTS: BeliefActionWeights = {
  detectionReward: 1,
  informationReward: 0.35,
  travelPenalty: 0.08,
};

const MAX_DIRECTIONAL_CANDIDATES = 12;
const MAX_MODE_CANDIDATES = 6;
const MAX_TOTAL_ACTIONS = MAX_DIRECTIONAL_CANDIDATES + MAX_MODE_CANDIDATES;
const MAX_SAMPLES_PER_ACTION = 12;
const MAX_BELIEF_POINTS_PER_AXIS = 20;
const MAX_RAW_BELIEF_CELLS = 100_000;

function requireFinite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`);
  return value;
}

function requireNonNegative(value: number, label: string): number {
  requireFinite(value, label);
  if (value < 0) throw new Error(`${label} must be non-negative`);
  return value;
}

function requireProbability(value: number, label: string): number {
  requireFinite(value, label);
  if (value < 0 || value > 1) throw new Error(`${label} must be between zero and one`);
  return value;
}

function normalize(values: readonly number[]): number[] {
  const nonNegative = values.map((value, index) => requireNonNegative(value, `Belief value ${index}`));
  const total = nonNegative.reduce((sum, value) => sum + value, 0);
  requireFinite(total, 'Belief total mass');
  if (total <= 0) throw new Error('Belief must contain positive mass');
  return nonNegative.map(value => value / total);
}

function logSumExp(logValues: readonly number[]): number {
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of logValues) maximum = Math.max(maximum, value);
  if (!Number.isFinite(maximum)) return Number.NEGATIVE_INFINITY;
  const scaledSum = logValues.reduce(
    (sum, value) => sum + (Number.isFinite(value) ? Math.exp(value - maximum) : 0),
    0,
  );
  return maximum + Math.log(scaledSum);
}

export function combineIndependentDetectionLikelihoods(
  likelihoods: readonly number[],
): number {
  let logNoDetectionProbability = 0;
  for (const [index, likelihood] of likelihoods.entries()) {
    const probability = requireProbability(likelihood, `Detection likelihood ${index}`);
    if (probability === 1) return 1;
    logNoDetectionProbability += Math.log1p(-probability);
  }
  return -Math.expm1(logNoDetectionProbability);
}

function entropy(probabilities: readonly number[]): number {
  return probabilities.reduce(
    (sum, probability) => probability > 0 ? sum - probability * Math.log(probability) : sum,
    0,
  );
}

/**
 * Binary observation model for one search action:
 * z ∈ {detection, no detection}.
 * A detection terminates the search, so its posterior entropy is zero.
 */
export function evaluateBeliefOutcome(
  beliefInput: readonly number[],
  detectionLikelihoods: readonly number[],
): BeliefOutcome {
  if (beliefInput.length === 0 || beliefInput.length !== detectionLikelihoods.length) {
    throw new Error('Belief and detection likelihoods must have the same non-zero length');
  }
  if (beliefInput.length > MAX_BELIEF_POINTS_PER_AXIS * MAX_BELIEF_POINTS_PER_AXIS) {
    throw new Error('Belief outcome evaluation accepts at most 400 states');
  }

  const belief = normalize(beliefInput);
  const likelihoods = detectionLikelihoods.map((value, index) =>
    requireProbability(value, `Detection likelihood ${index}`));
  const logNoDetectionMasses = belief.map((probability, index) => {
    const missProbability = 1 - likelihoods[index];
    return probability > 0 && missProbability > 0
      ? Math.log(probability) + Math.log(missProbability)
      : Number.NEGATIVE_INFINITY;
  });
  const maximumLogMass = Math.max(...logNoDetectionMasses);
  const logNoDetectionProbability = logSumExp(logNoDetectionMasses);
  const logDetectionProbability = logSumExp(belief.map((probability, index) => {
    const detectionProbability = likelihoods[index];
    return probability > 0 && detectionProbability > 0
      ? Math.log(probability) + Math.log(detectionProbability)
      : Number.NEGATIVE_INFINITY;
  }));
  let noDetectionProbability = 0;
  let noDetectionPosterior: number[];
  if (Number.isFinite(maximumLogMass)) {
    const scaledMasses = logNoDetectionMasses.map(logMass =>
      Number.isFinite(logMass) ? Math.exp(logMass - maximumLogMass) : 0);
    const scaledMass = scaledMasses.reduce((sum, mass) => sum + mass, 0);
    noDetectionPosterior = scaledMasses.map(mass => mass / scaledMass);
    noDetectionProbability = Math.exp(logNoDetectionProbability);
  } else {
    // Every state detects with certainty, so this observation branch is impossible.
    noDetectionPosterior = [...belief];
  }
  const detectionProbability = Number.isFinite(logDetectionProbability)
    ? Math.exp(logDetectionProbability)
    : 0;
  const priorEntropy = entropy(belief);
  const noDetectionEntropy = entropy(noDetectionPosterior);
  const expectedPosteriorEntropy = noDetectionProbability * noDetectionEntropy;

  return {
    detectionProbability,
    noDetectionProbability,
    noDetectionPosterior,
    priorEntropy,
    noDetectionEntropy,
    expectedPosteriorEntropy,
    expectedInformationGain: Math.max(0, priorEntropy - expectedPosteriorEntropy),
  };
}

export function chooseBeliefAction(
  belief: readonly number[],
  actions: readonly BeliefAction[],
  weights: BeliefActionWeights = DEFAULT_WEIGHTS,
): BeliefActionChoice {
  if (actions.length === 0) throw new Error('At least one belief action is required');
  requireNonNegative(weights.detectionReward, 'Detection reward');
  requireNonNegative(weights.informationReward, 'Information reward');
  requireNonNegative(weights.travelPenalty, 'Travel penalty');

  let best: BeliefActionChoice | null = null;
  for (const action of actions) {
    requireNonNegative(action.travelCost, `Travel cost for action ${action.id}`);
    const outcome = evaluateBeliefOutcome(belief, action.detectionLikelihoods);
    const score = weights.detectionReward * outcome.detectionProbability
      + weights.informationReward * outcome.expectedInformationGain
      - weights.travelPenalty * Math.max(0, action.travelCost);
    requireFinite(score, `Action ${action.id} score`);
    if (!best || score > best.score + 1e-12 || (
      Math.abs(score - best.score) <= 1e-12 && action.id < best.action.id
    )) {
      best = { action, outcome, score };
    }
  }
  return best!;
}

export interface BeliefPoint {
  x: number;
  y: number;
  probability: number;
}

export interface SpatialPoint {
  x: number;
  y: number;
}

export interface BeliefSearchBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface BeliefWaypointOptions {
  bounds: BeliefSearchBounds;
  actionRadius: number;
  directionalCandidates: number;
  modeCandidates: number;
  sampleSpacing: number;
  maxSamplesPerAction?: number;
}

export interface BeliefWaypointChoice extends BeliefActionChoice {
  waypoint: SpatialPoint;
}

export type SpatialDetectionModel = (
  sample: SpatialPoint,
  cell: BeliefPoint,
  headingDeg: number,
) => number;

interface SpatialBeliefAction extends BeliefAction {
  endpoint: SpatialPoint;
}

const clamp = (value: number, low: number, high: number): number =>
  Math.max(low, Math.min(high, value));

function uniqueEndpoints(points: readonly SpatialPoint[]): SpatialPoint[] {
  const unique: SpatialPoint[] = [];
  for (const point of points) {
    if (!unique.some(existing => Math.hypot(existing.x - point.x, existing.y - point.y) < 1e-6)) {
      unique.push(point);
    }
  }
  return unique;
}

export function aggregateBeliefGrid<T>(
  grid: readonly (readonly T[])[],
  maximumPointsPerAxis: number = 20,
  readPoint: (cell: T) => BeliefPoint = cell => cell as BeliefPoint,
): BeliefPoint[] {
  requireFinite(maximumPointsPerAxis, 'Belief aggregation axis budget');
  if (!Number.isInteger(maximumPointsPerAxis)) {
    throw new Error('Belief aggregation axis budget must be an integer');
  }
  if (maximumPointsPerAxis < 1) {
    throw new Error('Belief aggregation axis budget must be positive');
  }
  if (grid.length > MAX_RAW_BELIEF_CELLS) {
    throw new Error(`Belief aggregation accepts at most ${MAX_RAW_BELIEF_CELLS} rows`);
  }
  if (grid.length === 0) return [];
  let rawCellCount = 0;
  let maximumWidth = 0;
  for (const row of grid) {
    rawCellCount += row.length;
    if (rawCellCount > MAX_RAW_BELIEF_CELLS) {
      throw new Error(`Belief aggregation accepts at most ${MAX_RAW_BELIEF_CELLS} raw cells`);
    }
    maximumWidth = Math.max(maximumWidth, row.length);
  }
  const axisBudget = Math.min(
    MAX_BELIEF_POINTS_PER_AXIS,
    Math.floor(maximumPointsPerAxis),
  );
  const blockSize = Math.max(1, Math.ceil(Math.max(grid.length, maximumWidth) / axisBudget));
  const points: BeliefPoint[] = [];
  for (let rowStart = 0; rowStart < grid.length; rowStart += blockSize) {
    for (let columnStart = 0; columnStart < maximumWidth; columnStart += blockSize) {
      let probability = 0;
      let weightedX = 0;
      let weightedY = 0;
      for (let row = rowStart; row < Math.min(grid.length, rowStart + blockSize); row += 1) {
        for (let column = columnStart; column < Math.min(grid[row].length, columnStart + blockSize); column += 1) {
          const point = readPoint(grid[row][column]);
          requireFinite(point.x, 'Belief point x');
          requireFinite(point.y, 'Belief point y');
          const mass = requireNonNegative(point.probability, 'Belief point probability');
          probability = requireFinite(probability + mass, 'Aggregated belief mass');
          weightedX = requireFinite(weightedX + point.x * mass, 'Aggregated belief x');
          weightedY = requireFinite(weightedY + point.y * mass, 'Aggregated belief y');
        }
      }
      if (probability > 0) {
        points.push({
          x: requireFinite(weightedX / probability, 'Aggregated belief centroid x'),
          y: requireFinite(weightedY / probability, 'Aggregated belief centroid y'),
          probability,
        });
      }
    }
  }
  return points;
}

/**
 * Finite candidate approximation of belief-space actions. Candidates include
 * short exploratory headings and direct transits toward the strongest belief
 * modes; none is constrained to an IAMSAR sweep orientation.
 */
export function chooseBeliefWaypoint(
  cells: readonly BeliefPoint[],
  current: SpatialPoint,
  detectionAt: SpatialDetectionModel,
  options: BeliefWaypointOptions,
  weights?: BeliefActionWeights,
): BeliefWaypointChoice {
  if (cells.length === 0) throw new Error('At least one belief cell is required');
  if (cells.length > MAX_BELIEF_POINTS_PER_AXIS * MAX_BELIEF_POINTS_PER_AXIS) {
    throw new Error('Belief waypoint planning accepts at most 400 cells');
  }
  const { bounds } = options;
  for (const [label, value] of Object.entries(bounds)) requireFinite(value, `Search bound ${label}`);
  if (!(bounds.minX < bounds.maxX && bounds.minY < bounds.maxY)) {
    throw new Error('Search bounds must have positive dimensions');
  }
  requireFinite(current.x, 'Current x');
  requireFinite(current.y, 'Current y');
  requireFinite(options.actionRadius, 'Action radius');
  requireFinite(options.sampleSpacing, 'Sample spacing');
  requireFinite(options.directionalCandidates, 'Directional candidate count');
  requireFinite(options.modeCandidates, 'Mode candidate count');
  requireFinite(options.maxSamplesPerAction ?? MAX_SAMPLES_PER_ACTION, 'Sample count');
  if (!Number.isInteger(options.directionalCandidates)
    || !Number.isInteger(options.modeCandidates)
    || !Number.isInteger(options.maxSamplesPerAction ?? MAX_SAMPLES_PER_ACTION)) {
    throw new Error('Planner candidate and sample counts must be integers');
  }
  if (options.actionRadius <= 0 || options.sampleSpacing <= 0
    || options.directionalCandidates < 1 || options.modeCandidates < 1
    || (options.maxSamplesPerAction ?? MAX_SAMPLES_PER_ACTION) < 1) {
    throw new Error('Planner geometry and candidate counts must be positive');
  }
  for (const cell of cells) {
    requireFinite(cell.x, 'Belief cell x');
    requireFinite(cell.y, 'Belief cell y');
    requireNonNegative(cell.probability, 'Belief cell probability');
  }
  const endpoints: SpatialPoint[] = [];
  const directionCount = Math.min(
    MAX_DIRECTIONAL_CANDIDATES,
    Math.floor(options.directionalCandidates),
  );
  for (let index = 0; index < directionCount; index += 1) {
    const headingRad = (index * 2 * Math.PI) / directionCount;
    endpoints.push({
      x: clamp(current.x + Math.sin(headingRad) * options.actionRadius, bounds.minX, bounds.maxX),
      y: clamp(current.y + Math.cos(headingRad) * options.actionRadius, bounds.minY, bounds.maxY),
    });
  }

  const strongestModes = [...cells]
    .sort((a, b) => b.probability - a.probability || a.x - b.x || a.y - b.y)
    .slice(0, Math.min(MAX_MODE_CANDIDATES, Math.floor(options.modeCandidates)));
  for (const mode of strongestModes) {
    endpoints.push({
      x: clamp(mode.x, bounds.minX, bounds.maxX),
      y: clamp(mode.y, bounds.minY, bounds.maxY),
    });
  }

  const diagonal = Math.max(1e-9, Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY));
  const actions: SpatialBeliefAction[] = uniqueEndpoints(endpoints)
    .slice(0, MAX_TOTAL_ACTIONS)
    .map((endpoint, actionIndex) => {
    const dx = endpoint.x - current.x;
    const dy = endpoint.y - current.y;
    const distance = Math.hypot(dx, dy);
    const headingDeg = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
    const sampleCount = Math.max(1, Math.min(
      MAX_SAMPLES_PER_ACTION,
      Math.floor(options.maxSamplesPerAction ?? MAX_SAMPLES_PER_ACTION),
      Math.ceil(distance / Math.max(0.25, options.sampleSpacing)),
    ));
    const samples = Array.from({ length: sampleCount }, (_, index) => {
      const ratio = (index + 1) / sampleCount;
      return { x: current.x + dx * ratio, y: current.y + dy * ratio };
    });
    return {
      id: `${actionIndex}:${endpoint.x.toFixed(6)}:${endpoint.y.toFixed(6)}`,
      endpoint,
      travelCost: distance / diagonal,
      detectionLikelihoods: cells.map(cell => combineIndependentDetectionLikelihoods(
        samples.map(sample => detectionAt(sample, cell, headingDeg)),
      )),
    };
  });

  const choice = chooseBeliefAction(cells.map(cell => cell.probability), actions, weights);
  const selected = choice.action as SpatialBeliefAction;
  return { ...choice, waypoint: { ...selected.endpoint } };
}
