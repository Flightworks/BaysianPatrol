export interface BeliefTransitionOptions {
  width: number;
  height: number;
  cellSize: number;
  headingDeg: number;
  distance: number;
  sigmaAlong: number;
  sigmaCross: number;
}

interface SigmaPoint {
  along: number;
  cross: number;
  weight: number;
}

const MAX_GRID_CELLS = 100_000;

const clamp = (value: number, low: number, high: number): number =>
  Math.max(low, Math.min(high, value));

/**
 * Approximate Chapman-Kolmogorov prediction on a regular grid.
 * Five deterministic cubature points match the continuous transition moments
 * before bilinear grid deposition. Fractional deposition adds at most one
 * quarter cell² of variance per axis. Outside mass is conservatively projected
 * onto the nearest boundary cell so it remains visible to the search policy.
 */
export function propagateGridBelief(
  sourceInput: readonly number[],
  options: BeliefTransitionOptions,
): number[] {
  for (const [label, value] of Object.entries(options)) {
    if (!Number.isFinite(value)) throw new Error(`Transition ${label} must be finite`);
  }
  if (!Number.isInteger(options.width) || !Number.isInteger(options.height)) {
    throw new Error('Transition grid dimensions must be integers');
  }
  const width = Math.max(1, Math.floor(options.width));
  const height = Math.max(1, Math.floor(options.height));
  if (width > Math.floor(MAX_GRID_CELLS / height) || sourceInput.length > MAX_GRID_CELLS) {
    throw new Error(`Belief transition is limited to ${MAX_GRID_CELLS} grid cells`);
  }
  const gridCellCount = width * height;
  if (sourceInput.length !== gridCellCount) {
    throw new Error('Belief size must match transition grid dimensions');
  }
  if (!(options.cellSize > 0)) throw new Error('Transition cell size must be positive');
  if (options.sigmaAlong < 0 || options.sigmaCross < 0) {
    throw new Error('Transition uncertainty must be non-negative');
  }

  const source = sourceInput.map((value, index) => {
    if (!Number.isFinite(value)) throw new Error(`Belief value ${index} must be finite`);
    if (value < 0) throw new Error(`Belief value ${index} must be non-negative`);
    return value;
  });
  const sourceMass = source.reduce((sum, value) => sum + value, 0);
  if (!(sourceMass > 0)) throw new Error('Belief transition requires positive mass');
  const invSourceMass = 1 / sourceMass;
  const output = new Array(gridCellCount).fill(0);

  const headingRad = options.headingDeg * Math.PI / 180;
  const sinHeading = Math.sin(headingRad);
  const cosHeading = Math.cos(headingRad);
  const sigmaAlong = Math.max(0, options.sigmaAlong);
  const sigmaCross = Math.max(0, options.sigmaCross);
  const spread = Math.sqrt(3);
  const sigmaPoints: SigmaPoint[] = [
    { along: 0, cross: 0, weight: 1 / 3 },
    { along: spread * sigmaAlong, cross: 0, weight: 1 / 6 },
    { along: -spread * sigmaAlong, cross: 0, weight: 1 / 6 },
    { along: 0, cross: spread * sigmaCross, weight: 1 / 6 },
    { along: 0, cross: -spread * sigmaCross, weight: 1 / 6 },
  ];
  const meanDx = options.distance * sinHeading;
  const meanDy = options.distance * cosHeading;

  const deposit = (x: number, y: number, mass: number): void => {
    const boundedX = clamp(x, 0, width - 1);
    const boundedY = clamp(y, 0, height - 1);
    const x0 = Math.floor(boundedX);
    const y0 = Math.floor(boundedY);
    const x1 = Math.min(width - 1, x0 + 1);
    const y1 = Math.min(height - 1, y0 + 1);
    const fx = boundedX - x0;
    const fy = boundedY - y0;
    output[y0 * width + x0] += mass * (1 - fx) * (1 - fy);
    output[y0 * width + x1] += mass * fx * (1 - fy);
    output[y1 * width + x0] += mass * (1 - fx) * fy;
    output[y1 * width + x1] += mass * fx * fy;
  };

  for (let index = 0; index < source.length; index += 1) {
    const probability = source[index] * invSourceMass;
    if (probability <= 0) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    for (const point of sigmaPoints) {
      const offsetX = meanDx + point.along * sinHeading - point.cross * cosHeading;
      const offsetY = meanDy + point.along * cosHeading + point.cross * sinHeading;
      deposit(
        x + offsetX / options.cellSize,
        y + offsetY / options.cellSize,
        probability * point.weight,
      );
    }
  }

  const outputMass = output.reduce((sum, value) => sum + value, 0);
  if (!(outputMass > 0)) throw new Error('Belief transition produced no mass');
  return output.map(value => value / outputMass);
}
