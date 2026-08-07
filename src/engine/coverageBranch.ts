import type { ScenarioConfig } from '../types/simulation';
import type { BayesianGrid } from './bayesianGrid';

export interface Point {
  x: number;
  y: number;
}

export interface SearchBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface CoverageLeg {
  start: Point;
  end: Point;
  center: Point;
  trackSpacing: number;
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

export function effectiveRadarRange(config: ScenarioConfig): number {
  const seaClutterPenalty = Math.min(0.35, (config.windSpeed / 50) * 0.3);
  return Math.max(0.5, config.radarBaseRange * (1 - seaClutterPenalty));
}

function boundsFor(config: ScenarioConfig): SearchBounds {
  return {
    minX: config.searchAreaCenterX - config.searchAreaWidth / 2,
    maxX: config.searchAreaCenterX + config.searchAreaWidth / 2,
    minY: config.searchAreaCenterY - config.searchAreaHeight / 2,
    maxY: config.searchAreaCenterY + config.searchAreaHeight / 2,
  };
}

function lineInterval(center: Point, direction: Point, bounds: SearchBounds): [number, number] | null {
  let low = Number.NEGATIVE_INFINITY;
  let high = Number.POSITIVE_INFINITY;
  for (const [position, delta, minimum, maximum] of [
    [center.x, direction.x, bounds.minX, bounds.maxX],
    [center.y, direction.y, bounds.minY, bounds.maxY],
  ] as const) {
    if (Math.abs(delta) < 1e-12) {
      if (position < minimum || position > maximum) return null;
      continue;
    }
    const first = (minimum - position) / delta;
    const second = (maximum - position) / delta;
    low = Math.max(low, Math.min(first, second));
    high = Math.min(high, Math.max(first, second));
  }
  return low <= high ? [low, high] : null;
}

export function buildBayesianCoverageLeg(
  config: ScenarioConfig,
  grid: BayesianGrid,
  current: Point,
  completedCenters: readonly Point[] = [],
): CoverageLeg {
  const routeRad = config.meanHeading * Math.PI / 180;
  const route = { x: Math.sin(routeRad), y: Math.cos(routeRad) };
  const legDirection = { x: Math.cos(routeRad), y: -Math.sin(routeRad) };
  const radarRange = effectiveRadarRange(config);
  const trackSpacing = 1.4 * radarRange;

  let selected: Point | null = null;
  let selectedScore = Number.NEGATIVE_INFINITY;
  for (const row of grid.cells) {
    for (const cell of row) {
      const conflictsWithCompletedCorridor = completedCenters.some(center => {
        const alongDistance = Math.abs((cell.x - center.x) * route.x + (cell.y - center.y) * route.y);
        return alongDistance < trackSpacing * 0.95;
      });
      if (conflictsWithCompletedCorridor) continue;
      const uncovered = Math.max(0, 1 - cell.scanMemory);
      const travelPenalty = 1 + Math.hypot(cell.x - current.x, cell.y - current.y) / 200;
      const score = cell.pBayesianEvolved * (0.02 + uncovered * uncovered) / travelPenalty;
      if (score > selectedScore) {
        selectedScore = score;
        selected = { x: cell.x, y: cell.y };
      }
    }
  }

  if (!selected) {
    selected = {
      x: clamp(config.datumX, boundsFor(config).minX, boundsFor(config).maxX),
      y: clamp(config.datumY, boundsFor(config).minY, boundsFor(config).maxY),
    };
  }

  const bounds = boundsFor(config);
  const interval = lineInterval(selected, legDirection, bounds) ?? [-radarRange, radarRange];
  const symmetricHalfLength = Math.max(0, Math.min(Math.abs(interval[0]), Math.abs(interval[1])));
  let start: Point;
  let end: Point;
  if (symmetricHalfLength >= radarRange) {
    start = {
      x: selected.x - legDirection.x * symmetricHalfLength,
      y: selected.y - legDirection.y * symmetricHalfLength,
    };
    end = {
      x: selected.x + legDirection.x * symmetricHalfLength,
      y: selected.y + legDirection.y * symmetricHalfLength,
    };
  } else {
    start = {
      x: selected.x + legDirection.x * interval[0],
      y: selected.y + legDirection.y * interval[0],
    };
    end = {
      x: selected.x + legDirection.x * interval[1],
      y: selected.y + legDirection.y * interval[1],
    };
  }

  if (Math.hypot(end.x - current.x, end.y - current.y) < Math.hypot(start.x - current.x, start.y - current.y)) {
    [start, end] = [end, start];
  }
  return { start, end, center: selected, trackSpacing };
}

export class BranchCommitment {
  private activeTarget: Point | null = null;
  private readonly bounds: SearchBounds;
  private readonly minimumLegLength: number;
  private readonly captureRadius: number;

  public constructor(bounds: SearchBounds, minimumLegLength: number, captureRadius = 0.75) {
    this.bounds = bounds;
    this.minimumLegLength = minimumLegLength;
    this.captureRadius = captureRadius;
  }

  public resolve(current: Point, proposal: Point): Point {
    if (this.activeTarget && Math.hypot(this.activeTarget.x - current.x, this.activeTarget.y - current.y) > this.captureRadius) {
      return { ...this.activeTarget };
    }

    const dx = proposal.x - current.x;
    const dy = proposal.y - current.y;
    const proposalDistance = Math.hypot(dx, dy);
    const ux = proposalDistance > 1e-9 ? dx / proposalDistance : 0;
    const uy = proposalDistance > 1e-9 ? dy / proposalDistance : 1;
    const desiredDistance = Math.max(this.minimumLegLength, proposalDistance);
    let maximumDistance = Number.POSITIVE_INFINITY;
    if (ux > 1e-12) maximumDistance = Math.min(maximumDistance, (this.bounds.maxX - current.x) / ux);
    if (ux < -1e-12) maximumDistance = Math.min(maximumDistance, (this.bounds.minX - current.x) / ux);
    if (uy > 1e-12) maximumDistance = Math.min(maximumDistance, (this.bounds.maxY - current.y) / uy);
    if (uy < -1e-12) maximumDistance = Math.min(maximumDistance, (this.bounds.minY - current.y) / uy);
    const distance = Math.max(0, Math.min(desiredDistance, maximumDistance));
    this.activeTarget = {
      x: clamp(current.x + ux * distance, this.bounds.minX, this.bounds.maxX),
      y: clamp(current.y + uy * distance, this.bounds.minY, this.bounds.maxY),
    };
    return { ...this.activeTarget };
  }
}
