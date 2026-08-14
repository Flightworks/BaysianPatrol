import type { ScenarioConfig } from '../types/simulation';

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

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

export function effectiveRadarRange(config: ScenarioConfig): number {
  const seaClutterPenalty = Math.min(0.35, (config.windSpeed / 50) * 0.3);
  return Math.max(0.5, config.radarBaseRange * (1 - seaClutterPenalty));
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
