import type { ScenarioConfig, HelicopterState } from '../types/simulation.ts';
import type { BayesianGrid } from './bayesianGrid.ts';
import {
  aggregateBeliefGrid,
  chooseBeliefWaypoint,
  type BeliefPoint,
  type SpatialPoint,
} from './beliefSpacePlanner.ts';
import { advanceTowardWaypoint, estimateWaypointTravelMinutes } from './missionContract.ts';
import { calculatePdet } from './radarModel.ts';

/**
 * Approximate POMDP planner in belief space.
 *
 * A bounded set of spatial actions is evaluated with a binary observation
 * model (detection / non-detection). The utility combines immediate detection
 * probability, expected entropy reduction, transit cost and the existing
 * minute-by-minute safe-return shield. A four-minute receding horizon keeps
 * flight stable while allowing observations to change the next action.
 */
export class SIGMAPlanner {
  private readonly config: ScenarioConfig;
  private activeWaypoint: SpatialPoint | null = null;
  private replanAtMinutes = Number.NEGATIVE_INFINITY;
  private returning = false;

  constructor(config: ScenarioConfig) {
    this.config = config;
  }

  private moveToward(current: HelicopterState, target: SpatialPoint, dtMinutes: number): HelicopterState {
    return {
      ...advanceTowardWaypoint(
        current,
        target,
        this.config.helicoMaxSpeed,
        dtMinutes,
        this.config.windSpeed,
        this.config.windDirection,
      ),
      status: 'SEARCHING',
    };
  }

  private safeReturn(current: HelicopterState, dtMinutes: number): HelicopterState | null {
    const { frigateX, frigateY, helicoMaxSpeed, bingoFuelBuffer } = this.config;
    const returnTimeMinutes = estimateWaypointTravelMinutes(
      current,
      { x: frigateX, y: frigateY },
      helicoMaxSpeed,
      this.config.windSpeed,
      this.config.windDirection,
    );
    if (current.fuelRemaining <= 0) {
      return { ...current, fuelRemaining: 0, status: 'OUT_OF_FUEL' };
    }
    if (!this.returning && current.fuelRemaining > returnTimeMinutes + bingoFuelBuffer) return null;
    this.returning = true;
    const next = this.moveToward(current, { x: frigateX, y: frigateY }, dtMinutes);
    if (Math.hypot(next.x - frigateX, next.y - frigateY) <= 1e-9) {
      return { ...next, x: frigateX, y: frigateY, speed: 0, status: 'SAFE_RTB' };
    }
    return { ...next, status: 'BINGO_RETURN' };
  }

  private coarseBelief(grid: BayesianGrid): BeliefPoint[] {
    return aggregateBeliefGrid(grid.cells, 20, cell => ({
      x: cell.x,
      y: cell.y,
      probability: cell.pBayesianEvolved,
    }));
  }

  private chooseWaypoint(current: HelicopterState, grid: BayesianGrid): SpatialPoint {
    const halfWidth = this.config.searchAreaWidth / 2;
    const halfHeight = this.config.searchAreaHeight / 2;
    const belief = this.coarseBelief(grid);
    const radarParams = {
      baseRange: this.config.radarBaseRange,
      windSpeed: this.config.windSpeed,
      windDirection: this.config.windDirection,
    };
    const choice = chooseBeliefWaypoint(
      belief,
      current,
      (sample, cell, headingDeg) => {
        const dx = cell.x - sample.x;
        const dy = cell.y - sample.y;
        const distance = Math.hypot(dx, dy);
        const bearingDeg = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
        const baseDetection = calculatePdet(distance, bearingDeg, this.config.meanHeading, radarParams);
        const approach = (headingDeg - this.config.meanHeading) * Math.PI / 180;
        const perpendicularMultiplier = 0.70 + 0.50 * Math.abs(Math.sin(approach));
        return Math.min(0.98, baseDetection * perpendicularMultiplier);
      },
      {
        bounds: {
          minX: this.config.searchAreaCenterX - halfWidth,
          maxX: this.config.searchAreaCenterX + halfWidth,
          minY: this.config.searchAreaCenterY - halfHeight,
          maxY: this.config.searchAreaCenterY + halfHeight,
        },
        actionRadius: Math.max(
          this.config.radarBaseRange * 0.75,
          this.config.helicoMaxSpeed * 4 / 60,
        ),
        directionalCandidates: 12,
        modeCandidates: 6,
        sampleSpacing: Math.max(2, this.config.radarBaseRange * 0.6),
        maxSamplesPerAction: 12,
      },
    );
    return choice.waypoint;
  }

  private currentTarget(current: HelicopterState, grid: BayesianGrid, tMinutes: number): SpatialPoint {
    const arrived = this.activeWaypoint
      ? Math.hypot(this.activeWaypoint.x - current.x, this.activeWaypoint.y - current.y) <= 0.75
      : false;
    if (!this.activeWaypoint || arrived || tMinutes >= this.replanAtMinutes) {
      this.activeWaypoint = this.chooseWaypoint(current, grid);
      this.replanAtMinutes = tMinutes + 4;
    }
    return this.activeWaypoint;
  }

  public planStep(
    current: HelicopterState,
    grid: BayesianGrid,
    tMinutes: number,
    dtMinutes: number,
  ): HelicopterState {
    const returnState = this.safeReturn(current, dtMinutes);
    if (returnState) return returnState;

    const target = this.currentTarget(current, grid, tMinutes);
    return this.moveToward(current, target, dtMinutes);
  }
}
