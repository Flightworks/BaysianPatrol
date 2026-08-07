import type { ScenarioConfig, HelicopterState } from '../types/simulation';
import type { BayesianGrid } from './bayesianGrid';
import { buildBayesianCoverageLeg, type CoverageLeg, type Point } from './coverageBranch';
import { advanceTowardWaypoint, estimateWaypointTravelMinutes } from './missionContract';

/**
 * Bayesian coverage planner.
 * It selects a high-posterior, low-coverage corridor and commits to a complete
 * radar-sized leg before replanning. Recently completed corridors remain
 * excluded for the same 20-minute horizon used by radar scan memory.
 */
export class SIGMAPlanner {
  private readonly config: ScenarioConfig;
  private activeLeg: CoverageLeg | null = null;
  private activeEndpoint: 0 | 1 = 0;
  private completedLegs: Array<{ center: Point; completedAt: number }> = [];
  private returning = false;

  constructor(config: ScenarioConfig) {
    this.config = config;
  }

  private moveToward(current: HelicopterState, target: Point, dtMinutes: number): HelicopterState {
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

  private currentTarget(current: HelicopterState, grid: BayesianGrid, tMinutes: number): Point {
    if (!this.activeLeg) {
      const coverageExclusionMinutes = 20;
      this.completedLegs = this.completedLegs.filter(
        leg => tMinutes - leg.completedAt < coverageExclusionMinutes,
      );
      this.activeLeg = buildBayesianCoverageLeg(
        this.config,
        grid,
        current,
        this.completedLegs.map(leg => leg.center),
      );
      this.activeEndpoint = 0;
    }
    return this.activeEndpoint === 0 ? this.activeLeg.start : this.activeLeg.end;
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
    const next = this.moveToward(current, target, dtMinutes);
    if (Math.hypot(next.x - target.x, next.y - target.y) <= 1e-9 && this.activeLeg) {
      if (this.activeEndpoint === 0) {
        this.activeEndpoint = 1;
      } else {
        this.completedLegs.push({ center: this.activeLeg.center, completedAt: tMinutes });
        this.activeLeg = null;
        this.activeEndpoint = 0;
      }
    }
    return next;
  }
}
