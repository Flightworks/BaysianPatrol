import type { ScenarioConfig, HelicopterState } from '../types/simulation';
import { buildParallelSweepPlan, type SweepPoint } from './iamsarPattern';
import { advanceTowardWaypoint, estimateWaypointTravelMinutes } from './missionContract';

/**
 * Baseline IAMSAR: parallel sweep (PS) at constant track spacing.
 * It deliberately uses no Bayesian update and no learned decision.
 */
export class NaivePlanner {
  private readonly config: ScenarioConfig;
  private readonly waypoints: SweepPoint[];
  private waypointIndex = 0;
  private traversalDirection: 1 | -1 = 1;

  constructor(config: ScenarioConfig) {
    this.config = config;
    this.waypoints = buildParallelSweepPlan(config).waypoints;
  }

  private advanceWaypoint(): void {
    if (this.waypoints.length < 2) return;
    if (this.waypointIndex === this.waypoints.length - 1 && this.traversalDirection === 1) {
      this.traversalDirection = -1;
    } else if (this.waypointIndex === 0 && this.traversalDirection === -1) {
      this.traversalDirection = 1;
    }
    this.waypointIndex += this.traversalDirection;
  }

  public planStep(
    currentHelico: HelicopterState,
    _tMinutes: number,
    dtMinutes: number,
  ): HelicopterState {
    const { frigateX, frigateY, helicoMaxSpeed, bingoFuelBuffer, windSpeed, windDirection } = this.config;
    const returnTimeMinutes = estimateWaypointTravelMinutes(
      currentHelico,
      { x: frigateX, y: frigateY },
      helicoMaxSpeed,
      windSpeed,
      windDirection,
    );

    if (currentHelico.fuelRemaining <= 0) {
      return { ...currentHelico, fuelRemaining: 0, status: 'OUT_OF_FUEL' };
    }

    if (currentHelico.fuelRemaining <= returnTimeMinutes + bingoFuelBuffer) {
      const next = advanceTowardWaypoint(
        currentHelico,
        { x: frigateX, y: frigateY },
        helicoMaxSpeed,
        dtMinutes,
        windSpeed,
        windDirection,
      );
      const arrived = Math.hypot(next.x - frigateX, next.y - frigateY) <= 1e-9;
      return {
        ...next,
        x: arrived ? frigateX : next.x,
        y: arrived ? frigateY : next.y,
        speed: arrived ? 0 : next.speed,
        status: arrived ? 'SAFE_RTB' : 'BINGO_RETURN',
      };
    }

    const waypoint = this.waypoints[this.waypointIndex] ?? {
      x: this.config.searchAreaCenterX,
      y: this.config.searchAreaCenterY,
    };
    const next = advanceTowardWaypoint(
      currentHelico,
      waypoint,
      helicoMaxSpeed,
      dtMinutes,
      windSpeed,
      windDirection,
    );
    if (Math.hypot(next.x - waypoint.x, next.y - waypoint.y) <= 1e-9) {
      this.advanceWaypoint();
      return { ...next, x: waypoint.x, y: waypoint.y, status: 'SEARCHING' };
    }
    return { ...next, status: 'SEARCHING' };
  }
}
