import type { ScenarioConfig, HelicopterState } from '../types/simulation';
import { degToRad, radToDeg, normalizeAngle } from './random';
import { buildParallelSweepPlan, type SweepPoint } from './iamsarPattern';

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
    const { frigateX, frigateY, helicoMaxSpeed, bingoFuelBuffer } = this.config;
    const distStep = helicoMaxSpeed * (dtMinutes / 60);
    const distToFrigate = Math.hypot(frigateX - currentHelico.x, frigateY - currentHelico.y);
    const returnTimeMinutes = (distToFrigate / helicoMaxSpeed) * 60;

    if (currentHelico.fuelRemaining <= 0) {
      return { ...currentHelico, fuelRemaining: 0, status: 'OUT_OF_FUEL' };
    }

    if (currentHelico.fuelRemaining <= returnTimeMinutes + bingoFuelBuffer) {
      if (distToFrigate <= distStep) {
        return {
          x: frigateX,
          y: frigateY,
          heading: currentHelico.heading,
          speed: 0,
          fuelRemaining: Math.max(0, currentHelico.fuelRemaining - dtMinutes),
          status: 'SAFE_RTB',
        };
      }
      const heading = normalizeAngle(radToDeg(Math.atan2(frigateX - currentHelico.x, frigateY - currentHelico.y)));
      const headingRad = degToRad(heading);
      return {
        x: currentHelico.x + distStep * Math.sin(headingRad),
        y: currentHelico.y + distStep * Math.cos(headingRad),
        heading,
        speed: helicoMaxSpeed,
        fuelRemaining: Math.max(0, currentHelico.fuelRemaining - dtMinutes),
        status: 'BINGO_RETURN',
      };
    }

    const waypoint = this.waypoints[this.waypointIndex] ?? {
      x: this.config.searchAreaCenterX,
      y: this.config.searchAreaCenterY,
    };
    const distance = Math.hypot(waypoint.x - currentHelico.x, waypoint.y - currentHelico.y);

    if (distance <= distStep) {
      const heading = distance > 1e-9
        ? normalizeAngle(radToDeg(Math.atan2(waypoint.x - currentHelico.x, waypoint.y - currentHelico.y)))
        : currentHelico.heading;
      this.advanceWaypoint();
      return {
        x: waypoint.x,
        y: waypoint.y,
        heading,
        speed: helicoMaxSpeed,
        fuelRemaining: Math.max(0, currentHelico.fuelRemaining - dtMinutes),
        status: 'SEARCHING',
      };
    }

    const heading = normalizeAngle(radToDeg(Math.atan2(waypoint.x - currentHelico.x, waypoint.y - currentHelico.y)));
    const headingRad = degToRad(heading);
    return {
      x: currentHelico.x + distStep * Math.sin(headingRad),
      y: currentHelico.y + distStep * Math.cos(headingRad),
      heading,
      speed: helicoMaxSpeed,
      fuelRemaining: Math.max(0, currentHelico.fuelRemaining - dtMinutes),
      status: 'SEARCHING',
    };
  }
}
