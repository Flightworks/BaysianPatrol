import type { ScenarioConfig, HelicopterState } from '../types/simulation';
import type { BayesianGrid } from './bayesianGrid';
import { degToRad, radToDeg, normalizeAngle } from './random';

/**
 * AMI (Advanced Bayesian Interception) Trajectory Planner.
 * Guided dynamically by:
 * 1. Bayesian Posterior probability density P_posterior(M_i, t)
 * 2. Expanding cross-track uncertainty corridor L_half(t) = 2.2 * sigma_cross(t) + R_eff (>95% coverage)
 * 3. Perpendicular approach angle advantage (maximizing SER and visual wake detection)
 */
export class AMIPlanner {
  private config: ScenarioConfig;
  private phase: 'COUP_DE_FAUX' | 'BAYESIAN_PERPENDICULAR_CREEPING' | 'BINGO_RETURN' = 'COUP_DE_FAUX';
  private creepingDirection: number = 1;
  private creepingLegIndex: number = 0;

  constructor(config: ScenarioConfig) {
    this.config = config;
  }

  /**
   * Determine next helicopter position and state after dt minutes.
   */
  public planStep(
    currentHelico: HelicopterState,
    grid: BayesianGrid,
    tMinutes: number,
    dtMinutes: number
  ): HelicopterState {
    const { frigateX, frigateY, helicoMaxSpeed, bingoFuelBuffer } = this.config;
    const dtHours = dtMinutes / 60.0;
    const distStep = helicoMaxSpeed * dtHours;

    // 1. Check Bingo Fuel status
    const distToFrigate = Math.hypot(frigateX - currentHelico.x, frigateY - currentHelico.y);
    const returnTimeMinutes = (distToFrigate / helicoMaxSpeed) * 60.0;
    const minFuelNeeded = returnTimeMinutes + bingoFuelBuffer;

    if (currentHelico.fuelRemaining <= minFuelNeeded) {
      this.phase = 'BINGO_RETURN';
    }

    if (currentHelico.fuelRemaining <= 0) {
      return {
        ...currentHelico,
        fuelRemaining: 0,
        status: 'OUT_OF_FUEL',
      };
    }

    if (this.phase === 'BINGO_RETURN') {
      if (distToFrigate <= distStep) {
        return {
          x: frigateX,
          y: frigateY,
          heading: currentHelico.heading,
          fuelRemaining: Math.max(0, currentHelico.fuelRemaining - dtMinutes),
          status: 'BINGO_RETURN',
        };
      }

      const headingToFrigate = normalizeAngle(
        radToDeg(Math.atan2(frigateX - currentHelico.x, frigateY - currentHelico.y))
      );
      const rad = degToRad(headingToFrigate);

      return {
        x: currentHelico.x + distStep * Math.sin(rad),
        y: currentHelico.y + distStep * Math.cos(rad),
        heading: headingToFrigate,
        fuelRemaining: Math.max(0, currentHelico.fuelRemaining - dtMinutes),
        status: 'BINGO_RETURN',
      };
    }

    // 2. Tactical Search Trajectory Planning
    const targetWaypoint = this.computeWaypoint(currentHelico, grid, tMinutes);
    const distToWaypoint = Math.hypot(targetWaypoint.x - currentHelico.x, targetWaypoint.y - currentHelico.y);

    let nextX = currentHelico.x;
    let nextY = currentHelico.y;
    let desiredHeading = currentHelico.heading;

    if (distToWaypoint <= distStep + 0.3) {
      nextX = targetWaypoint.x;
      nextY = targetWaypoint.y;
      if (this.phase === 'COUP_DE_FAUX') {
        this.phase = 'BAYESIAN_PERPENDICULAR_CREEPING';
        this.creepingLegIndex++;
      } else if (this.phase === 'BAYESIAN_PERPENDICULAR_CREEPING') {
        this.creepingDirection *= -1;
        this.creepingLegIndex++;
      }
    } else {
      desiredHeading = normalizeAngle(
        radToDeg(Math.atan2(targetWaypoint.x - currentHelico.x, targetWaypoint.y - currentHelico.y))
      );
      const rad = degToRad(desiredHeading);
      nextX += distStep * Math.sin(rad);
      nextY += distStep * Math.cos(rad);
    }

    return {
      x: nextX,
      y: nextY,
      heading: desiredHeading,
      fuelRemaining: currentHelico.fuelRemaining - dtMinutes,
      status: 'SEARCHING',
    };
  }

  /**
   * Calculate tactical waypoint prioritizing:
   * 1. High remaining Bayesian posterior probability density P_posterior(M_i).
   * 2. Dynamically expanding leg sweep width L_half(t) = 2.2 * sigma_cross(t) + R_eff.
   * 3. Perpendicular approach angle relative to target heading (maximizing SER & visual wake detection).
   */
  private computeWaypoint(
    helico: HelicopterState,
    grid: BayesianGrid,
    tMinutes: number
  ): { x: number; y: number } {
    const {
      datumX, datumY, sigmaDatumY, meanHeading, meanSpeed, sigmaHeading,
      sigmaRouteDrift, helicoMaxSpeed, radarBaseRange, windSpeed
    } = this.config;

    const tHours = tMinutes / 60.0;
    const targetRad = degToRad(meanHeading);

    // Effective radar range
    const seaClutterPenalty = Math.min(0.35, (windSpeed / 50.0) * 0.30);
    const R_eff = radarBaseRange * (1.0 - seaClutterPenalty);

    // Dynamically expanding cross-track standard deviation sigma_cross(t)
    const sigmaCross = Math.sqrt(
      Math.pow(sigmaDatumY, 2) +
      Math.pow(meanSpeed * Math.sin(degToRad(sigmaHeading)) * tHours, 2) +
      Math.pow(sigmaRouteDrift * 0.3 * Math.sqrt(Math.max(0.01, tHours)), 2) +
      9.0
    );

    // Dynamic Leg Half-Width L_half(t) extending to 2.2 * sigma_cross + R_eff (>95% coverage)
    const L_half = Math.max(18.0, 2.2 * sigmaCross + R_eff);
    
    // Estimate advance intercept point for "Coup de faux"
    const distFromDatum = Math.hypot(helico.x - datumX, helico.y - datumY);
    const estimatedInterceptTimeHours = tHours + distFromDatum / (helicoMaxSpeed + meanSpeed);
    
    const interceptMeanX = datumX + meanSpeed * Math.sin(targetRad) * estimatedInterceptTimeHours;
    const interceptMeanY = datumY + meanSpeed * Math.cos(targetRad) * estimatedInterceptTimeHours;

    if (this.phase === 'COUP_DE_FAUX') {
      // Perpendicular sweep across target track (Aspect angle = 90 deg -> maximum SER & wake intersection)
      const perpHeading = normalizeAngle(meanHeading + 90);
      const perpRad = degToRad(perpHeading);

      return {
        x: interceptMeanX + L_half * Math.sin(perpRad),
        y: interceptMeanY + L_half * Math.cos(perpRad),
      };
    } else {
      // BAYESIAN PERPENDICULAR CREEPING LINE:
      // Evaluate tactical value per unscanned cell combining posterior probability AND perpendicular aspect angle
      let maxScore = -1.0;
      let maxCell = { x: interceptMeanX, y: interceptMeanY };

      for (let j = 0; j < grid.heightCells; j++) {
        for (let i = 0; i < grid.widthCells; i++) {
          const cell = grid.cells[j][i];
          if (cell.scanned || cell.pPresence < 0.0001) continue;

          // Bearing from helicopter to cell
          const bearingRad = Math.atan2(cell.x - helico.x, cell.y - helico.y);
          const bearingDeg = normalizeAngle((bearingRad * 180.0) / Math.PI);

          // Aspect angle alpha relative to target course
          const aspectRad = degToRad(normalizeAngle(bearingDeg - meanHeading));
          const perpFactor = Math.abs(Math.sin(aspectRad)); // 1.0 when approach is perpendicular!

          // Tactical Score: P_posterior * PerpendicularApproachBonus
          const score = cell.pPresence * (0.6 + 0.4 * perpFactor);

          if (score > maxScore) {
            maxScore = score;
            maxCell = { x: cell.x, y: cell.y };
          }
        }
      }

      if (maxScore > 0) {
        return maxCell;
      }

      // Fallback perpendicular leg upstream along target course extending to L_half
      const perpHeading = normalizeAngle(meanHeading + 90 * this.creepingDirection);
      const perpRad = degToRad(perpHeading);

      const trackSpacing = 1.4 * R_eff;
      const upstreamDist = Math.max(-12, (this.creepingLegIndex * trackSpacing) - 6.0);
      const trackRad = degToRad(normalizeAngle(meanHeading + 180));

      const legCenterX = interceptMeanX + upstreamDist * Math.sin(trackRad);
      const legCenterY = interceptMeanY + upstreamDist * Math.cos(trackRad);

      return {
        x: legCenterX + L_half * Math.sin(perpRad),
        y: legCenterY + L_half * Math.cos(perpRad),
      };
    }
  }
}
