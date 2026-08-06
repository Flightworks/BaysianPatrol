import type { ScenarioConfig, HelicopterState } from '../types/simulation';
import { degToRad, radToDeg, normalizeAngle } from './random';

/**
 * Naïve Search Strategy: Classic IAMSAR Creeping Line Search (Râteau Classique à Pas Constant).
 * Executes parallel transverse search legs perpendicular to target course with constant track spacing
 * and leg width derived directly from radar sensor performance and Datum spatial uncertainty.
 */
export class NaivePlanner {
  private config: ScenarioConfig;
  private legDirection: number = 1; // +1 = Right (starboard), -1 = Left (port)
  private legIndex: number = 0;

  constructor(config: ScenarioConfig) {
    this.config = config;
  }

  /**
   * Plan next step position for Naïve helicopter trajectory.
   */
  public planStep(
    currentHelico: HelicopterState,
    tMinutes: number,
    dtMinutes: number
  ): HelicopterState {
    const {
      frigateX, frigateY, helicoMaxSpeed, bingoFuelBuffer,
      datumX, datumY, meanHeading, meanSpeed,
      radarBaseRange, windSpeed, sigmaDatumX
    } = this.config;

    const dtHours = dtMinutes / 60.0;
    const distStep = helicoMaxSpeed * dtHours;

    // 1. Bingo Fuel Check
    const distToFrigate = Math.hypot(frigateX - currentHelico.x, frigateY - currentHelico.y);
    const returnTimeMinutes = (distToFrigate / helicoMaxSpeed) * 60.0;
    const minFuelNeeded = returnTimeMinutes + bingoFuelBuffer;

    if (currentHelico.fuelRemaining <= minFuelNeeded) {
      if (distToFrigate <= distStep) {
        return {
          x: frigateX,
          y: frigateY,
          heading: currentHelico.heading,
          speed: 0,
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
        speed: helicoMaxSpeed,
        fuelRemaining: Math.max(0, currentHelico.fuelRemaining - dtMinutes),
        status: 'BINGO_RETURN',
      };
    }

    if (currentHelico.fuelRemaining <= 0) {
      return {
        ...currentHelico,
        fuelRemaining: 0,
        status: 'OUT_OF_FUEL',
      };
    }

    // 2. Sensor Performance Dependent Parameters
    // Effective radar range taking sea clutter into account
    const seaClutterPenalty = Math.min(0.35, (windSpeed / 50.0) * 0.30);
    const R_eff = radarBaseRange * (1.0 - seaClutterPenalty);

    // Track spacing S derived from sensor coverage (1.5 x R_eff ensuring 25% overlap)
    const trackSpacing = 1.5 * R_eff;

    // Leg half-width L derived from Datum spatial uncertainty + sensor range
    const legHalfWidth = Math.max(16.0, 2.0 * sigmaDatumX + R_eff);

    // Estimated Mean Intercept Point ahead of target
    const tHours = tMinutes / 60.0;
    const targetRad = degToRad(meanHeading);
    const distFromDatum = Math.hypot(currentHelico.x - datumX, currentHelico.y - datumY);
    const estimatedTransitHours = tHours + distFromDatum / (helicoMaxSpeed + meanSpeed);

    const interceptMeanX = datumX + meanSpeed * Math.sin(targetRad) * estimatedTransitHours;
    const interceptMeanY = datumY + meanSpeed * Math.cos(targetRad) * estimatedTransitHours;

    // 3. Creeping Line Search Leg Calculation
    // Perpendicular vector relative to target track
    const perpHeading = normalizeAngle(meanHeading + 90 * this.legDirection);
    const perpRad = degToRad(perpHeading);

    // Downstream leg center position advancing along target track by legIndex * trackSpacing
    const trackRad = degToRad(normalizeAngle(meanHeading + 180));
    const legDistanceAlongTrack = (this.legIndex * trackSpacing) - 8.0;

    const legCenterX = interceptMeanX + legDistanceAlongTrack * Math.sin(trackRad);
    const legCenterY = interceptMeanY + legDistanceAlongTrack * Math.cos(trackRad);

    const waypointX = legCenterX + legHalfWidth * Math.sin(perpRad);
    const waypointY = legCenterY + legHalfWidth * Math.cos(perpRad);

    const distToWaypoint = Math.hypot(waypointX - currentHelico.x, waypointY - currentHelico.y);

    let nextX = currentHelico.x;
    let nextY = currentHelico.y;
    let desiredHeading = currentHelico.heading;

    if (distToWaypoint <= distStep + 0.3) {
      nextX = waypointX;
      nextY = waypointY;
      // Switch leg direction (starboard <-> port) and advance leg index
      this.legDirection *= -1;
      this.legIndex++;
    } else {
      desiredHeading = normalizeAngle(
        radToDeg(Math.atan2(waypointX - currentHelico.x, waypointY - currentHelico.y))
      );
      const rad = degToRad(desiredHeading);
      nextX += distStep * Math.sin(rad);
      nextY += distStep * Math.cos(rad);
    }

    return {
      x: nextX,
      y: nextY,
      heading: desiredHeading,
      speed: helicoMaxSpeed,
      fuelRemaining: currentHelico.fuelRemaining - dtMinutes,
      status: 'SEARCHING',
    };
  }
}
