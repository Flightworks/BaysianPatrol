import { degToRad, normalizeAngle } from './random';

/**
 * Radar Sensor & SER (Radar Cross Section) Model with Environmental Factors (Wind/Sea State).
 */

export interface RadarParams {
  baseRange: number;       // R_0 in NM
  windSpeed: number;       // knots
  windDirection: number;   // degrees (from which wind blows)
}

/**
 * Calculate Radar Cross Section (SER) multiplier based on aspect angle alpha (degrees).
 * Alpha is aspect angle relative to target heading:
 * 90° / 270° (broadside/wake): Maximum SER (1.0)
 * 0° / 180° (head-on / stern): Minimum SER (0.25)
 */
export function calculateSER(aspectAngleDeg: number): number {
  const rad = degToRad(aspectAngleDeg);
  const SER_min = 0.25;
  const SER_max = 1.0;
  return SER_min + (SER_max - SER_min) * Math.pow(Math.sin(rad), 2);
}

/**
 * Calculate effective radar range R_eff along a specific bearing thetaBearing (degrees)
 * taking into account wind-induced sea clutter and target aspect angle.
 */
export function calculateDirectionalRadarRange(
  bearingDeg: number,
  targetEstimatedHeadingDeg: number,
  params: RadarParams
): number {
  const { baseRange, windSpeed, windDirection } = params;

  // 1. Wind Sea Clutter Penalty (Upwind sector experiences heavy wave return)
  let clutterPenalty = 0.0;
  if (windSpeed > 5) {
    const relativeWindAngle = degToRad(normalizeAngle(bearingDeg - windDirection));
    const upwindFactor = Math.max(0, Math.cos(relativeWindAngle)); // 1.0 directly upwind
    clutterPenalty = Math.min(0.45, (windSpeed / 45.0) * 0.45 * upwindFactor);
  }

  // 2. Aspect Angle SER Factor (Target broadside produces stronger radar returns)
  const aspectAngle = normalizeAngle(bearingDeg - targetEstimatedHeadingDeg);
  const serFactor = calculateSER(aspectAngle);

  // Effective range modulated by wind clutter and target aspect
  const rEff = baseRange * (1.0 - clutterPenalty) * (0.75 + 0.35 * serFactor);
  return rEff;
}

/**
 * Generate 2D contour points of the anisotropic radar range footprint.
 * Returns array of [x, y] relative offset coordinates in NM.
 */
export function getRadarFootprintContour(
  targetHeadingDeg: number,
  params: RadarParams,
  steps: number = 36
): Array<{ angle: number; r: number; dx: number; dy: number }> {
  const points: Array<{ angle: number; r: number; dx: number; dy: number }> = [];

  for (let i = 0; i < steps; i++) {
    const angleDeg = (i * 360.0) / steps;
    const r = calculateDirectionalRadarRange(angleDeg, targetHeadingDeg, params);
    const rad = degToRad(angleDeg);

    points.push({
      angle: angleDeg,
      r,
      dx: r * Math.sin(rad),
      dy: r * Math.cos(rad),
    });
  }

  return points;
}

/**
 * Calculate probability of detection P_det for a grid cell given:
 * - dist: distance from helicopter to cell center (NM)
 * - bearingDeg: bearing from helicopter to cell center (degrees)
 * - estimatedTargetHeadingDeg: target estimated course (degrees)
 * - params: RadarParams
 */
export function calculatePdet(
  dist: number,
  bearingDeg: number,
  estimatedTargetHeadingDeg: number,
  params: RadarParams
): number {
  const rEff = calculateDirectionalRadarRange(bearingDeg, estimatedTargetHeadingDeg, params);

  // Outside directional effective range
  if (dist > rEff) return 0.0;

  const aspectAngle = normalizeAngle(bearingDeg - estimatedTargetHeadingDeg);
  const serFactor = calculateSER(aspectAngle);

  const distRatio = dist / rEff;
  const pMax = 0.95;

  const pDet = pMax * serFactor * (1.0 - Math.pow(distRatio, 2));
  return Math.max(0.0, Math.min(0.98, pDet));
}
