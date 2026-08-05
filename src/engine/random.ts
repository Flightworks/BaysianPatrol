/**
 * Utility functions for stochastic random sampling and geometric math.
 */

// Box-Muller transform for standard Normal distribution N(0,1)
let spareRandom: number | null = null;

export function standardGaussian(): number {
  if (spareRandom !== null) {
    const val = spareRandom;
    spareRandom = null;
    return val;
  }
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  
  const mag = Math.sqrt(-2.0 * Math.log(u));
  const z0 = mag * Math.cos(2.0 * Math.PI * v);
  const z1 = mag * Math.sin(2.0 * Math.PI * v);
  
  spareRandom = z1;
  return z0;
}

export function randomGaussian(mean: number, std: number): number {
  return mean + std * standardGaussian();
}

export function randomUniform(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export function degToRad(deg: number): number {
  return (deg * Math.PI) / 180.0;
}

export function radToDeg(rad: number): number {
  return (rad * 180.0) / Math.PI;
}

// Normalize angle to [0, 360)
export function normalizeAngle(deg: number): number {
  let a = deg % 360;
  if (a < 0) a += 360;
  return a;
}

// Shortest angle difference in degrees [-180, 180]
export function angleDiff(angle1: number, angle2: number): number {
  let diff = (angle2 - angle1 + 180) % 360 - 180;
  return diff < -180 ? diff + 360 : diff;
}
