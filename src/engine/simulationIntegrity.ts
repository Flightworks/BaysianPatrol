const DETECTION_STREAM_SALT = 0x0051f15e;
const STEP_MIX = 0x9e3779b1;
const PRNG_INCREMENT = 0x6d2b79f5;

/**
 * Returns one deterministic radar draw for a realization and time step.
 * The result does not depend on how many earlier detection opportunities
 * a strategy encountered, so paired strategies share the same draw at t.
 */
export function radarDetectionUniform(realizationSeed: number, step: number): number {
  if (!Number.isFinite(realizationSeed) || !Number.isInteger(realizationSeed)) {
    throw new RangeError('realizationSeed must be a finite integer');
  }
  if (!Number.isFinite(step) || !Number.isInteger(step) || step < 1) {
    throw new RangeError('step must be a positive integer');
  }

  let state = (
    (realizationSeed >>> 0)
    ^ Math.imul(step >>> 0, STEP_MIX)
    ^ DETECTION_STREAM_SALT
  ) >>> 0;
  if (state === 0) state = PRNG_INCREMENT;
  state = (state + PRNG_INCREMENT) >>> 0;
  let value = state;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

/** Mission states that must end the time step before radar detection. */
export function shouldTerminateBeforeDetection(status: string, outOfBounds: boolean): boolean {
  return status === 'SAFE_RTB' || status === 'OUT_OF_FUEL' || outOfBounds;
}
