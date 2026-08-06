export interface FuelMetrics {
  returnTimeMinutes: number;
  marginMinutes: number;
}

export interface HybridVectorInput {
  helicoX: number;
  helicoY: number;
  headingDeg: number;
  speed: number;
  maxSpeed: number;
  fuelRemaining: number;
  maxFuel: number;
  frigateX: number;
  frigateY: number;
  halfWidth: number;
  halfHeight: number;
  peakX: number;
  peakY: number;
  entropy: number;
  elapsedMinutes: number;
  bingoBuffer: number;
}

export interface MinimalHelicopterState {
  x: number;
  y: number;
  heading: number;
  fuelRemaining: number;
}

export interface WaypointStep extends MinimalHelicopterState {
  speed: number;
}

const clamp = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));

export function computeFuelMargin(
  fuelRemaining: number,
  distanceToFrigate: number,
  maxSpeed: number,
  bingoBuffer: number,
): FuelMetrics {
  const returnTimeMinutes = maxSpeed > 0 ? (distanceToFrigate / maxSpeed) * 60 : Number.POSITIVE_INFINITY;
  return { returnTimeMinutes, marginMinutes: fuelRemaining - returnTimeMinutes - bingoBuffer };
}

export function scaleBeliefForObservation(probabilities: readonly number[]): number[] {
  const peak = probabilities.reduce((maximum, value) => Math.max(maximum, value), 0);
  if (peak <= 0) return probabilities.map(() => 0);
  return probabilities.map(value => value / peak);
}

export function normalizedEntropy(probabilities: readonly number[]): number {
  const total = probabilities.reduce((sum, value) => sum + Math.max(0, value), 0);
  if (total <= 0 || probabilities.length <= 1) return 0;
  let entropy = 0;
  for (const value of probabilities) {
    const p = Math.max(0, value) / total;
    if (p > 0) entropy -= p * Math.log(p);
  }
  return clamp(entropy / Math.log(probabilities.length), 0, 1);
}

export function buildHybridVector(input: HybridVectorInput): number[] {
  const headingRad = input.headingDeg * Math.PI / 180;
  const distanceToFrigate = Math.hypot(input.frigateX-input.helicoX, input.frigateY-input.helicoY);
  const fuel = computeFuelMargin(input.fuelRemaining, distanceToFrigate, input.maxSpeed, input.bingoBuffer);
  return [
    Math.sin(headingRad),
    Math.cos(headingRad),
    clamp(input.speed / input.maxSpeed, 0, 1),
    clamp(fuel.marginMinutes / input.maxFuel, -1, 1),
    clamp((input.frigateX-input.helicoX) / input.halfWidth, -1, 1),
    clamp((input.frigateY-input.helicoY) / input.halfHeight, -1, 1),
    clamp((input.peakX-input.helicoX) / input.halfWidth, -1, 1),
    clamp((input.peakY-input.helicoY) / input.halfHeight, -1, 1),
    clamp(input.entropy, 0, 1),
    clamp(input.elapsedMinutes / input.maxFuel, 0, 1),
  ];
}

export function applyRelativeWaypoint(
  state: MinimalHelicopterState,
  action: readonly number[],
  halfWidth: number,
  halfHeight: number,
  maxSpeed: number,
  dtMinutes: number,
  centerX: number = 0,
  centerY: number = 0,
): WaypointStep {
  const waypointX = clamp(state.x + clamp(action[0] ?? 0, -1, 1) * halfWidth, centerX-halfWidth, centerX+halfWidth);
  const waypointY = clamp(state.y + clamp(action[1] ?? 0, -1, 1) * halfHeight, centerY-halfHeight, centerY+halfHeight);
  const dx = waypointX-state.x;
  const dy = waypointY-state.y;
  const remaining = Math.hypot(dx, dy);
  const heading = remaining > 1e-9 ? (Math.atan2(dx, dy)*180/Math.PI+360)%360 : state.heading;
  const travel = Math.min(remaining, maxSpeed/60*dtMinutes);
  const headingRad = heading*Math.PI/180;
  return {
    x: state.x + travel*Math.sin(headingRad),
    y: state.y + travel*Math.cos(headingRad),
    heading,
    speed: maxSpeed,
    fuelRemaining: Math.max(0, state.fuelRemaining-dtMinutes),
  };
}
