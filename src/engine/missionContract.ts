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

export interface GroundTrackSolution {
  groundSpeed: number;
  airHeading: number;
  trackMaintained: boolean;
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

export function solveGroundTrack(
  airSpeed: number,
  trackHeading: number,
  windSpeed: number,
  windFromDirection: number,
): GroundTrackSolution {
  const trackRad = trackHeading * Math.PI / 180;
  const windToRad = (windFromDirection + 180) * Math.PI / 180;
  const trackX = Math.sin(trackRad);
  const trackY = Math.cos(trackRad);
  const rightX = Math.cos(trackRad);
  const rightY = -Math.sin(trackRad);
  const windX = windSpeed * Math.sin(windToRad);
  const windY = windSpeed * Math.cos(windToRad);
  const alongWind = windX * trackX + windY * trackY;
  const crossWind = windX * rightX + windY * rightY;
  if (Math.abs(crossWind) > airSpeed) {
    const airX = -rightX * Math.sign(crossWind) * airSpeed;
    const airY = -rightY * Math.sign(crossWind) * airSpeed;
    return {
      groundSpeed: 0,
      airHeading: (Math.atan2(airX, airY) * 180 / Math.PI + 360) % 360,
      trackMaintained: false,
    };
  }
  const airAlongTrack = Math.sqrt(Math.max(0, airSpeed * airSpeed - crossWind * crossWind));
  const airX = trackX * airAlongTrack - rightX * crossWind;
  const airY = trackY * airAlongTrack - rightY * crossWind;
  const airHeading = (Math.atan2(airX, airY) * 180 / Math.PI + 360) % 360;

  return {
    groundSpeed: Math.max(0, airAlongTrack + alongWind),
    airHeading,
    trackMaintained: true,
  };
}

export function advanceTowardWaypoint(
  state: MinimalHelicopterState,
  waypoint: { x: number; y: number },
  airSpeed: number,
  dtMinutes: number,
  windSpeed: number = 0,
  windFromDirection: number = 0,
): WaypointStep {
  const dx = waypoint.x - state.x;
  const dy = waypoint.y - state.y;
  const remaining = Math.hypot(dx, dy);
  if (remaining <= 1e-9) {
    return {
      ...state,
      speed: airSpeed,
      fuelRemaining: Math.max(0, state.fuelRemaining - dtMinutes),
    };
  }

  const trackHeading = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
  const solution = solveGroundTrack(airSpeed, trackHeading, windSpeed, windFromDirection);
  const travel = Math.min(remaining, solution.groundSpeed / 60 * dtMinutes);
  const trackRad = trackHeading * Math.PI / 180;
  return {
    x: state.x + travel * Math.sin(trackRad),
    y: state.y + travel * Math.cos(trackRad),
    heading: solution.airHeading,
    speed: airSpeed,
    fuelRemaining: Math.max(0, state.fuelRemaining - dtMinutes),
  };
}

export function estimateWaypointTravelMinutes(
  state: Pick<MinimalHelicopterState, 'x' | 'y'>,
  waypoint: { x: number; y: number },
  airSpeed: number,
  windSpeed: number = 0,
  windFromDirection: number = 0,
): number {
  const dx = waypoint.x - state.x;
  const dy = waypoint.y - state.y;
  const distance = Math.hypot(dx, dy);
  if (distance <= 1e-9) return 0;
  const trackHeading = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
  const groundSpeed = solveGroundTrack(
    airSpeed,
    trackHeading,
    windSpeed,
    windFromDirection,
  ).groundSpeed;
  return groundSpeed > 1e-9 ? distance / groundSpeed * 60 : Number.POSITIVE_INFINITY;
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
  windSpeed: number = 0,
  windFromDirection: number = 0,
): WaypointStep {
  const waypointX = clamp(state.x + clamp(action[0] ?? 0, -1, 1) * halfWidth, centerX-halfWidth, centerX+halfWidth);
  const waypointY = clamp(state.y + clamp(action[1] ?? 0, -1, 1) * halfHeight, centerY-halfHeight, centerY+halfHeight);
  return advanceTowardWaypoint(
    state,
    { x: waypointX, y: waypointY },
    maxSpeed,
    dtMinutes,
    windSpeed,
    windFromDirection,
  );
}
