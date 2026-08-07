export interface InitialTargetTruthInput {
  datumX: number;
  datumY: number;
  spatialOffsetX: number;
  spatialOffsetY: number;
  timeOffsetMinutes: number;
  speed: number;
  heading: number;
  currentSpeed?: number;
  currentHeading?: number;
}

export interface TargetTruthState {
  x: number;
  y: number;
  speed: number;
  heading: number;
}

export interface TargetAdvanceInput {
  dtMinutes: number;
  headingNoise: number;
  speedNoise: number;
  currentSpeed: number;
  currentHeading: number;
}

const toRad = (degrees: number) => degrees * Math.PI / 180;
const normalizeHeading = (degrees: number) => ((degrees % 360) + 360) % 360;

const velocity = (speed: number, heading: number) => ({
  x: speed * Math.sin(toRad(heading)),
  y: speed * Math.cos(toRad(heading)),
});

export function deriveInitialTargetTruth(input: InitialTargetTruthInput): TargetTruthState {
  const targetVelocity = velocity(input.speed, input.heading);
  const currentVelocity = velocity(input.currentSpeed ?? 0, input.currentHeading ?? 0);
  const timeHours = input.timeOffsetMinutes / 60;

  return {
    x: input.datumX + input.spatialOffsetX + (targetVelocity.x + currentVelocity.x) * timeHours,
    y: input.datumY + input.spatialOffsetY + (targetVelocity.y + currentVelocity.y) * timeHours,
    speed: input.speed,
    heading: normalizeHeading(input.heading),
  };
}

export function advanceTargetTruth(
  state: TargetTruthState,
  input: TargetAdvanceInput,
): TargetTruthState {
  const heading = normalizeHeading(state.heading + input.headingNoise);
  const speed = Math.max(5, Math.min(45, state.speed + input.speedNoise));
  const targetVelocity = velocity(speed, heading);
  const currentVelocity = velocity(input.currentSpeed, input.currentHeading);
  const dtHours = input.dtMinutes / 60;

  return {
    x: state.x + (targetVelocity.x + currentVelocity.x) * dtHours,
    y: state.y + (targetVelocity.y + currentVelocity.y) * dtHours,
    speed,
    heading,
  };
}
