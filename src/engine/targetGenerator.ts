import type { ScenarioConfig, TargetState, PathPoint } from '../types/simulation.ts';
import { degToRad, normalizeAngle, SeededRandom } from './random.ts';
import { advanceTargetTruth, deriveInitialTargetTruth } from './targetUncertainty.ts';

export interface RunRealizationParams {
  windSpeed: number;
  windDirection: number;
  datumX: number;
  datumY: number;
  targetInitialX: number;
  targetInitialY: number;
  frigateX: number;
  frigateY: number;
  frigateSectorDeg: number;
  helicoSpeed: number;
  initialSpeed: number;
  initialHeading: number;
  datumTimeOffsetMinutes: number;
  seed: number;
}

/**
 * Generates exact physical Ground Truth target trajectories with
 * per-run 360-degree omnidirectional frigate departure sectors and environmental noise.
 */
export class TargetSim {
  private config: ScenarioConfig;
  private realization: RunRealizationParams;
  private state: TargetState;
  private path: PathPoint[] = [];
  private rng: SeededRandom;

  constructor(config: ScenarioConfig, customRealization?: RunRealizationParams, suiteRng?: SeededRandom) {
    this.config = config;
    const source = suiteRng ?? new SeededRandom(config.monteCarloSeed ?? 2026);

    if (customRealization) {
      this.realization = customRealization;
      this.rng = new SeededRandom(customRealization.seed);
    } else {
      const gaussian = (mean: number, std: number) => source.gaussian(mean, std);
      // 1. Draw stochastic environmental parameters
      const windSpeed = Math.max(0, gaussian(config.windSpeed, config.sigmaWindSpeed || 3.0));
      const windDirection = normalizeAngle(gaussian(config.windDirection, config.sigmaWindDirection || 15.0));
      
      // 2. Keep the observed datum fixed and draw hidden ground truth around it.
      const datumX = config.datumX;
      const datumY = config.datumY;
      const spatialOffsetX = gaussian(0, config.sigmaDatumX);
      const spatialOffsetY = gaussian(0, config.sigmaDatumY);

      // 3. Draw 360-degree Omnidirectional Frigate Departure Sector
      const frigateSectorDeg = source.uniform() * 360.0;
      const sectorRad = degToRad(frigateSectorDeg);

      // Nominal distance R from search area center (with stochastic variance)
      const nominalR = Math.hypot(config.frigateX, config.frigateY);
      const rDist = Math.max(15.0, gaussian(nominalR > 0 ? nominalR : 25.0, config.sigmaFrigatePosition || 4.0));

      // Frigate coordinates in randomized sector
      const frigateX = config.searchAreaCenterX + rDist * Math.sin(sectorRad);
      const frigateY = config.searchAreaCenterY + rDist * Math.cos(sectorRad);

      const helicoSpeed = Math.max(80.0, gaussian(config.helicoMaxSpeed, config.sigmaHelicoSpeed || 5.0));
      const initialHeading = normalizeAngle(gaussian(config.meanHeading, config.sigmaHeading));
      const initialSpeed = Math.max(5.0, gaussian(config.meanSpeed, config.sigmaSpeed));
      const datumTimeOffsetMinutes = gaussian(0, config.sigmaT);
      const initialTruth = deriveInitialTargetTruth({
        datumX,
        datumY,
        spatialOffsetX,
        spatialOffsetY,
        timeOffsetMinutes: datumTimeOffsetMinutes,
        speed: initialSpeed,
        heading: initialHeading,
        currentSpeed: windSpeed * 0.025,
        currentHeading: normalizeAngle(windDirection + 180 + 15),
      });
      const seed = Math.floor(source.uniform()*0xffffffff);

      this.realization = {
        windSpeed,
        windDirection,
        datumX,
        datumY,
        targetInitialX: initialTruth.x,
        targetInitialY: initialTruth.y,
        frigateX,
        frigateY,
        frigateSectorDeg,
        helicoSpeed,
        initialSpeed,
        initialHeading,
        datumTimeOffsetMinutes,
        seed,
      };
      this.rng = new SeededRandom(seed);
    }

    // Hidden ground truth at t=0, distinct from the observed datum.
    this.state = {
      x: this.realization.targetInitialX,
      y: this.realization.targetInitialY,
      speed: this.realization.initialSpeed,
      heading: this.realization.initialHeading,
    };

    this.path.push({ t: 0, x: this.state.x, y: this.state.y });
  }

  public getRealization(): RunRealizationParams {
    return { ...this.realization };
  }

  public step(tMinutes: number, dtMinutes: number): TargetState {
    const { sigmaRouteDrift, sigmaSpeedDrift } = this.config;
    const { windSpeed, windDirection } = this.realization;
    const dtHours = dtMinutes / 60.0;

    this.state = advanceTargetTruth(this.state, {
      dtMinutes,
      headingNoise: this.rng.gaussian(0, sigmaRouteDrift * Math.sqrt(dtHours)),
      speedNoise: this.rng.gaussian(0, sigmaSpeedDrift * Math.sqrt(dtHours)),
      currentSpeed: windSpeed * 0.025,
      currentHeading: normalizeAngle(windDirection + 180 + 15),
    });

    this.path.push({ t: tMinutes, x: this.state.x, y: this.state.y });
    return { ...this.state };
  }

  public getPath(): PathPoint[] {
    return [...this.path];
  }
}

export function buildCanonicalTargetPath(
  config: ScenarioConfig,
  realization: RunRealizationParams,
  endTimeMinutes: number = config.helicoEndurance,
): PathPoint[] {
  const target = new TargetSim(config, realization);
  const steps = Math.ceil(endTimeMinutes / config.dt);
  for (let step = 1; step <= steps; step++) {
    target.step(step * config.dt, config.dt);
  }
  return target.getPath();
}
