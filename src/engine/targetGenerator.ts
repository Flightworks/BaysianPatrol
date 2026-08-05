import type { ScenarioConfig, TargetState, PathPoint } from '../types/simulation';
import { randomGaussian, degToRad, normalizeAngle } from './random';

export interface RunRealizationParams {
  windSpeed: number;
  windDirection: number;
  datumX: number;
  datumY: number;
  frigateX: number;
  frigateY: number;
  frigateSectorDeg: number;
  helicoSpeed: number;
  initialSpeed: number;
  initialHeading: number;
  departureTimeMinutes: number;
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

  constructor(config: ScenarioConfig, customRealization?: RunRealizationParams) {
    this.config = config;

    if (customRealization) {
      this.realization = customRealization;
    } else {
      // 1. Draw stochastic environmental parameters
      const windSpeed = Math.max(0, randomGaussian(config.windSpeed, config.sigmaWindSpeed || 3.0));
      const windDirection = normalizeAngle(randomGaussian(config.windDirection, config.sigmaWindDirection || 15.0));
      
      // 2. Draw Datum initial position noise
      const datumX = randomGaussian(config.datumX, config.sigmaDatumX);
      const datumY = randomGaussian(config.datumY, config.sigmaDatumY);

      // 3. Draw 360-degree Omnidirectional Frigate Departure Sector
      // Random azimuth angle theta between 0 and 360 degrees
      const frigateSectorDeg = Math.random() * 360.0;
      const sectorRad = degToRad(frigateSectorDeg);

      // Nominal distance R from search area center (with stochastic variance)
      const nominalR = Math.hypot(config.frigateX, config.frigateY);
      const rDist = Math.max(15.0, randomGaussian(nominalR > 0 ? nominalR : 25.0, config.sigmaFrigatePosition || 4.0));

      // Frigate coordinates in randomized sector
      const frigateX = config.searchAreaCenterX + rDist * Math.sin(sectorRad);
      const frigateY = config.searchAreaCenterY + rDist * Math.cos(sectorRad);

      const helicoSpeed = Math.max(80.0, randomGaussian(config.helicoMaxSpeed, config.sigmaHelicoSpeed || 5.0));

      const initialHeading = normalizeAngle(randomGaussian(config.meanHeading, config.sigmaHeading));
      const initialSpeed = Math.max(5.0, randomGaussian(config.meanSpeed, config.sigmaSpeed));
      const departureTimeMinutes = randomGaussian(0, config.sigmaT);

      this.realization = {
        windSpeed,
        windDirection,
        datumX,
        datumY,
        frigateX,
        frigateY,
        frigateSectorDeg,
        helicoSpeed,
        initialSpeed,
        initialHeading,
        departureTimeMinutes,
      };
    }

    // Initial position at t=0
    this.state = {
      x: this.realization.datumX,
      y: this.realization.datumY,
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
    const { windSpeed, windDirection, departureTimeMinutes } = this.realization;

    if (tMinutes >= departureTimeMinutes) {
      const dtHours = dtMinutes / 60.0;

      const headingNoise = randomGaussian(0, sigmaRouteDrift * Math.sqrt(dtHours));
      const speedNoise = randomGaussian(0, sigmaSpeedDrift * Math.sqrt(dtHours));

      this.state.heading = normalizeAngle(this.state.heading + headingNoise);
      this.state.speed = Math.max(5.0, Math.min(45.0, this.state.speed + speedNoise));

      const currentSpeed = windSpeed * 0.025;
      const currentDir = normalizeAngle(windDirection + 180 + 15);

      const targetRad = degToRad(this.state.heading);
      const currentRad = degToRad(currentDir);

      const targetVx = this.state.speed * Math.sin(targetRad);
      const targetVy = this.state.speed * Math.cos(targetRad);

      const currentVx = currentSpeed * Math.sin(currentRad);
      const currentVy = currentSpeed * Math.cos(currentRad);

      const totalVx = targetVx + currentVx;
      const totalVy = targetVy + currentVy;

      this.state.x += totalVx * dtHours;
      this.state.y += totalVy * dtHours;
    }

    this.path.push({ t: tMinutes, x: this.state.x, y: this.state.y });
    return { ...this.state };
  }

  public getPath(): PathPoint[] {
    return [...this.path];
  }
}
