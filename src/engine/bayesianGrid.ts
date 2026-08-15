import type { ScenarioConfig, GridCell } from '../types/simulation.ts';
import { degToRad, normalizeAngle } from './random.ts';
import { calculatePdet } from './radarModel.ts';
import type { RadarParams } from './radarModel.ts';
import { propagateGridBelief } from './beliefTransition.ts';

/**
 * Advanced Bayesian Grid Engine supporting 3 grid modes:
 * 1. Classical A Priori P_classical(M_i, t)
 * 2. Standard Bayesian Posterior P_BayesianStandard(M_i, t)
 * 3. Evolved Bayesian Posterior P_BayesianEvolved(M_i, t) with Helicopter-Target Perpendicular Approach Boost
 */
export class BayesianGrid {
  private config: ScenarioConfig;
  private beliefInitialized = false;
  private lastPredictionMinutes = 0;
  public widthCells: number;
  public heightCells: number;
  public cells: GridCell[][];
  
  public probsClassical: number[];
  public probsBayesianStandard: number[];
  public probsBayesianEvolved: number[];
  
  public minX: number;
  public maxX: number;
  public minY: number;
  public maxY: number;

  constructor(config: ScenarioConfig) {
    this.config = config;
    const { searchAreaWidth, searchAreaHeight, searchAreaCenterX, searchAreaCenterY, gridCellSize } = config;

    this.minX = searchAreaCenterX - searchAreaWidth / 2;
    this.maxX = searchAreaCenterX + searchAreaWidth / 2;
    this.minY = searchAreaCenterY - searchAreaHeight / 2;
    this.maxY = searchAreaCenterY + searchAreaHeight / 2;

    this.widthCells = Math.ceil(searchAreaWidth / gridCellSize);
    this.heightCells = Math.ceil(searchAreaHeight / gridCellSize);

    const totalCells = this.widthCells * this.heightCells;
    this.cells = [];
    this.probsClassical = new Array(totalCells).fill(0);
    this.probsBayesianStandard = new Array(totalCells).fill(0);
    this.probsBayesianEvolved = new Array(totalCells).fill(0);
    

    for (let j = 0; j < this.heightCells; j++) {
      const row: GridCell[] = [];
      const cellY = this.minY + (j + 0.5) * gridCellSize;
      for (let i = 0; i < this.widthCells; i++) {
        const cellX = this.minX + (i + 0.5) * gridCellSize;
        row.push({
          i,
          j,
          x: cellX,
          y: cellY,
          pPresence: 0,
          pClassical: 0,
          pBayesianStandard: 0,
          pBayesianEvolved: 0,
          pDet: 0,
          scanned: false,
          scanMemory: 0,
        });
      }
      this.cells.push(row);
    }

    this.updatePriorDensity(0);
  }

  /**
   * Recompute the analytical classical layer, then predict both Bayesian
   * posteriors from their previous state through the target transition model.
   */
  public updatePriorDensity(tMinutes: number): void {
    const {
      datumX, datumY, sigmaDatumX, sigmaDatumY, sigmaT,
      meanHeading, meanSpeed, sigmaSpeed, sigmaHeading, sigmaRouteDrift,
      windSpeed, windDirection,
    } = this.config;
    const tHours = tMinutes / 60;
    const currentSpeed = windSpeed * 0.025;
    const currentDir = normalizeAngle(windDirection + 180 + 15);
    const currRad = degToRad(currentDir);
    const currVx = currentSpeed * Math.sin(currRad);
    const currVy = currentSpeed * Math.cos(currRad);
    const meanRad = degToRad(meanHeading);
    const meanVx = meanSpeed * Math.sin(meanRad) + currVx;
    const meanVy = meanSpeed * Math.cos(meanRad) + currVy;
    const muX = datumX + meanVx * tHours;
    const muY = datumY + meanVy * tHours;
    const routeVariance = Math.pow(sigmaRouteDrift * 0.3, 2) * Math.max(0.01, tHours);
    const crossSpeed = meanSpeed * Math.sin(degToRad(sigmaHeading));
    const processAlongVariance = Math.pow(sigmaSpeed * tHours, 2) + routeVariance;
    const processCrossVariance = Math.pow(crossSpeed * tHours, 2) + routeVariance;
    const sinHeading = Math.sin(meanRad);
    const cosHeading = Math.cos(meanRad);
    const sigmaTimeHoursSquared = Math.pow(sigmaT / 60, 2);
    const temporalAlongVariance = sigmaSpeed * sigmaSpeed * sigmaTimeHoursSquared;
    const temporalCrossVariance = crossSpeed * crossSpeed * sigmaTimeHoursSquared;
    const temporalVarianceX = sigmaTimeHoursSquared * meanVx * meanVx
      + temporalAlongVariance * sinHeading * sinHeading
      + temporalCrossVariance * cosHeading * cosHeading;
    const temporalVarianceY = sigmaTimeHoursSquared * meanVy * meanVy
      + temporalAlongVariance * cosHeading * cosHeading
      + temporalCrossVariance * sinHeading * sinHeading;
    const temporalCovarianceXY = sigmaTimeHoursSquared * meanVx * meanVy
      + (temporalAlongVariance - temporalCrossVariance) * sinHeading * cosHeading;
    const varianceX = sigmaDatumX * sigmaDatumX + 9 + temporalVarianceX
      + processAlongVariance * sinHeading * sinHeading
      + processCrossVariance * cosHeading * cosHeading;
    const varianceY = sigmaDatumY * sigmaDatumY + 9 + temporalVarianceY
      + processAlongVariance * cosHeading * cosHeading
      + processCrossVariance * sinHeading * sinHeading;
    const covarianceXY = temporalCovarianceXY
      + (processAlongVariance - processCrossVariance) * sinHeading * cosHeading;
    const determinant = varianceX * varianceY - covarianceXY * covarianceXY;
    if (!(determinant > 0) || !Number.isFinite(determinant)) {
      throw new Error('Classical belief covariance must be positive definite');
    }
    const normFactor = 1 / (2 * Math.PI * Math.sqrt(determinant));

    let sumClassical = 0;
    let index = 0;
    for (let j = 0; j < this.heightCells; j += 1) {
      for (let i = 0; i < this.widthCells; i += 1) {
        const cell = this.cells[j][i];
        const dx = cell.x - muX;
        const dy = cell.y - muY;
        const quadratic = (
          varianceY * dx * dx
          - 2 * covarianceXY * dx * dy
          + varianceX * dy * dy
        ) / determinant;
        const exponent = -0.5 * quadratic;
        const probability = exponent > -25 ? Math.exp(exponent) * normFactor : 0;
        cell.pClassical = probability;
        this.probsClassical[index] = probability;
        sumClassical += probability;
        index += 1;
      }
    }
    if (!(sumClassical > 0)) throw new Error('Classical belief contains no mass');
    for (index = 0; index < this.probsClassical.length; index += 1) {
      this.probsClassical[index] /= sumClassical;
    }

    if (!this.beliefInitialized || tMinutes < this.lastPredictionMinutes) {
      this.probsBayesianStandard = [...this.probsClassical];
      this.probsBayesianEvolved = [...this.probsClassical];
      this.beliefInitialized = true;
    } else if (tMinutes > this.lastPredictionMinutes) {
      const previousHours = this.lastPredictionMinutes / 60;
      const deltaHours = (tMinutes - this.lastPredictionMinutes) / 60;
      const routeVarianceAt = (hours: number): number =>
        Math.pow(sigmaRouteDrift * 0.3, 2) * Math.max(0.01, hours);
      const alongVarianceAt = (hours: number): number =>
        Math.pow(sigmaSpeed * hours, 2) + routeVarianceAt(hours);
      const crossVarianceAt = (hours: number): number =>
        Math.pow(crossSpeed * hours, 2) + routeVarianceAt(hours);
      const meanGroundSpeed = Math.hypot(meanVx, meanVy);
      const transitionHeading = meanGroundSpeed > 1e-12
        ? normalizeAngle(Math.atan2(meanVx, meanVy) * 180 / Math.PI)
        : normalizeAngle(meanHeading);
      const transition = {
        width: this.widthCells,
        height: this.heightCells,
        cellSize: this.config.gridCellSize,
        headingDeg: transitionHeading,
        distance: meanGroundSpeed * deltaHours,
        sigmaAlong: Math.sqrt(Math.max(0, alongVarianceAt(tHours) - alongVarianceAt(previousHours))),
        sigmaCross: Math.sqrt(Math.max(0, crossVarianceAt(tHours) - crossVarianceAt(previousHours))),
      };
      this.probsBayesianStandard = propagateGridBelief(this.probsBayesianStandard, transition);
      this.probsBayesianEvolved = propagateGridBelief(this.probsBayesianEvolved, transition);
    }

    this.lastPredictionMinutes = tMinutes;
    index = 0;
    for (let j = 0; j < this.heightCells; j += 1) {
      for (let i = 0; i < this.widthCells; i += 1) {
        const cell = this.cells[j][i];
        cell.pClassical = this.probsClassical[index];
        cell.pBayesianStandard = this.probsBayesianStandard[index];
        cell.pBayesianEvolved = this.probsBayesianEvolved[index];
        cell.pPresence = this.probsBayesianEvolved[index];
        index += 1;
      }
    }
  }

  /**
   * Apply Bayesian Update upon radar scan from helicopter position (hx, hy) heading helicoHeadingDeg.
   * Negative evidence remains in the posterior and moves through the target
   * transition model. The half-life below affects only the visual coverage
   * memory, never the Bayesian belief.
   */
  public updateBayesianScan(
    hx: number,
    hy: number,
    helicoHeadingDeg: number,
    radarParams: RadarParams,
    dtMinutes: number = 1.0,
    coverageHalfLifeMinutes: number = 20.0
  ): void {
    const { meanHeading } = this.config;
    const rMax = radarParams.baseRange * 1.4;
    let idx = 0;

    if (!(coverageHalfLifeMinutes > 0) || !Number.isFinite(coverageHalfLifeMinutes)) {
      throw new Error('Coverage half-life must be positive and finite');
    }
    const decayFactor = Math.exp((-Math.LN2 / coverageHalfLifeMinutes) * dtMinutes);
    for (let i = 0; i < this.probsBayesianStandard.length; i++) {
      this.cells[Math.floor(i / this.widthCells)][i % this.widthCells].scanMemory *= decayFactor;
    }

    // Aspect angle multiplier
    const aspectApproachRad = degToRad(normalizeAngle(helicoHeadingDeg - meanHeading));
    const perpApproachMultiplier = 0.70 + 0.50 * Math.abs(Math.sin(aspectApproachRad));

    for (let j = 0; j < this.heightCells; j++) {
      for (let i = 0; i < this.widthCells; i++) {
        const cell = this.cells[j][i];
        const dx = cell.x - hx;
        const dy = cell.y - hy;

        if (Math.abs(dx) <= rMax && Math.abs(dy) <= rMax) {
          const dist = Math.hypot(dx, dy);
          if (dist <= rMax) {
            const bearingRad = Math.atan2(dx, dy);
            const bearingDeg = normalizeAngle((bearingRad * 180.0) / Math.PI);

            const pDetBase = calculatePdet(dist, bearingDeg, meanHeading, radarParams);
            cell.pDet = pDetBase;

            if (pDetBase > 0) {
              cell.scanned = true;
              cell.scanMemory = 1.0;

              // Standard posterior after a negative observation.
              this.probsBayesianStandard[idx] *= (1.0 - pDetBase);

              // Tactical posterior, boosted for a perpendicular sensor approach.
              const pDetEvolved = Math.min(0.98, pDetBase * perpApproachMultiplier);
              this.probsBayesianEvolved[idx] *= (1.0 - pDetEvolved);
            }
          }
        }
        idx++;
      }
    }

    // Re-evaluate both Bayesian Posteriors
    let sumStandard = 0.0;
    let sumEvolved = 0.0;
    idx = 0;

    for (let j = 0; j < this.heightCells; j++) {
      for (let i = 0; i < this.widthCells; i++) {
        const pStdUnnorm = this.probsBayesianStandard[idx];
        this.cells[j][i].pBayesianStandard = pStdUnnorm;
        this.probsBayesianStandard[idx] = pStdUnnorm;
        sumStandard += pStdUnnorm;

        const pEvoUnnorm = this.probsBayesianEvolved[idx];
        this.cells[j][i].pBayesianEvolved = pEvoUnnorm;
        this.probsBayesianEvolved[idx] = pEvoUnnorm;
        sumEvolved += pEvoUnnorm;

        idx++;
      }
    }

    if (!(sumStandard > 0) || !Number.isFinite(sumStandard)
      || !(sumEvolved > 0) || !Number.isFinite(sumEvolved)) {
      throw new Error('Bayesian posterior must retain positive mass');
    }
    const invStandard = 1.0 / sumStandard;
    const invEvolved = 1.0 / sumEvolved;

    idx = 0;
    for (let j = 0; j < this.heightCells; j++) {
      for (let i = 0; i < this.widthCells; i++) {
        this.cells[j][i].pBayesianStandard *= invStandard;
        this.probsBayesianStandard[idx] *= invStandard;

        this.cells[j][i].pBayesianEvolved *= invEvolved;
        this.probsBayesianEvolved[idx] *= invEvolved;

        this.cells[j][i].pPresence = this.cells[j][i].pBayesianEvolved;
        idx++;
      }
    }
  }

  public getFlatClassicalProbs(): number[] {
    return [...this.probsClassical];
  }

  public getFlatBayesianStandardProbs(): number[] {
    return [...this.probsBayesianStandard];
  }

  public getFlatBayesianEvolvedProbs(): number[] {
    return [...this.probsBayesianEvolved];
  }
}
