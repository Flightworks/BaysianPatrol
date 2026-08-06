import type { ScenarioConfig, GridCell } from '../types/simulation';
import { degToRad, normalizeAngle } from './random';
import { calculatePdet } from './radarModel';
import type { RadarParams } from './radarModel';

/**
 * Advanced Bayesian Grid Engine supporting 3 grid modes:
 * 1. Classical A Priori P_classical(M_i, t)
 * 2. Standard Bayesian Posterior P_BayesianStandard(M_i, t)
 * 3. Evolved Bayesian Posterior P_BayesianEvolved(M_i, t) with Helicopter-Target Perpendicular Approach Boost
 */
export class BayesianGrid {
  private config: ScenarioConfig;
  public widthCells: number;
  public heightCells: number;
  public cells: GridCell[][];
  
  public probsClassical: number[];
  public probsBayesianStandard: number[];
  public probsBayesianEvolved: number[];
  
  public clearanceStandard: number[]; // Accumulated non-detection factor L_standard(M_i)
  public clearanceEvolved: number[];  // Accumulated non-detection factor L_evolved(M_i)
  
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
    
    this.clearanceStandard = new Array(totalCells).fill(1.0);
    this.clearanceEvolved = new Array(totalCells).fill(1.0);

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
        });
      }
      this.cells.push(row);
    }

    this.updatePriorDensity(0);
  }

  /**
   * Recompute prior density P_classical(M_i, t) and update both Bayesian posteriors.
   */
  public updatePriorDensity(tMinutes: number): void {
    const {
      datumX, datumY, sigmaDatumX, sigmaDatumY,
      meanHeading, meanSpeed, sigmaSpeed, sigmaHeading, sigmaRouteDrift,
      windSpeed, windDirection
    } = this.config;

    const tHours = tMinutes / 60.0;
    
    // Wind surface current drift
    const currentSpeed = windSpeed * 0.025;
    const currentDir = normalizeAngle(windDirection + 180 + 15);
    const currRad = degToRad(currentDir);
    const currVx = currentSpeed * Math.sin(currRad);
    const currVy = currentSpeed * Math.cos(currRad);

    // Target mean velocity vector
    const meanRad = degToRad(meanHeading);
    const meanVx = meanSpeed * Math.sin(meanRad) + currVx;
    const meanVy = meanSpeed * Math.cos(meanRad) + currVy;

    // Mean target position at time t
    const muX = datumX + meanVx * tHours;
    const muY = datumY + meanVy * tHours;

    // Expanding variances along and cross target track
    const sigmaAlong = Math.sqrt(
      Math.pow(sigmaDatumX, 2) +
      Math.pow(sigmaSpeed * tHours, 2) +
      Math.pow(sigmaRouteDrift * 0.3 * Math.sqrt(Math.max(0.01, tHours)), 2) +
      9.0
    );

    const sigmaCross = Math.sqrt(
      Math.pow(sigmaDatumY, 2) +
      Math.pow(meanSpeed * Math.sin(degToRad(sigmaHeading)) * tHours, 2) +
      Math.pow(sigmaRouteDrift * 0.3 * Math.sqrt(Math.max(0.01, tHours)), 2) +
      9.0
    );

    const varAlong = sigmaAlong * sigmaAlong;
    const varCross = sigmaCross * sigmaCross;

    const sinHeading = Math.sin(meanRad);
    const cosHeading = Math.cos(meanRad);
    const normFactor = 1.0 / (2.0 * Math.PI * sigmaAlong * sigmaCross);

    let sumClassical = 0.0;
    let sumStandard = 0.0;
    let sumEvolved = 0.0;
    let idx = 0;

    for (let j = 0; j < this.heightCells; j++) {
      for (let i = 0; i < this.widthCells; i++) {
        const cell = this.cells[j][i];
        
        const dx = cell.x - muX;
        const dy = cell.y - muY;

        const uAlong = dx * sinHeading + dy * cosHeading;
        const uCross = -dx * cosHeading + dy * sinHeading;

        const exponent = -0.5 * ((uAlong * uAlong) / varAlong + (uCross * uCross) / varCross);
        let pPrior = 0.0;
        
        if (exponent > -25.0) {
          pPrior = Math.exp(exponent) * normFactor;
        }

        cell.pClassical = pPrior;
        this.probsClassical[idx] = pPrior;
        sumClassical += pPrior;

        // Standard Bayesian Posterior
        const pStdUnnorm = pPrior * this.clearanceStandard[idx];
        cell.pBayesianStandard = pStdUnnorm;
        this.probsBayesianStandard[idx] = pStdUnnorm;
        sumStandard += pStdUnnorm;

        // Evolved Bayesian Posterior (Perpendicular Approach Boost)
        const pEvoUnnorm = pPrior * this.clearanceEvolved[idx];
        cell.pBayesianEvolved = pEvoUnnorm;
        this.probsBayesianEvolved[idx] = pEvoUnnorm;
        sumEvolved += pEvoUnnorm;

        cell.pPresence = pEvoUnnorm;

        idx++;
      }
    }

    // Normalize all distributions over search grid
    const invClassical = sumClassical > 0 ? 1.0 / sumClassical : 1.0;
    const invStandard = sumStandard > 0 ? 1.0 / sumStandard : 1.0;
    const invEvolved = sumEvolved > 0 ? 1.0 / sumEvolved : 1.0;

    idx = 0;
    for (let j = 0; j < this.heightCells; j++) {
      for (let i = 0; i < this.widthCells; i++) {
        this.cells[j][i].pClassical *= invClassical;
        this.probsClassical[idx] *= invClassical;

        this.cells[j][i].pBayesianStandard *= invStandard;
        this.probsBayesianStandard[idx] *= invStandard;

        this.cells[j][i].pBayesianEvolved *= invEvolved;
        this.probsBayesianEvolved[idx] *= invEvolved;
        
        this.cells[j][i].pPresence = this.cells[j][i].pBayesianEvolved;
        idx++;
      }
    }
  }

  /**
   * Apply Bayesian Update upon radar scan from helicopter position (hx, hy) heading helicoHeadingDeg.
   * Calculates BOTH Standard Bayesian clearance and Evolved Perpendicular-Approach clearance continuously.
   * Includes temporal memory relaxation (targets move, so probability flows back to previously scanned areas).
   */
  public updateBayesianScan(
    hx: number,
    hy: number,
    helicoHeadingDeg: number,
    radarParams: RadarParams,
    dtMinutes: number = 1.0,
    memoryHalfLifeMinutes: number = 20.0
  ): void {
    const { meanHeading } = this.config;
    const rMax = radarParams.baseRange * 1.4;
    let idx = 0;

    // Temporal Memory Decay: Scanned areas relax back towards 1.0 (unscanned state) over time
    const decayFactor = Math.exp((-Math.LN2 / memoryHalfLifeMinutes) * dtMinutes);
    for (let i = 0; i < this.clearanceStandard.length; i++) {
      this.clearanceStandard[i] = 1.0 - (1.0 - this.clearanceStandard[i]) * decayFactor;
      this.clearanceEvolved[i] = 1.0 - (1.0 - this.clearanceEvolved[i]) * decayFactor;
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

              // 1. Standard Bayesian Clearance
              this.clearanceStandard[idx] *= (1.0 - pDetBase);

              // 2. Evolved Bayesian Clearance (Boosted when helicopter route is perpendicular to target track)
              const pDetEvolved = Math.min(0.98, pDetBase * perpApproachMultiplier);
              this.clearanceEvolved[idx] *= (1.0 - pDetEvolved);
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
        const pStdUnnorm = this.probsClassical[idx] * this.clearanceStandard[idx];
        this.cells[j][i].pBayesianStandard = pStdUnnorm;
        this.probsBayesianStandard[idx] = pStdUnnorm;
        sumStandard += pStdUnnorm;

        const pEvoUnnorm = this.probsClassical[idx] * this.clearanceEvolved[idx];
        this.cells[j][i].pBayesianEvolved = pEvoUnnorm;
        this.probsBayesianEvolved[idx] = pEvoUnnorm;
        sumEvolved += pEvoUnnorm;

        idx++;
      }
    }

    const invStandard = sumStandard > 0 ? 1.0 / sumStandard : 1.0;
    const invEvolved = sumEvolved > 0 ? 1.0 / sumEvolved : 1.0;

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
