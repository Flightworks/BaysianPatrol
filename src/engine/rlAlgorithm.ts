import * as ort from 'onnxruntime-web';
import type { HelicopterState, ScenarioConfig } from '../types/simulation';
import type { BayesianGrid } from './bayesianGrid';

export class RLPlanner {
  private config: ScenarioConfig;
  private session: ort.InferenceSession | null = null;
  private initPromise: Promise<void>;

  constructor(config: ScenarioConfig) {
    this.config = config;
    this.initPromise = this.initOnnxSession();
  }

  public async init() {
    await this.initPromise;
  }

  private async initOnnxSession() {
    try {
      // Load trained ONNX model with cache busting parameter
      this.session = await ort.InferenceSession.create(`/models/baysian_patrol_policy.onnx?t=${Date.now()}`);
      console.log('Session ONNX Runtime Web chargée avec succès pour le Modèle RL PPO.');
    } catch (err) {
      console.warn("Chargement session ONNX non prêt ou modèle non trouvé, utilisation du planificateur de secours:", err);
    }
  }

  /**
   * Evaluates the RL policy given current aircraft state, Bayesian grid belief, and time step.
   */
  public async planStepAsync(
    helico: HelicopterState,
    grid: BayesianGrid,
    _t: number,
    dt: number
  ): Promise<HelicopterState> {
    const { helicoMaxSpeed, frigateX, frigateY, searchAreaWidth, searchAreaHeight, radarBaseRange, windSpeed, windDirection } = this.config;

    // 1. Action Masking & Safety Layer: Check Bingo Fuel
    const distToFrigate = Math.hypot(helico.x - frigateX, helico.y - frigateY);
    const timeToFrigate = (distToFrigate / helicoMaxSpeed) * 60.0;
    const bingoBuffer = this.config.bingoFuelBuffer || 15.0;

    if (helico.fuelRemaining <= timeToFrigate + bingoBuffer) {
      const bingoHeading = (Math.atan2(frigateX - helico.x, frigateY - helico.y) * 180.0) / Math.PI;
      const normalizedHeading = (bingoHeading + 360) % 360;

      const fuelRemaining = helico.fuelRemaining - dt;
      const distStep = (helicoMaxSpeed / 60.0) * dt;

      const newX = helico.x + distStep * Math.sin((normalizedHeading * Math.PI) / 180.0);
      const newY = helico.y + distStep * Math.cos((normalizedHeading * Math.PI) / 180.0);

      const status = fuelRemaining <= 0 ? 'OUT_OF_FUEL' : 'BINGO_RETURN';

      return {
        x: newX,
        y: newY,
        heading: normalizedHeading,
        speed: helicoMaxSpeed,
        fuelRemaining: Math.max(0, fuelRemaining),
        status,
      };
    }

    // 2. Prepare 2D Grid & State Vector Observation for Neural Network Policy
    const gridDim = 32;
    const gridData = new Float32Array(2 * gridDim * gridDim);
    
    const stepI = grid.widthCells / gridDim;
    const stepJ = grid.heightCells / gridDim;

    for (let j = 0; j < gridDim; j++) {
      for (let i = 0; i < gridDim; i++) {
        const srcI = Math.min(grid.widthCells - 1, Math.floor(i * stepI));
        const srcJ = Math.min(grid.heightCells - 1, Math.floor(j * stepJ));
        const cell = grid.cells[srcJ][srcI];

        // Channel 0: P(x,y,t)
        gridData[0 * gridDim * gridDim + j * gridDim + i] = cell.pBayesianEvolved;
        // Channel 1: Scanned mask
        gridData[1 * gridDim * gridDim + j * gridDim + i] = cell.scanMemory;
      }
    }

    // Vector Observation (10-dim)
    const xNorm = helico.x / (searchAreaWidth / 2);
    const yNorm = helico.y / (searchAreaHeight / 2);
    const headingRad = (helico.heading * Math.PI) / 180.0;
    const headingSin = Math.sin(headingRad);
    const headingCos = Math.cos(headingRad);
    const speedNorm = Math.max(0, Math.min(1, helico.speed / helicoMaxSpeed));
    const fuelRatio = helico.fuelRemaining / (this.config.helicoEndurance || 180.0);
    const distFrigateNorm = distToFrigate / searchAreaWidth;
    
    const windRad = (windDirection * Math.PI) / 180.0;
    const windXNorm = (windSpeed * Math.sin(windRad)) / 50.0;
    const windYNorm = (windSpeed * Math.cos(windRad)) / 50.0;
    const radarNorm = radarBaseRange / 30.0;

    const vectorData = new Float32Array([
      xNorm, yNorm, headingSin, headingCos, speedNorm,
      fuelRatio, distFrigateNorm, windXNorm, windYNorm, radarNorm
    ]);

    let actionDeltaHeading = 0.0;
    let actionDeltaSpeed = 0.0;

    // 3. Evaluate ONNX Neural Network Policy Weights
    if (this.session) {
      try {
        const gridTensor = new ort.Tensor('float32', gridData, [1, 2, gridDim, gridDim]);
        const vectorTensor = new ort.Tensor('float32', vectorData, [1, 10]);

        const outputs = await this.session.run({ grid: gridTensor, vector: vectorTensor });
        if (outputs && outputs.action && outputs.action.data) {
          const actionData = outputs.action.data as Float32Array;
          actionDeltaHeading = actionData[0] || 0.0;
          actionDeltaSpeed = actionData[1] || 0.0;
        }
      } catch (err) {
        console.warn("Erreur lors de l'inférence ONNX:", err);
      }
    } else {
      // Backup Heuristic if ONNX not ready
      let maxCell = { x: 0, y: 0, prob: -1 };
      for (let j = 0; j < grid.heightCells; j++) {
        for (let i = 0; i < grid.widthCells; i++) {
          const cell = grid.cells[j][i];
          if (cell.pBayesianEvolved > maxCell.prob) {
            maxCell = { x: cell.x, y: cell.y, prob: cell.pBayesianEvolved };
          }
        }
      }
      const targetHeadingDeg = (Math.atan2(maxCell.x - helico.x, maxCell.y - helico.y) * 180.0) / Math.PI;
      const normTargetHeading = (targetHeadingDeg + 360) % 360;
      let diff = normTargetHeading - helico.heading;
      while (diff < -180) diff += 360;
      while (diff > 180) diff -= 360;
      actionDeltaHeading = Math.sign(diff) * Math.min(1.0, Math.abs(diff) / 180.0);
      actionDeltaSpeed = 0.5;
    }

    // Kinematics Update: Max turn rate is 180 deg/min (180 * dt per step)
    const maxTurnRateDeg = 180.0 * dt;
    const deltaHeadingDeg = actionDeltaHeading * maxTurnRateDeg;
    const newHeading = (helico.heading + deltaHeadingDeg + 360) % 360;

    const currentSpeed = Math.min(helicoMaxSpeed, Math.max(60.0, helicoMaxSpeed * (0.8 + 0.2 * actionDeltaSpeed)));
    const distStep = (currentSpeed / 60.0) * dt;

    const newX = helico.x + distStep * Math.sin((newHeading * Math.PI) / 180.0);
    const newY = helico.y + distStep * Math.cos((newHeading * Math.PI) / 180.0);

    const fuelRemaining = helico.fuelRemaining - dt;

    return {
      x: newX,
      y: newY,
      heading: newHeading,
      speed: currentSpeed,
      fuelRemaining: Math.max(0, fuelRemaining),
      status: fuelRemaining <= 0 ? 'OUT_OF_FUEL' : 'SEARCHING',
    };
  }

  // Synchronous fallback wrapper
  public planStep(
    helico: HelicopterState,
    grid: BayesianGrid,
    _t: number,
    dt: number
  ): HelicopterState {
    // For synchronous calls, default to max probability cell target
    let maxCell = { x: 0, y: 0, prob: -1 };
    for (let j = 0; j < grid.heightCells; j++) {
      for (let i = 0; i < grid.widthCells; i++) {
        const cell = grid.cells[j][i];
        if (cell.pBayesianEvolved > maxCell.prob) {
          maxCell = { x: cell.x, y: cell.y, prob: cell.pBayesianEvolved };
        }
      }
    }
    const targetHeadingDeg = (Math.atan2(maxCell.x - helico.x, maxCell.y - helico.y) * 180.0) / Math.PI;
    const normTargetHeading = (targetHeadingDeg + 360) % 360;
    const maxTurn = 180.0 * dt;
    let diff = normTargetHeading - helico.heading;
    while (diff < -180) diff += 360;
    while (diff > 180) diff -= 360;
    const actualTurn = Math.sign(diff) * Math.min(Math.abs(diff), maxTurn);
    const newHeading = (helico.heading + actualTurn + 360) % 360;
    const newSpeed = this.config.helicoMaxSpeed * 0.95;
    const distStep = (newSpeed / 60.0) * dt;

    return {
      x: helico.x + distStep * Math.sin((newHeading * Math.PI) / 180.0),
      y: helico.y + distStep * Math.cos((newHeading * Math.PI) / 180.0),
      heading: newHeading,
      speed: newSpeed,
      fuelRemaining: Math.max(0, helico.fuelRemaining - dt),
      status: helico.fuelRemaining - dt <= 0 ? 'OUT_OF_FUEL' : 'SEARCHING',
    };
  }
}
