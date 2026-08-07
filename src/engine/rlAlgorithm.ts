import * as ort from 'onnxruntime-web';
import type { HelicopterState, ScenarioConfig } from '../types/simulation';
import type { BayesianGrid } from './bayesianGrid';
import {
  applyRelativeWaypoint,
  buildHybridVector,
  computeFuelMargin,
  normalizedEntropy,
  scaleBeliefForObservation,
} from './missionContract';
import { SharedAsyncResource } from './sharedAsyncResource';

const policySessionResource = new SharedAsyncResource<ort.InferenceSession | null>();

/** Browser counterpart of the canonical Python v2.3.1 waypoint contract. */
export class RLPlanner {
  private config: ScenarioConfig;
  private session: ort.InferenceSession | null = null;
  private initPromise: Promise<void>;

  constructor(config: ScenarioConfig) {
    this.config = config;
    this.initPromise = this.initOnnxSession();
  }

  public async init(): Promise<void> {
    await this.initPromise;
  }

  private async initOnnxSession(): Promise<void> {
    this.session = await policySessionResource.get(async () => {
      try {
        const session = await ort.InferenceSession.create('/models/baysian_patrol_policy.onnx');
        console.log('Session ONNX Runtime Web partagée chargée pour la stratégie hybride.');
        return session;
      } catch (error) {
        console.warn('Modèle ONNX indisponible : expert bayésien de secours utilisé.', error);
        return null;
      }
    });
  }

  private safeReturn(helico: HelicopterState, dt: number): HelicopterState | null {
    const { frigateX, frigateY, helicoMaxSpeed, bingoFuelBuffer } = this.config;
    const dx = frigateX-helico.x;
    const dy = frigateY-helico.y;
    const distance = Math.hypot(dx, dy);
    const metrics = computeFuelMargin(helico.fuelRemaining, distance, helicoMaxSpeed, bingoFuelBuffer || 15);
    if (metrics.marginMinutes > dt && helico.status !== 'BINGO_RETURN') return null;
    const maxTravel = helicoMaxSpeed/60*dt;
    if (distance <= Math.max(1, maxTravel)) {
      return {
        x: frigateX, y: frigateY, heading: helico.heading, speed: 0,
        fuelRemaining: Math.max(0, helico.fuelRemaining-dt), status: 'SAFE_RTB',
      };
    }
    const heading = (Math.atan2(dx, dy)*180/Math.PI+360)%360;
    const radians = heading*Math.PI/180;
    return {
      x: helico.x+maxTravel*Math.sin(radians),
      y: helico.y+maxTravel*Math.cos(radians),
      heading, speed: helicoMaxSpeed,
      fuelRemaining: Math.max(0, helico.fuelRemaining-dt),
      status: 'BINGO_RETURN',
    };
  }

  private observation(helico: HelicopterState, grid: BayesianGrid, elapsedMinutes: number): {
    gridData: Float32Array;
    vectorData: Float32Array;
    expertAction: [number, number];
  } {
    const gridDim = 32;
    const rawProbability = grid.getFlatBayesianEvolvedProbs();
    const scaledProbability = scaleBeliefForObservation(rawProbability);
    const gridData = new Float32Array(2*gridDim*gridDim);
    const stepI = grid.widthCells/gridDim;
    const stepJ = grid.heightCells/gridDim;
    let peak = { x: 0, y: 0, probability: -1 };

    for (let j=0; j<gridDim; j++) {
      for (let i=0; i<gridDim; i++) {
        const srcI = Math.min(grid.widthCells-1, Math.floor(i*stepI));
        const srcJ = Math.min(grid.heightCells-1, Math.floor(j*stepJ));
        const sourceIndex = srcJ*grid.widthCells+srcI;
        const cell = grid.cells[srcJ][srcI];
        gridData[j*gridDim+i] = scaledProbability[sourceIndex] ?? 0;
        gridData[gridDim*gridDim+j*gridDim+i] = cell.scanMemory;
        if (cell.pBayesianEvolved > peak.probability) {
          peak = { x: cell.x, y: cell.y, probability: cell.pBayesianEvolved };
        }
      }
    }

    const halfWidth = this.config.searchAreaWidth/2;
    const halfHeight = this.config.searchAreaHeight/2;
    const vectorData = new Float32Array(buildHybridVector({
      helicoX: helico.x, helicoY: helico.y, headingDeg: helico.heading,
      speed: helico.speed, maxSpeed: this.config.helicoMaxSpeed,
      fuelRemaining: helico.fuelRemaining, maxFuel: this.config.helicoEndurance,
      frigateX: this.config.frigateX, frigateY: this.config.frigateY,
      halfWidth, halfHeight, peakX: peak.x, peakY: peak.y,
      entropy: normalizedEntropy(rawProbability), elapsedMinutes,
      bingoBuffer: this.config.bingoFuelBuffer || 15,
    }));
    const expertAction: [number, number] = [
      Math.max(-1, Math.min(1, (peak.x-helico.x)/halfWidth)),
      Math.max(-1, Math.min(1, (peak.y-helico.y)/halfHeight)),
    ];
    return { gridData, vectorData, expertAction };
  }

  public async planStepAsync(helico: HelicopterState, grid: BayesianGrid, t: number, dt: number): Promise<HelicopterState> {
    const returnState = this.safeReturn(helico, dt);
    if (returnState) return returnState;
    const observation = this.observation(helico, grid, t);
    let action: [number, number] = observation.expertAction;
    if (this.session) {
      try {
        const outputs = await this.session.run({
          grid: new ort.Tensor('float32', observation.gridData, [1, 2, 32, 32]),
          vector: new ort.Tensor('float32', observation.vectorData, [1, 10]),
        });
        const values = outputs.action?.data as Float32Array | undefined;
        if (values && values.length >= 2) action = [values[0], values[1]];
      } catch (error) {
        console.warn('Inférence ONNX échouée : expert de secours utilisé.', error);
      }
    }
    const next = applyRelativeWaypoint(
      helico, action, this.config.searchAreaWidth/2, this.config.searchAreaHeight/2,
      this.config.helicoMaxSpeed, dt, this.config.searchAreaCenterX, this.config.searchAreaCenterY,
    );
    return { ...next, status: next.fuelRemaining <= 0 ? 'OUT_OF_FUEL' : 'SEARCHING' };
  }

  public planStep(helico: HelicopterState, grid: BayesianGrid, t: number, dt: number): HelicopterState {
    const returnState = this.safeReturn(helico, dt);
    if (returnState) return returnState;
    const action = this.observation(helico, grid, t).expertAction;
    const next = applyRelativeWaypoint(
      helico, action, this.config.searchAreaWidth/2, this.config.searchAreaHeight/2,
      this.config.helicoMaxSpeed, dt, this.config.searchAreaCenterX, this.config.searchAreaCenterY,
    );
    return { ...next, status: next.fuelRemaining <= 0 ? 'OUT_OF_FUEL' : 'SEARCHING' };
  }
}
