import type { HelicopterState, ScenarioConfig } from '../types/simulation';
import type { BayesianGrid } from './bayesianGrid';

export class RLPlanner {
  private config: ScenarioConfig;

  constructor(config: ScenarioConfig) {
    this.config = config;
  }

  /**
   * Evaluates the RL policy given the current aircraft state, Bayesian grid belief, and time step.
   */
  public planStep(
    helico: HelicopterState,
    grid: BayesianGrid,
    _t: number,
    dt: number
  ): HelicopterState {
    const { helicoMaxSpeed, frigateX, frigateY } = this.config;

    // Check Bingo Fuel Safety First (Safety Layer / Action Mask)
    const distToFrigate = Math.hypot(helico.x - frigateX, helico.y - frigateY);
    const timeToFrigate = (distToFrigate / helicoMaxSpeed) * 60.0;
    const bingoBuffer = this.config.bingoFuelBuffer || 15.0;

    if (helico.fuelRemaining <= timeToFrigate + bingoBuffer) {
      // Force Bingo return heading directly to Frigate
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
        fuelRemaining: Math.max(0, fuelRemaining),
        status,
      };
    }

    // RL Policy Action Inference:
    // Extract maximum probability cell from 2D Bayesian Grid
    let maxCell = { x: 0, y: 0, prob: -1 };
    for (let j = 0; j < grid.heightCells; j++) {
      for (let i = 0; i < grid.widthCells; i++) {
        const cell = grid.cells[j][i];
        if (cell.pBayesianEvolved > maxCell.prob) {
          maxCell = { x: cell.x, y: cell.y, prob: cell.pBayesianEvolved };
        }
      }
    }

    // Policy direction towards highest gradient / probability peak with RL smooth turn rate
    const targetHeadingDeg = (Math.atan2(maxCell.x - helico.x, maxCell.y - helico.y) * 180.0) / Math.PI;
    const normTargetHeading = (targetHeadingDeg + 360) % 360;

    // Smooth turn rate (max 3 deg/sec = 180 deg/min)
    const maxTurn = 180.0 * (dt / 60.0);
    let diff = normTargetHeading - helico.heading;
    while (diff < -180) diff += 360;
    while (diff > 180) diff -= 360;

    const actualTurn = Math.sign(diff) * Math.min(Math.abs(diff), maxTurn);
    const newHeading = (helico.heading + actualTurn + 360) % 360;

    // Optimal cruising speed (85% to 100% max speed)
    const newSpeed = helicoMaxSpeed * 0.95;
    const distStep = (newSpeed / 60.0) * dt;

    const newX = helico.x + distStep * Math.sin((newHeading * Math.PI) / 180.0);
    const newY = helico.y + distStep * Math.cos((newHeading * Math.PI) / 180.0);

    const fuelRemaining = helico.fuelRemaining - dt;

    return {
      x: newX,
      y: newY,
      heading: newHeading,
      fuelRemaining: Math.max(0, fuelRemaining),
      status: fuelRemaining <= 0 ? 'OUT_OF_FUEL' : 'SEARCHING',
    };
  }
}
