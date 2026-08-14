import test from 'node:test';
import assert from 'node:assert/strict';
import { SIGMAPlanner } from '../src/engine/sigmaAlgorithm.ts';
import type { ScenarioConfig, HelicopterState } from '../src/types/simulation.ts';
import type { BayesianGrid } from '../src/engine/bayesianGrid.ts';

const config: ScenarioConfig = {
  searchAreaWidth: 40, searchAreaHeight: 40, searchAreaCenterX: 0, searchAreaCenterY: 0,
  datumX: 0, datumY: 0, sigmaDatumX: 3, sigmaDatumY: 3,
  meanHeading: 0, meanSpeed: 15, sigmaT: 10, sigmaHeading: 10, sigmaSpeed: 3,
  windDirection: 0, windSpeed: 0, sigmaWindSpeed: 0, sigmaWindDirection: 0,
  frigateX: 0, frigateY: 0, sigmaFrigatePosition: 0,
  helicoMaxSpeed: 60, sigmaHelicoSpeed: 0, helicoEndurance: 180, bingoFuelBuffer: 20,
  sigmaRouteDrift: 3, sigmaSpeedDrift: 1, radarBaseRange: 5, gridCellSize: 1,
  dt: 1, numIterations: 1, strategy: 'SIGMA',
};

function twoLobeGrid(westProbability: number, eastProbability: number): BayesianGrid {
  return {
    cells: [[
      { x: -12, y: 0, pBayesianEvolved: westProbability, scanMemory: 0 },
      { x: 12, y: 0, pBayesianEvolved: eastProbability, scanMemory: 0 },
    ]],
  } as BayesianGrid;
}

test('SIGMA replans from the updated belief before completing an area-wide sweep', () => {
  const planner = new SIGMAPlanner(config);
  let state: HelicopterState = {
    x: 0, y: 0, heading: 0, speed: 0, fuelRemaining: 180, status: 'SEARCHING',
  };
  state = planner.planStep(state, twoLobeGrid(0.65, 0.35), 1, 1);
  assert.ok(state.x < 0, 'first action must exploit the western dominant lobe');
  const eastDominant = twoLobeGrid(0.05, 0.95);
  state = planner.planStep(state, eastDominant, 2, 1);
  state = planner.planStep(state, eastDominant, 3, 1);
  state = planner.planStep(state, eastDominant, 4, 1);
  const beforeReplan = state.x;
  state = planner.planStep(state, eastDominant, 5, 1);
  assert.ok(state.x > beforeReplan, 'the bounded belief action must replan east by minute five');
});

test('SIGMA keeps the waypoint inside the geofence', () => {
  const planner = new SIGMAPlanner(config);
  const state: HelicopterState = {
    x: 19.5, y: 0, heading: 90, speed: 0, fuelRemaining: 180, status: 'SEARCHING',
  };
  const next = planner.planStep(state, twoLobeGrid(0.05, 0.95), 1, 1);
  assert.ok(next.x <= 20 && next.x >= -20 && next.y <= 20 && next.y >= -20);
});

test('SIGMA prioritizes safe return before belief-space search', () => {
  const planner = new SIGMAPlanner(config);
  const state: HelicopterState = {
    x: 10, y: 0, heading: 90, speed: 60, fuelRemaining: 25, status: 'SEARCHING',
  };
  const next = planner.planStep(state, twoLobeGrid(0.95, 0.05), 1, 1);
  assert.equal(next.status, 'BINGO_RETURN');
  assert.ok(next.x < state.x);
});
