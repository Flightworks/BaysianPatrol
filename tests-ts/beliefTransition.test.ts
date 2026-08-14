import test from 'node:test';
import assert from 'node:assert/strict';
import { propagateGridBelief } from '../src/engine/beliefTransition.ts';
import { BayesianGrid } from '../src/engine/bayesianGrid.ts';
import type { ScenarioConfig } from '../src/types/simulation.ts';

const config: ScenarioConfig = {
  searchAreaWidth: 40, searchAreaHeight: 40, searchAreaCenterX: 0, searchAreaCenterY: 0,
  datumX: 0, datumY: 0, sigmaDatumX: 2, sigmaDatumY: 2,
  meanHeading: 90, meanSpeed: 6, sigmaT: 0, sigmaHeading: 0, sigmaSpeed: 0,
  windDirection: 0, windSpeed: 0, sigmaWindSpeed: 0, sigmaWindDirection: 0,
  frigateX: 0, frigateY: 0, sigmaFrigatePosition: 0,
  helicoMaxSpeed: 100, sigmaHelicoSpeed: 0, helicoEndurance: 180, bingoFuelBuffer: 20,
  sigmaRouteDrift: 0, radarBaseRange: 5, gridCellSize: 1,
  dt: 1, numIterations: 1, strategy: 'SIGMA',
};

test('transition moves a point belief along the target motion model', () => {
  const source = new Array(9 * 9).fill(0);
  source[4 * 9 + 3] = 1;
  const predicted = propagateGridBelief(source, {
    width: 9,
    height: 9,
    cellSize: 1,
    headingDeg: 90,
    distance: 2,
    sigmaAlong: 0,
    sigmaCross: 0,
  });
  assert.ok(predicted[4 * 9 + 5] > 0.999999);
  assert.ok(Math.abs(predicted.reduce((sum, probability) => sum + probability, 0) - 1) < 1e-12);
});

test('transition is deterministic and preserves a normalized distribution', () => {
  const source = Array.from({ length: 100 }, (_, index) => index + 1);
  const total = source.reduce((sum, value) => sum + value, 0);
  const normalized = source.map(value => value / total);
  const options = {
    width: 10, height: 10, cellSize: 1, headingDeg: 35,
    distance: 0.8, sigmaAlong: 0.4, sigmaCross: 0.2,
  };
  const first = propagateGridBelief(normalized, options);
  const second = propagateGridBelief(normalized, options);
  assert.deepEqual(first, second);
  assert.ok(Math.abs(first.reduce((sum, probability) => sum + probability, 0) - 1) < 1e-12);
});

test('BayesianGrid predicts from the previous posterior instead of rebuilding it from the datum', () => {
  const grid = new BayesianGrid(config);
  let selectedIndex = 0;
  let selectedDistance = Number.POSITIVE_INFINITY;
  for (let j = 0; j < grid.heightCells; j += 1) {
    for (let i = 0; i < grid.widthCells; i += 1) {
      const index = j * grid.widthCells + i;
      const cell = grid.cells[j][i];
      const distance = Math.hypot(cell.x + 5, cell.y);
      if (distance < selectedDistance) {
        selectedDistance = distance;
        selectedIndex = index;
      }
      cell.pBayesianStandard = 0;
      cell.pBayesianEvolved = 0;
    }
  }
  grid.probsBayesianStandard.fill(0);
  grid.probsBayesianEvolved.fill(0);
  grid.probsBayesianStandard[selectedIndex] = 1;
  grid.probsBayesianEvolved[selectedIndex] = 1;
  const selectedCell = grid.cells[Math.floor(selectedIndex / grid.widthCells)][selectedIndex % grid.widthCells];
  selectedCell.pBayesianStandard = 1;
  selectedCell.pBayesianEvolved = 1;

  grid.updatePriorDensity(60);

  const meanX = grid.cells.flat().reduce(
    (sum, cell) => sum + cell.x * cell.pBayesianEvolved,
    0,
  );
  assert.ok(Math.abs(meanX - (selectedCell.x + 6)) < 0.15);
});

test('a negative observation is propagated with the moving target belief', () => {
  const baseline = new BayesianGrid(config);
  const observed = new BayesianGrid(config);
  observed.updateBayesianScan(
    0,
    0,
    90,
    { baseRange: 5, windSpeed: 0, windDirection: 0 },
    1,
    20,
  );

  baseline.updatePriorDensity(60);
  observed.updatePriorDensity(60);
  const massAroundPredictedPosition = (grid: BayesianGrid): number => grid.cells.flat().reduce(
    (sum, cell) => Math.hypot(cell.x - 6, cell.y) <= 3
      ? sum + cell.pBayesianEvolved
      : sum,
    0,
  );

  const baselineMass = massAroundPredictedPosition(baseline);
  const observedMass = massAroundPredictedPosition(observed);
  assert.ok(observedMass < baselineMass * 0.85);
});

test('fractional grid deposition preserves the mean with bounded discretization variance', () => {
  const source = new Array(11 * 11).fill(0);
  source[5 * 11 + 5] = 1;
  const predicted = propagateGridBelief(source, {
    width: 11, height: 11, cellSize: 1, headingDeg: 90,
    distance: 0.5, sigmaAlong: 0, sigmaCross: 0,
  });
  const meanX = predicted.reduce((sum, probability, index) => sum + (index % 11) * probability, 0);
  const varianceX = predicted.reduce(
    (sum, probability, index) => sum + Math.pow((index % 11) - meanX, 2) * probability,
    0,
  );
  assert.ok(Math.abs(meanX - 5.5) < 1e-12);
  assert.ok(varianceX <= 0.25 + 1e-12);
});

test('boundary policy projects outside mass onto the nearest edge cell', () => {
  const source = new Array(3 * 3).fill(0);
  source[1 * 3 + 2] = 1;
  const predicted = propagateGridBelief(source, {
    width: 3, height: 3, cellSize: 1, headingDeg: 90,
    distance: 2, sigmaAlong: 0, sigmaCross: 0,
  });
  assert.ok(predicted[1 * 3 + 2] > 0.999999);
});

test('transition rejects non-finite motion parameters', () => {
  const source = new Array(9).fill(1 / 9);
  assert.throws(() => propagateGridBelief(source, {
    width: 3, height: 3, cellSize: 1, headingDeg: 0,
    distance: Number.NaN, sigmaAlong: 0, sigmaCross: 0,
  }), /finite/);
});

test('transition rejects more than 100000 grid cells before allocation', () => {
  const source = new Array(100_001).fill(1);
  assert.throws(() => propagateGridBelief(source, {
    width: 100_001, height: 1, cellSize: 1, headingDeg: 0,
    distance: 0, sigmaAlong: 0, sigmaCross: 0,
  }), /100000/);
});

test('initial belief includes temporal uncertainty along the estimated route', () => {
  const temporalGrid = new BayesianGrid({
    ...config,
    searchAreaWidth: 120,
    searchAreaHeight: 120,
    datumX: 0,
    datumY: 0,
    sigmaDatumX: 1,
    sigmaDatumY: 1,
    meanHeading: 90,
    meanSpeed: 10,
    sigmaT: 60,
    sigmaSpeed: 0,
    sigmaHeading: 0,
    windSpeed: 0,
  });
  const cells = temporalGrid.cells.flat();
  const meanX = cells.reduce((sum, cell) => sum + cell.x * cell.pClassical, 0);
  const meanY = cells.reduce((sum, cell) => sum + cell.y * cell.pClassical, 0);
  const varianceX = cells.reduce((sum, cell) => sum + Math.pow(cell.x - meanX, 2) * cell.pClassical, 0);
  const varianceY = cells.reduce((sum, cell) => sum + Math.pow(cell.y - meanY, 2) * cell.pClassical, 0);
  assert.ok(Math.abs(varianceX - 110) < 2, `varianceX=${varianceX}`);
  assert.ok(Math.abs(varianceY - 10) < 0.5, `varianceY=${varianceY}`);
});

test('classical datum covariance remains aligned to map axes for an oblique route', () => {
  const covarianceGrid = new BayesianGrid({
    ...config,
    searchAreaWidth: 80,
    searchAreaHeight: 80,
    sigmaDatumX: 1,
    sigmaDatumY: 4,
    meanHeading: 45,
    meanSpeed: 0,
  });
  const cells = covarianceGrid.cells.flat();
  const meanX = cells.reduce((sum, cell) => sum + cell.x * cell.pClassical, 0);
  const meanY = cells.reduce((sum, cell) => sum + cell.y * cell.pClassical, 0);
  const varianceX = cells.reduce((sum, cell) => sum + Math.pow(cell.x - meanX, 2) * cell.pClassical, 0);
  const varianceY = cells.reduce((sum, cell) => sum + Math.pow(cell.y - meanY, 2) * cell.pClassical, 0);
  assert.ok(Math.abs(varianceX - 10) < 0.5);
  assert.ok(Math.abs(varianceY - 25) < 0.5);
});

test('Bayesian scan rejects a posterior with no remaining mass', () => {
  const grid = new BayesianGrid(config);
  grid.probsBayesianStandard.fill(0);
  grid.probsBayesianEvolved.fill(0);
  for (const cell of grid.cells.flat()) {
    cell.pBayesianStandard = 0;
    cell.pBayesianEvolved = 0;
  }
  assert.throws(() => grid.updateBayesianScan(
    0, 0, 90, { baseRange: 5, windSpeed: 0, windDirection: 0 }, 1, 20,
  ), /positive mass/);
});

test('classical process covariance keeps the cross term for an oblique route', () => {
  const grid = new BayesianGrid({
    ...config,
    searchAreaWidth: 120,
    searchAreaHeight: 120,
    sigmaDatumX: 1,
    sigmaDatumY: 1,
    meanHeading: 45,
    meanSpeed: 10,
    sigmaSpeed: 4,
    sigmaHeading: 0,
  });
  grid.updatePriorDensity(60);
  const cells = grid.cells.flat();
  const meanX = cells.reduce((sum, cell) => sum + cell.x * cell.pClassical, 0);
  const meanY = cells.reduce((sum, cell) => sum + cell.y * cell.pClassical, 0);
  const covariance = cells.reduce(
    (sum, cell) => sum + (cell.x - meanX) * (cell.y - meanY) * cell.pClassical,
    0,
  );
  assert.ok(Math.abs(covariance - 8) < 0.5);
});

test('incremental transitions accumulate the configured process variance', () => {
  const grid = new BayesianGrid({
    ...config,
    searchAreaWidth: 120,
    searchAreaHeight: 120,
    datumX: 0,
    datumY: 0,
    meanHeading: 90,
    meanSpeed: 0,
    sigmaSpeed: 6,
    sigmaHeading: 0,
  });
  let centerIndex = 0;
  let centerDistance = Number.POSITIVE_INFINITY;
  for (const cell of grid.cells.flat()) {
    const distance = Math.hypot(cell.x, cell.y);
    if (distance < centerDistance) {
      centerDistance = distance;
      centerIndex = cell.j * grid.widthCells + cell.i;
    }
    cell.pBayesianStandard = 0;
    cell.pBayesianEvolved = 0;
  }
  grid.probsBayesianStandard.fill(0);
  grid.probsBayesianEvolved.fill(0);
  grid.probsBayesianStandard[centerIndex] = 1;
  grid.probsBayesianEvolved[centerIndex] = 1;
  grid.cells[Math.floor(centerIndex / grid.widthCells)][centerIndex % grid.widthCells].pBayesianStandard = 1;
  grid.cells[Math.floor(centerIndex / grid.widthCells)][centerIndex % grid.widthCells].pBayesianEvolved = 1;

  grid.updatePriorDensity(30);
  grid.updatePriorDensity(60);
  const cells = grid.cells.flat();
  const meanX = cells.reduce((sum, cell) => sum + cell.x * cell.pBayesianEvolved, 0);
  const varianceX = cells.reduce(
    (sum, cell) => sum + Math.pow(cell.x - meanX, 2) * cell.pBayesianEvolved,
    0,
  );
  assert.ok(varianceX >= 36 && varianceX <= 36.5, `varianceX=${varianceX}`);
});
