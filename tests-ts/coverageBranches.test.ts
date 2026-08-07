import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BranchCommitment,
  buildBayesianCoverageLeg,
} from '../src/engine/coverageBranch.ts';
import type { ScenarioConfig } from '../src/types/simulation.ts';
import type { BayesianGrid } from '../src/engine/bayesianGrid.ts';

const baseConfig: ScenarioConfig = {
  searchAreaWidth: 60, searchAreaHeight: 40, searchAreaCenterX: 0, searchAreaCenterY: 0,
  datumX: 10, datumY: 0, sigmaDatumX: 3, sigmaDatumY: 3,
  meanHeading: 0, meanSpeed: 30, sigmaT: 10, sigmaHeading: 15, sigmaSpeed: 4,
  windDirection: 0, windSpeed: 0, sigmaWindSpeed: 0, sigmaWindDirection: 0,
  frigateX: 0, frigateY: -20, sigmaFrigatePosition: 0,
  helicoMaxSpeed: 120, sigmaHelicoSpeed: 0, helicoEndurance: 180, bingoFuelBuffer: 20,
  sigmaRouteDrift: 3, sigmaSpeedDrift: 1, radarBaseRange: 12, gridCellSize: 2,
  dt: 1, numIterations: 250, strategy: 'TRIO',
};

function fakeGrid(config: ScenarioConfig): BayesianGrid {
  const cells = [];
  const minX = config.searchAreaCenterX - config.searchAreaWidth / 2;
  const minY = config.searchAreaCenterY - config.searchAreaHeight / 2;
  const widthCells = Math.ceil(config.searchAreaWidth / config.gridCellSize);
  const heightCells = Math.ceil(config.searchAreaHeight / config.gridCellSize);
  for (let j = 0; j < heightCells; j++) {
    const row = [];
    for (let i = 0; i < widthCells; i++) {
      row.push({
        i, j,
        x: minX + (i + 0.5) * config.gridCellSize,
        y: minY + (j + 0.5) * config.gridCellSize,
        pBayesianEvolved: 0.001,
        scanMemory: 0.95,
      });
    }
    cells.push(row);
  }
  return { cells } as BayesianGrid;
}

const headingOf = (a: { x: number; y: number }, b: { x: number; y: number }) =>
  (Math.atan2(b.x - a.x, b.y - a.y) * 180 / Math.PI + 360) % 360;

const axialDifference = (a: number, b: number) => {
  const delta = Math.abs(a - b) % 180;
  return Math.min(delta, 180 - delta);
};

test('Bayesian planner creates a radar-sized leg through high posterior and low coverage', () => {
  const config = { ...baseConfig, meanHeading: 20 };
  const grid = fakeGrid(config);
  const preferred = grid.cells[Math.floor(grid.cells.length / 2)][Math.floor(grid.cells[0].length * 0.7)];
  preferred.pBayesianEvolved = 0.3;
  preferred.scanMemory = 0;

  const leg = buildBayesianCoverageLeg(config, grid, { x: config.frigateX, y: config.frigateY });
  const length = Math.hypot(leg.end.x - leg.start.x, leg.end.y - leg.start.y);
  const midpoint = { x: (leg.start.x + leg.end.x) / 2, y: (leg.start.y + leg.end.y) / 2 };

  assert.ok(length >= 2 * config.radarBaseRange);
  assert.ok(axialDifference(headingOf(leg.start, leg.end), config.meanHeading + 90) < 1e-6);
  assert.ok(Math.hypot(midpoint.x - preferred.x, midpoint.y - preferred.y) <= config.gridCellSize * 1.5);
});

test('Bayesian planner excludes a completed radar corridor from the next leg', () => {
  const config = { ...baseConfig, meanHeading: 0 };
  const grid = fakeGrid(config);
  for (const row of grid.cells) {
    for (const cell of row) {
      cell.pBayesianEvolved = 0.01;
      cell.scanMemory = 0;
    }
  }

  const first = buildBayesianCoverageLeg(config, grid, { x: 0, y: 0 });
  const second = buildBayesianCoverageLeg(config, grid, { x: first.end.x, y: first.end.y }, [first.center]);
  const separationAlongRoute = Math.abs(second.center.y - first.center.y);

  assert.ok(separationAlongRoute >= first.trackSpacing * 0.95);
});

test('hybrid branch commitment holds a meaningful waypoint instead of changing every minute', () => {
  const commitment = new BranchCommitment({ minX: -30, maxX: 30, minY: -20, maxY: 20 }, 12);
  const first = commitment.resolve({ x: 0, y: 0 }, { x: 2, y: 0 });
  const changedProposal = commitment.resolve({ x: 1, y: 0 }, { x: 1, y: 10 });

  assert.ok(Math.hypot(first.x, first.y) >= 12 - 1e-9);
  assert.deepEqual(changedProposal, first);

  const replacement = commitment.resolve(first, { x: first.x, y: 10 });
  assert.notDeepEqual(replacement, first);
});
