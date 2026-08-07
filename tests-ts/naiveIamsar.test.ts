import test from 'node:test';
import assert from 'node:assert/strict';
import { buildParallelSweepPlan } from '../src/engine/iamsarPattern.ts';
import type { ScenarioConfig } from '../src/types/simulation.ts';

const config: ScenarioConfig = {
  searchAreaWidth: 60,
  searchAreaHeight: 40,
  searchAreaCenterX: 0,
  searchAreaCenterY: 0,
  datumX: -15,
  datumY: -15,
  sigmaDatumX: 3,
  sigmaDatumY: 3,
  meanHeading: 45,
  meanSpeed: 16,
  sigmaT: 10,
  sigmaHeading: 15,
  sigmaSpeed: 4,
  windDirection: 270,
  windSpeed: 15,
  sigmaWindSpeed: 4,
  sigmaWindDirection: 15,
  frigateX: 25,
  frigateY: -18,
  sigmaFrigatePosition: 0,
  helicoMaxSpeed: 130,
  sigmaHelicoSpeed: 0,
  helicoEndurance: 180,
  bingoFuelBuffer: 20,
  sigmaRouteDrift: 3,
  sigmaSpeedDrift: 1,
  radarBaseRange: 12,
  gridCellSize: 0.5,
  dt: 1,
  numIterations: 250,
  strategy: 'TRIO',
};

test('IAMSAR parallel sweep remains half a track spacing inside the search area', () => {
  const plan = buildParallelSweepPlan(config);
  const minX = -config.searchAreaWidth / 2;
  const maxX = config.searchAreaWidth / 2;
  const minY = -config.searchAreaHeight / 2;
  const maxY = config.searchAreaHeight / 2;

  assert.ok(plan.trackSpacing > 0);
  assert.ok(plan.waypoints.length >= 4);
  for (const point of plan.waypoints) {
    assert.ok(point.x > minX && point.x < maxX);
    assert.ok(point.y > minY && point.y < maxY);
  }
  assert.ok(Math.abs(plan.waypoints[0].x - (maxX - plan.trackSpacing / 2)) < 1e-9);
  assert.ok(Math.abs(plan.waypoints[0].y - (minY + plan.trackSpacing / 2)) < 1e-9);
});

test('IAMSAR parallel sweep alternates long parallel legs with constant short shifts', () => {
  const plan = buildParallelSweepPlan(config);
  const points = plan.waypoints;
  const longLegs: number[] = [];
  const shifts: number[] = [];

  for (let index = 1; index < points.length; index += 1) {
    const dx = Math.abs(points[index].x - points[index - 1].x);
    const dy = Math.abs(points[index].y - points[index - 1].y);
    if (dx > dy) longLegs.push(dx);
    else if (dy > 0) shifts.push(dy);
  }

  assert.ok(longLegs.length >= 2);
  assert.ok(longLegs.every((length) => Math.abs(length - longLegs[0]) < 1e-9));
  assert.ok(shifts.every((length) => length <= plan.trackSpacing + 1e-9));
  assert.equal(Math.sign(points[1].x - points[0].x), -1);
  assert.equal(Math.sign(points[3].x - points[2].x), 1);
});
