import test from 'node:test';
import assert from 'node:assert/strict';
import { buildParallelSweepPlan } from '../src/engine/iamsarPattern.ts';
import type { ScenarioConfig } from '../src/types/simulation.ts';

export const config: ScenarioConfig = {
  searchAreaWidth: 60,
  searchAreaHeight: 40,
  searchAreaCenterX: 0,
  searchAreaCenterY: 0,
  datumX: 10,
  datumY: 0,
  sigmaDatumX: 3,
  sigmaDatumY: 3,
  meanHeading: 0,
  meanSpeed: 30,
  sigmaT: 10,
  sigmaHeading: 15,
  sigmaSpeed: 4,
  windDirection: 0,
  windSpeed: 0,
  sigmaWindSpeed: 0,
  sigmaWindDirection: 0,
  frigateX: 0,
  frigateY: -20,
  sigmaFrigatePosition: 0,
  helicoMaxSpeed: 120,
  sigmaHelicoSpeed: 0,
  helicoEndurance: 180,
  bingoFuelBuffer: 20,
  sigmaRouteDrift: 3,
  sigmaSpeedDrift: 1,
  radarBaseRange: 12,
  gridCellSize: 1,
  dt: 1,
  numIterations: 250,
  strategy: 'TRIO',
};

const headingOf = (a: { x: number; y: number }, b: { x: number; y: number }) =>
  (Math.atan2(b.x - a.x, b.y - a.y) * 180 / Math.PI + 360) % 360;

const axialDifference = (a: number, b: number) => {
  const delta = Math.abs(a - b) % 180;
  return Math.min(delta, 180 - delta);
};

test('IAMSAR sweep is anchored on the datum propagated during helicopter transit', () => {
  const plan = buildParallelSweepPlan(config);
  const transitHours = Math.hypot(config.datumX - config.frigateX, config.datumY - config.frigateY)
    / config.helicoMaxSpeed;

  assert.ok(Math.abs(plan.estimatedCenter.x - config.datumX) < 1e-9);
  assert.ok(Math.abs(plan.estimatedCenter.y - (config.datumY + config.meanSpeed * transitHours)) < 1e-9);

  const centerFallsOnALeg = plan.waypoints.some((point, index) => {
    if (index % 2 !== 0) return false;
    const next = plan.waypoints[index + 1];
    if (!next) return false;
    const legLength = Math.hypot(next.x - point.x, next.y - point.y);
    const splitLength = Math.hypot(plan.estimatedCenter.x - point.x, plan.estimatedCenter.y - point.y)
      + Math.hypot(next.x - plan.estimatedCenter.x, next.y - plan.estimatedCenter.y);
    return Math.abs(splitLength - legLength) < 1e-6;
  });
  assert.equal(centerFallsOnALeg, true);
});

test('IAMSAR long legs are perpendicular to estimated mobile route and remain in Airplan', () => {
  const routedConfig = { ...config, meanHeading: 35 };
  const plan = buildParallelSweepPlan(routedConfig);
  const expectedLegHeading = (routedConfig.meanHeading + 90) % 360;
  const minX = routedConfig.searchAreaCenterX - routedConfig.searchAreaWidth / 2;
  const maxX = routedConfig.searchAreaCenterX + routedConfig.searchAreaWidth / 2;
  const minY = routedConfig.searchAreaCenterY - routedConfig.searchAreaHeight / 2;
  const maxY = routedConfig.searchAreaCenterY + routedConfig.searchAreaHeight / 2;

  assert.ok(axialDifference(plan.legHeadingDeg, expectedLegHeading) < 1e-9);
  assert.ok(plan.waypoints.length >= 4);

  for (let index = 0; index < plan.waypoints.length; index += 2) {
    const start = plan.waypoints[index];
    const end = plan.waypoints[index + 1];
    assert.ok(end);
    assert.ok(axialDifference(headingOf(start, end), expectedLegHeading) < 1e-6);
  }

  for (const point of plan.waypoints) {
    assert.ok(point.x >= minX && point.x <= maxX);
    assert.ok(point.y >= minY && point.y <= maxY);
  }
});

test('IAMSAR sweep is deterministic for an identical mission estimate', () => {
  assert.deepEqual(buildParallelSweepPlan(config), buildParallelSweepPlan(config));
});
