import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeFuelMargin,
  scaleBeliefForObservation,
  buildHybridVector,
  applyRelativeWaypoint,
  advanceTowardWaypoint,
  solveGroundTrack,
} from '../src/engine/missionContract.ts';
import { SeededRandom } from '../src/engine/random.ts';

test('fuel margin includes return time and reserve', () => {
  const result = computeFuelMargin(100, 10, 140, 15);
  assert.ok(Math.abs(result.returnTimeMinutes - 10 / 140 * 60) < 1e-9);
  assert.ok(Math.abs(result.marginMinutes - (100 - 10 / 140 * 60 - 15)) < 1e-9);
});

test('belief observation is peak-scaled while internal mass is untouched', () => {
  const belief = [0.1, 0.2, 0.7];
  const scaled = scaleBeliefForObservation(belief);
  assert.ok(Math.abs(scaled[0] - 1 / 7) < 1e-12);
  assert.ok(Math.abs(scaled[1] - 2 / 7) < 1e-12);
  assert.equal(scaled[2], 1);
  assert.deepEqual(belief, [0.1, 0.2, 0.7]);
});

test('hybrid vector distinguishes opposite frigates', () => {
  const common = {
    helicoX: 0, helicoY: 0, headingDeg: 0, speed: 140, maxSpeed: 140,
    fuelRemaining: 100, maxFuel: 180, halfWidth: 50, halfHeight: 50,
    peakX: 5, peakY: 10, entropy: 0.5, elapsedMinutes: 20, bingoBuffer: 15,
  };
  const right = buildHybridVector({ ...common, frigateX: 30, frigateY: 0 });
  const left = buildHybridVector({ ...common, frigateX: -30, frigateY: 0 });
  assert.notDeepEqual(right, left);
  assert.equal(right.length, 10);
});

test('seeded random reproduces uniform and gaussian sequences', () => {
  const a = new SeededRandom(2026);
  const b = new SeededRandom(2026);
  const sequenceA = [a.uniform(), a.gaussian(), a.uniform(), a.gaussian()];
  const sequenceB = [b.uniform(), b.gaussian(), b.uniform(), b.gaussian()];
  assert.deepEqual(sequenceA, sequenceB);
  assert.notDeepEqual(sequenceA, [new SeededRandom(2027).uniform()]);
});

test('geofence clips a waypoint to the search bounds', () => {
  const next = applyRelativeWaypoint({ x: 49.9, y: 0, heading: 0, fuelRemaining: 100 }, [1, 0], 50, 50, 140, 1);
  assert.ok(next.x <= 50);
});

test('relative waypoint action east flies east at max speed', () => {
  const next = applyRelativeWaypoint({ x: 0, y: 0, heading: 0, fuelRemaining: 100 }, [1, 0], 50, 50, 140, 1);
  assert.ok(next.x > 0);
  assert.ok(Math.abs(next.y) < 1e-9);
  assert.equal(next.speed, 140);
  assert.equal(next.heading, 90);
});

test('headwind reduces ground speed while airspeed stays constant', () => {
  const solution = solveGroundTrack(120, 90, 20, 90);
  assert.ok(Math.abs(solution.groundSpeed - 100) < 1e-9);

  const next = advanceTowardWaypoint(
    { x: 0, y: 0, heading: 90, fuelRemaining: 100 },
    { x: 100, y: 0 },
    120,
    1,
    20,
    90,
  );
  assert.ok(Math.abs(next.x - 100 / 60) < 1e-9);
  assert.ok(Math.abs(next.y) < 1e-9);
  assert.equal(next.speed, 120);
});

test('tailwind increases ground speed', () => {
  const solution = solveGroundTrack(120, 90, 20, 270);
  assert.ok(Math.abs(solution.groundSpeed - 140) < 1e-9);
});

test('crosswind uses a crab angle while maintaining the intended ground track', () => {
  const solution = solveGroundTrack(120, 90, 20, 0);
  assert.ok(solution.airHeading < 90);
  assert.ok(solution.groundSpeed < 120);

  const next = advanceTowardWaypoint(
    { x: 0, y: 0, heading: 90, fuelRemaining: 100 },
    { x: 100, y: 0 },
    120,
    1,
    20,
    0,
  );
  assert.ok(next.x > 0);
  assert.ok(Math.abs(next.y) < 1e-9);
  assert.equal(next.speed, 120);
  assert.equal(next.heading, solution.airHeading);
});

test('an impossible crosswind is reported and does not fabricate progress on the requested track', () => {
  const solution = solveGroundTrack(20, 90, 30, 0);
  assert.equal(solution.trackMaintained, false);
  assert.equal(solution.groundSpeed, 0);

  const next = advanceTowardWaypoint(
    { x: 0, y: 0, heading: 90, fuelRemaining: 100 },
    { x: 100, y: 0 },
    20,
    1,
    30,
    0,
  );
  assert.equal(next.x, 0);
  assert.equal(next.y, 0);
  assert.equal(next.speed, 20);
});
