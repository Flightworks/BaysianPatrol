import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeFuelMargin,
  scaleBeliefForObservation,
  buildHybridVector,
  applyRelativeWaypoint,
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
