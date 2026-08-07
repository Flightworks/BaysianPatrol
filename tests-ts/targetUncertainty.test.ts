import test from 'node:test';
import assert from 'node:assert/strict';
import {
  deriveInitialTargetTruth,
  advanceTargetTruth,
} from '../src/engine/targetUncertainty.ts';
import { buildCanonicalTargetPath, TargetSim } from '../src/engine/targetGenerator.ts';
import { PRESETS } from '../src/engine/presets.ts';
import { SeededRandom } from '../src/engine/random.ts';

test('ground truth starts away from the observed datum using spatial and temporal uncertainty', () => {
  const truth = deriveInitialTargetTruth({
    datumX: 0,
    datumY: 0,
    spatialOffsetX: 2,
    spatialOffsetY: -1,
    timeOffsetMinutes: 10,
    speed: 12,
    heading: 90,
  });

  assert.ok(Math.abs(truth.x - 4) < 1e-9);
  assert.ok(Math.abs(truth.y + 1) < 1e-9);
  assert.equal(truth.speed, 12);
  assert.equal(truth.heading, 90);
});

test('ground truth moves immediately from t0', () => {
  const initial = deriveInitialTargetTruth({
    datumX: 5,
    datumY: -3,
    spatialOffsetX: 0,
    spatialOffsetY: 0,
    timeOffsetMinutes: 0,
    speed: 18,
    heading: 0,
  });
  const next = advanceTargetTruth(initial, {
    dtMinutes: 1,
    headingNoise: 0,
    speedNoise: 0,
    currentSpeed: 0,
    currentHeading: 0,
  });

  assert.ok(Math.abs(next.x - 5) < 1e-9);
  assert.ok(Math.abs(next.y - (-2.7)) < 1e-9);
  assert.ok(Math.hypot(next.x - initial.x, next.y - initial.y) > 0.29);
});

test('one canonical target path is generated for the complete paired realization', () => {
  const config = PRESETS[0].config;
  const realization = new TargetSim(config, undefined, new SeededRandom(2026)).getRealization();
  const first = buildCanonicalTargetPath(config, realization, 10);
  const second = buildCanonicalTargetPath(config, realization, 10);

  assert.deepEqual(first, second);
  assert.equal(first[0].t, 0);
  assert.equal(first.at(-1)?.t, 10);
  assert.equal(first[0].x, realization.targetInitialX);
  assert.ok(Math.hypot(first[1].x - first[0].x, first[1].y - first[0].y) > 0);

  const unevenConfig = { ...config, dt: 3, helicoEndurance: 10 };
  const unevenPath = buildCanonicalTargetPath(unevenConfig, realization);
  assert.equal(unevenPath.at(-1)?.t, 12);
});
