import assert from 'node:assert/strict';
import test from 'node:test';

import {
  radarDetectionUniform,
  shouldTerminateBeforeDetection,
} from '../src/engine/simulationIntegrity.ts';


test('radar noise is indexed by realization seed and time step', () => {
  const seed = 5_371_572;
  const stepTwoWithoutStepOne = radarDetectionUniform(seed, 2);
  radarDetectionUniform(seed, 1);
  const stepTwoAfterStepOne = radarDetectionUniform(seed, 2);

  assert.equal(stepTwoAfterStepOne, stepTwoWithoutStepOne);
  assert.notEqual(radarDetectionUniform(seed, 1), stepTwoWithoutStepOne);
  assert.equal(radarDetectionUniform(seed, 2), radarDetectionUniform(seed, 2));
});

test('safe return is terminal before any radar detection attempt', () => {
  assert.equal(shouldTerminateBeforeDetection('SAFE_RTB', false), true);
  assert.equal(shouldTerminateBeforeDetection('OUT_OF_FUEL', false), true);
  assert.equal(shouldTerminateBeforeDetection('SEARCHING', true), true);
  assert.equal(shouldTerminateBeforeDetection('SEARCHING', false), false);
});
