import test from 'node:test';
import assert from 'node:assert/strict';
import { PRESETS } from '../src/engine/presets.ts';

const standard = PRESETS.find(preset => preset.id === 'standard');
if (!standard) throw new Error('Standard preset is required');

test('standard fast-boat scenario defaults radar detection range to 4 NM', () => {
  assert.equal(standard.config.radarBaseRange, 4);
});

test('standard fast-boat scenario defaults spatial datum uncertainty to 20 NM', () => {
  assert.equal(standard.config.sigmaDatumX, 20);
  assert.equal(standard.config.sigmaDatumY, 20);
});

test('standard fast-boat scenario defaults initial time uncertainty to 60 minutes', () => {
  assert.equal(standard.config.sigmaT, 60);
});
