import test from 'node:test';
import assert from 'node:assert/strict';
import { getPlaybackEndTime, interpolatePathAtTime } from '../src/engine/playback.ts';

const path = [
  { t: 0, x: 0, y: 0 },
  { t: 2, x: 10, y: 4 },
  { t: 5, x: 16, y: 10 },
];

test('replay interpolates a moving target between recorded positions', () => {
  assert.deepEqual(interpolatePathAtTime(path, 1), { t: 1, x: 5, y: 2 });
  assert.deepEqual(interpolatePathAtTime(path, 3), { t: 3, x: 12, y: 6 });
});

test('replay duration ends with the actual run instead of an artificial 120-minute tail', () => {
  assert.equal(getPlaybackEndTime(path, [{ t: 0 }, { t: 4 }]), 5);
});
