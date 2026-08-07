import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applySharedTargetReplay,
  getPlaybackEndTime,
  interpolatePathAtTime,
} from '../src/engine/playback.ts';

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

test('target replay continues until the last strategy finishes', () => {
  const runs = [
    { helicoPath: [{ t: 0 }, { t: 6 }], targetPath: [{ t: 0, x: 0, y: 0 }] },
    { helicoPath: [{ t: 0 }, { t: 12 }], targetPath: [{ t: 0, x: 0, y: 0 }] },
    { helicoPath: [{ t: 0 }, { t: 9 }], targetPath: [{ t: 0, x: 0, y: 0 }] },
  ];
  const canonicalTarget = Array.from({ length: 13 }, (_, t) => ({ t, x: t, y: t / 2 }));

  applySharedTargetReplay(runs, canonicalTarget);

  assert.ok(runs.every(run => run.targetPath.at(-1)?.t === 12));
  assert.ok(runs.every(run => run.targetPath.length === 13));
});
