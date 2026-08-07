import test from 'node:test';
import assert from 'node:assert/strict';
import { SharedAsyncResource } from '../src/engine/sharedAsyncResource.ts';

test('shared async resource initializes a heavy session only once across a campaign', async () => {
  const resource = new SharedAsyncResource<{ id: number }>();
  let loads = 0;
  const loader = async () => ({ id: ++loads });

  const sessions = await Promise.all(Array.from({ length: 250 }, () => resource.get(loader)));

  assert.equal(loads, 1);
  assert.equal(sessions.length, 250);
  assert.ok(sessions.every((session) => session === sessions[0]));
});

test('shared async resource also caches an unavailable optional resource', async () => {
  const resource = new SharedAsyncResource<null>();
  let loads = 0;

  const first = await resource.get(async () => { loads += 1; return null; });
  const second = await resource.get(async () => { loads += 1; return null; });

  assert.equal(first, null);
  assert.equal(second, null);
  assert.equal(loads, 1);
});
