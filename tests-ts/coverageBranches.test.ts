import test from 'node:test';
import assert from 'node:assert/strict';
import { BranchCommitment } from '../src/engine/coverageBranch.ts';

test('hybrid branch commitment holds a meaningful waypoint instead of changing every minute', () => {
  const commitment = new BranchCommitment({ minX: -30, maxX: 30, minY: -20, maxY: 20 }, 12);
  const first = commitment.resolve({ x: 0, y: 0 }, { x: 2, y: 0 });
  const changedProposal = commitment.resolve({ x: 1, y: 0 }, { x: 1, y: 10 });

  assert.ok(Math.hypot(first.x, first.y) >= 12 - 1e-9);
  assert.deepEqual(changedProposal, first);

  const replacement = commitment.resolve(first, { x: first.x, y: 10 });
  assert.notDeepEqual(replacement, first);
});
