import test from 'node:test';
import assert from 'node:assert/strict';
import {
  aggregateBeliefGrid,
  chooseBeliefAction,
  chooseBeliefWaypoint,
  combineIndependentDetectionLikelihoods,
  evaluateBeliefOutcome,
  type BeliefAction,
} from '../src/engine/beliefSpacePlanner.ts';

const west: BeliefAction = {
  id: 'west',
  detectionLikelihoods: [0.9, 0.05],
  travelCost: 0.1,
};

const east: BeliefAction = {
  id: 'east',
  detectionLikelihoods: [0.05, 0.9],
  travelCost: 0.1,
};

test('a non-detection moves posterior mass away from the searched lobe', () => {
  const outcome = evaluateBeliefOutcome([0.6, 0.4], west.detectionLikelihoods);
  assert.ok(Math.abs(outcome.detectionProbability - 0.56) < 1e-12);
  assert.ok(outcome.noDetectionPosterior[0] < 0.15);
  assert.ok(outcome.noDetectionPosterior[1] > 0.85);
});

test('independent scans combine into cumulative detection probability', () => {
  assert.ok(Math.abs(combineIndependentDetectionLikelihoods([0.5, 0.5]) - 0.75) < 1e-12);
  assert.ok(Math.abs(combineIndependentDetectionLikelihoods([0.2, 0.3, 0.4]) - 0.664) < 1e-12);
});

test('certain detection keeps a valid posterior representation for the unreachable branch', () => {
  const outcome = evaluateBeliefOutcome([0.7, 0.3], [1, 1]);
  assert.equal(outcome.noDetectionProbability, 0);
  assert.ok(Math.abs(outcome.noDetectionPosterior.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test('zero-mass belief is rejected instead of producing an invalid distribution', () => {
  assert.throws(() => evaluateBeliefOutcome([0, 0], [0.5, 0.5]), /positive mass/);
});

test('an action that discriminates hypotheses has more expected information value', () => {
  const discriminating = evaluateBeliefOutcome([0.5, 0.5], [0.9, 0.1]);
  const nonDiscriminating = evaluateBeliefOutcome([0.5, 0.5], [0.5, 0.5]);
  assert.ok(discriminating.expectedInformationGain > nonDiscriminating.expectedInformationGain);
});

test('belief-space policy exploits a likely lobe then switches after a non-detection', () => {
  const initialBelief = [0.6, 0.4];
  const firstChoice = chooseBeliefAction(initialBelief, [west, east]);
  assert.equal(firstChoice.action.id, 'west');
  const secondChoice = chooseBeliefAction(firstChoice.outcome.noDetectionPosterior, [west, east]);
  assert.equal(secondChoice.action.id, 'east');
});

test('belief-space policy exploits a concentrated high-probability lobe', () => {
  const choice = chooseBeliefAction([0.9, 0.1], [west, east]);
  assert.equal(choice.action.id, 'west');
  assert.ok(choice.outcome.detectionProbability > 0.8);
});

test('equal-information actions prefer the lower travel cost', () => {
  const near = { id: 'near', detectionLikelihoods: [0.6, 0.4], travelCost: 0.1 };
  const far = { id: 'far', detectionLikelihoods: [0.6, 0.4], travelCost: 0.9 };
  assert.equal(chooseBeliefAction([0.5, 0.5], [far, near]).action.id, 'near');
});

test('belief aggregation preserves mass and caps a 120 by 120 grid at 400 points', () => {
  const grid = Array.from({ length: 120 }, (_, y) => Array.from({ length: 120 }, (_, x) => ({
    x, y, probability: 1 / (120 * 120),
  })));
  const aggregated = aggregateBeliefGrid(grid, 20);
  assert.ok(aggregated.length <= 400);
  assert.ok(Math.abs(aggregated.reduce((sum, cell) => sum + cell.probability, 0) - 1) < 1e-10);
});

test('geometric belief planner changes side after a negative observation', () => {
  const cells = [
    { x: -12, y: 0, probability: 0.6 },
    { x: 12, y: 0, probability: 0.4 },
  ];
  let detectionCalls = 0;
  const detectionAt = (
    sample: { x: number; y: number },
    cell: { x: number; y: number },
  ) => {
    detectionCalls += 1;
    return Math.hypot(sample.x - cell.x, sample.y - cell.y) <= 3 ? 0.9 : 0.02;
  };
  const options = {
    bounds: { minX: -20, maxX: 20, minY: -20, maxY: 20 },
    actionRadius: 8,
    directionalCandidates: 8,
    modeCandidates: 2,
    sampleSpacing: 2,
    maxSamplesPerAction: 3,
  };

  const first = chooseBeliefWaypoint(cells, { x: 0, y: 0 }, detectionAt, options);
  assert.ok(first.waypoint.x < -6);
  assert.ok(detectionCalls <= 2 * 10 * 3);
  const updatedCells = cells.map((cell, index) => ({
    ...cell,
    probability: first.outcome.noDetectionPosterior[index],
  }));
  detectionCalls = 0;
  const second = chooseBeliefWaypoint(updatedCells, { x: 0, y: 0 }, detectionAt, options);
  assert.ok(second.waypoint.x > 6);
  assert.ok(detectionCalls <= 2 * 10 * 3);
});

test('invalid numeric inputs are rejected fail-closed', () => {
  assert.throws(() => evaluateBeliefOutcome([Number.NaN, 1], [0.5, 0.5]), /finite/);
  assert.throws(() => evaluateBeliefOutcome([0.5, 0.5], [1.1, 0.5]), /between zero and one/);
  assert.throws(
    () => chooseBeliefAction([0.5, 0.5], [west, east], {
      detectionReward: 1,
      informationReward: -1,
      travelPenalty: 0.1,
    }),
    /non-negative/,
  );
});

test('candidate and sampling budgets remain globally capped', () => {
  const cells = [
    { x: -12, y: 0, probability: 0.6 },
    { x: 12, y: 0, probability: 0.4 },
  ];
  let detectionCalls = 0;
  chooseBeliefWaypoint(
    cells,
    { x: 0, y: 0 },
    () => {
      detectionCalls += 1;
      return 0.5;
    },
    {
      bounds: { minX: -20, maxX: 20, minY: -20, maxY: 20 },
      actionRadius: 8,
      directionalCandidates: 1000,
      modeCandidates: 1000,
      sampleSpacing: 0.25,
      maxSamplesPerAction: 1000,
    },
  );
  assert.ok(detectionCalls <= 2 * 18 * 12);
});

test('an almost impossible non-detection still has its normalized posterior', () => {
  const outcome = evaluateBeliefOutcome([0.5, 0.5], [1, 1 - 1e-14]);
  assert.equal(outcome.noDetectionPosterior[0], 0);
  assert.ok(outcome.noDetectionPosterior[1] > 0.999999);
});

test('direct waypoint planning rejects more than 400 belief cells', () => {
  const cells = Array.from({ length: 401 }, (_, index) => ({
    x: index % 20,
    y: Math.floor(index / 20),
    probability: 1 / 401,
  }));
  assert.throws(() => chooseBeliefWaypoint(
    cells,
    { x: 0, y: 0 },
    () => 0.5,
    {
      bounds: { minX: -20, maxX: 20, minY: -20, maxY: 20 },
      actionRadius: 8,
      directionalCandidates: 12,
      modeCandidates: 6,
      sampleSpacing: 2,
      maxSamplesPerAction: 12,
    },
  ), /400/);
});

test('log-domain update preserves a posterior when ordinary products underflow', () => {
  const outcome = evaluateBeliefOutcome(
    [Number.MIN_VALUE, 1],
    [0.5, 1],
  );
  assert.ok(outcome.noDetectionPosterior[0] > 0.999999);
  assert.equal(outcome.noDetectionPosterior[1], 0);
});

test('derived non-finite action scores are rejected', () => {
  const action: BeliefAction = {
    id: 'overflow',
    detectionLikelihoods: [0.9, 0.1],
    travelCost: 0,
  };
  assert.throws(() => chooseBeliefAction(
    [0.5, 0.5],
    [action],
    {
      detectionReward: Number.MAX_VALUE,
      informationReward: Number.MAX_VALUE,
      travelPenalty: 0,
    },
  ), /finite/);
});

test('aggregation rejects a non-positive point budget', () => {
  assert.throws(() => aggregateBeliefGrid([
    [{ x: 0, y: 0, pPresenceEvolved: 1 }],
  ], 0), /positive/);
});

test('aggregation rejects a weighted coordinate overflow', () => {
  const cells = [
    { x: Number.MAX_VALUE, y: 0, probability: 1 },
    { x: Number.MAX_VALUE, y: 0, probability: 1 },
    ...Array.from({ length: 19 }, () => ({ x: 0, y: 0, probability: 0 })),
  ];
  assert.throws(() => aggregateBeliefGrid([cells]), /finite/);
});

test('aggregation rejects more than 100000 raw cells', () => {
  const point = { x: 0, y: 0, probability: 1 / 100001 };
  const row = new Array(100001).fill(point);
  assert.throws(() => aggregateBeliefGrid([row]), /raw cells/);
});

test('normalization rejects a finite-mass sum that overflows', () => {
  assert.throws(() => evaluateBeliefOutcome(
    [Number.MAX_VALUE, Number.MAX_VALUE],
    [0.5, 0.5],
  ), /finite/);
});

test('an infinitesimal detection probability is not lost by subtraction', () => {
  const outcome = evaluateBeliefOutcome([1], [Number.MIN_VALUE]);
  assert.equal(outcome.detectionProbability, Number.MIN_VALUE);
});

test('independent detection combination preserves infinitesimal likelihoods', () => {
  assert.equal(
    combineIndependentDetectionLikelihoods([Number.MIN_VALUE]),
    Number.MIN_VALUE,
  );
});

test('fractional planner counts are rejected instead of truncated', () => {
  assert.throws(() => chooseBeliefWaypoint(
    [{ x: 0, y: 0, probability: 1 }],
    { x: 0, y: 0 },
    () => 0.5,
    {
      bounds: { minX: -10, maxX: 10, minY: -10, maxY: 10 },
      actionRadius: 4,
      directionalCandidates: 1.5,
      modeCandidates: 1,
      sampleSpacing: 1,
      maxSamplesPerAction: 12,
    },
  ), /integer/);
});

test('aggregation rejects more than 100000 rows even when they are empty', () => {
  const rows = new Array(100001).fill([]);
  assert.throws(() => aggregateBeliefGrid(rows), /rows/);
});

test('fractional aggregation budgets are rejected instead of truncated', () => {
  assert.throws(() => aggregateBeliefGrid([
    [{ x: 0, y: 0, probability: 1 }],
  ], 1.5), /integer/);
});

test('belief outcome evaluation rejects more than 400 states', () => {
  const belief = new Array(401).fill(1 / 401);
  const likelihoods = new Array(401).fill(0.5);
  assert.throws(() => evaluateBeliefOutcome(belief, likelihoods), /400/);
});
