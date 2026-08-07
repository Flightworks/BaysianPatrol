export interface TimedPoint {
  t: number;
  x: number;
  y: number;
}

export function interpolatePathAtTime<T extends TimedPoint>(path: readonly T[], time: number): TimedPoint | null {
  if (path.length === 0) return null;
  if (time <= path[0].t) return { t: time, x: path[0].x, y: path[0].y };
  const last = path[path.length - 1];
  if (time >= last.t) return { t: time, x: last.x, y: last.y };

  const nextIndex = path.findIndex(point => point.t >= time);
  const next = path[nextIndex];
  const previous = path[nextIndex - 1];
  if (!previous || next.t === previous.t) return { t: time, x: next.x, y: next.y };
  const ratio = (time - previous.t) / (next.t - previous.t);
  return {
    t: time,
    x: previous.x + (next.x - previous.x) * ratio,
    y: previous.y + (next.y - previous.y) * ratio,
  };
}

export function getPlaybackEndTime(...paths: ReadonlyArray<readonly { t: number }[]>): number {
  let maximum = 0;
  for (const path of paths) {
    for (const point of path) maximum = Math.max(maximum, point.t);
  }
  return maximum;
}

export function applySharedTargetReplay<T extends {
  helicoPath: readonly { t: number }[];
  targetPath: TimedPoint[];
}>(runs: T[], canonicalTargetPath: readonly TimedPoint[]): number {
  const replayEnd = getPlaybackEndTime(...runs.map(run => run.helicoPath));
  const sharedPath = canonicalTargetPath
    .filter(point => point.t <= replayEnd)
    .map(point => ({ ...point }));

  for (const run of runs) {
    run.targetPath = sharedPath.map(point => ({ ...point }));
  }
  return replayEnd;
}
