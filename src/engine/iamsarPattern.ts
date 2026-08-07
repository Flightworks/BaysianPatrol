import type { ScenarioConfig } from '../types/simulation.ts';

export interface SweepPoint {
  x: number;
  y: number;
}

export interface ParallelSweepPlan {
  trackSpacing: number;
  sweepWidth: number;
  orientation: 'ROUTE_PERPENDICULAR';
  legHeadingDeg: number;
  estimatedCenter: SweepPoint;
  waypoints: SweepPoint[];
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

function effectiveRadarRange(config: ScenarioConfig): number {
  const seaClutterPenalty = Math.min(0.35, (config.windSpeed / 50) * 0.3);
  return Math.max(0.5, config.radarBaseRange * (1 - seaClutterPenalty));
}

function estimatedPositionAtSearchStart(config: ScenarioConfig, inset: number): SweepPoint {
  const transitHours = Math.hypot(config.datumX - config.frigateX, config.datumY - config.frigateY)
    / Math.max(1, config.helicoMaxSpeed);
  const routeRad = config.meanHeading * Math.PI / 180;
  const currentSpeed = config.windSpeed * 0.025;
  const currentRad = ((config.windDirection + 195) % 360) * Math.PI / 180;
  const x = config.datumX
    + (config.meanSpeed * Math.sin(routeRad) + currentSpeed * Math.sin(currentRad)) * transitHours;
  const y = config.datumY
    + (config.meanSpeed * Math.cos(routeRad) + currentSpeed * Math.cos(currentRad)) * transitHours;
  const halfWidth = config.searchAreaWidth / 2;
  const halfHeight = config.searchAreaHeight / 2;
  return {
    x: clamp(x, config.searchAreaCenterX - halfWidth + inset, config.searchAreaCenterX + halfWidth - inset),
    y: clamp(y, config.searchAreaCenterY - halfHeight + inset, config.searchAreaCenterY + halfHeight - inset),
  };
}

function clipLineToRectangle(
  origin: SweepPoint,
  direction: SweepPoint,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
): [SweepPoint, SweepPoint] | null {
  let tMin = Number.NEGATIVE_INFINITY;
  let tMax = Number.POSITIVE_INFINITY;
  for (const [position, delta, low, high] of [
    [origin.x, direction.x, minX, maxX],
    [origin.y, direction.y, minY, maxY],
  ] as const) {
    if (Math.abs(delta) < 1e-12) {
      if (position < low || position > high) return null;
      continue;
    }
    const first = (low - position) / delta;
    const second = (high - position) / delta;
    tMin = Math.max(tMin, Math.min(first, second));
    tMax = Math.min(tMax, Math.max(first, second));
  }
  if (tMin > tMax) return null;
  return [
    { x: origin.x + direction.x * tMin, y: origin.y + direction.y * tMin },
    { x: origin.x + direction.x * tMax, y: origin.y + direction.y * tMax },
  ];
}

/**
 * Parallel sweep driven by the estimated mobile route.
 * Long legs cross the route at 90 degrees; adjacent legs progress down-route.
 * The track at offset zero crosses the datum propagated during helicopter transit.
 */
export function buildParallelSweepPlan(config: ScenarioConfig): ParallelSweepPlan {
  const radarRange = effectiveRadarRange(config);
  const sweepWidth = 2 * radarRange;
  const nominalSpacing = 0.75 * sweepWidth;
  const trackSpacing = Math.min(nominalSpacing, config.searchAreaWidth / 3, config.searchAreaHeight / 3);
  const inset = Math.min(trackSpacing / 2, config.searchAreaWidth / 4, config.searchAreaHeight / 4);
  const estimatedCenter = estimatedPositionAtSearchStart(config, inset);

  const routeRad = config.meanHeading * Math.PI / 180;
  const route = { x: Math.sin(routeRad), y: Math.cos(routeRad) };
  const leg = { x: Math.cos(routeRad), y: -Math.sin(routeRad) };
  const legHeadingDeg = (config.meanHeading + 90) % 360;

  const minX = config.searchAreaCenterX - config.searchAreaWidth / 2 + inset;
  const maxX = config.searchAreaCenterX + config.searchAreaWidth / 2 - inset;
  const minY = config.searchAreaCenterY - config.searchAreaHeight / 2 + inset;
  const maxY = config.searchAreaCenterY + config.searchAreaHeight / 2 - inset;
  const corners = [
    { x: minX, y: minY }, { x: minX, y: maxY },
    { x: maxX, y: minY }, { x: maxX, y: maxY },
  ];
  const projections = corners.map(point =>
    (point.x - estimatedCenter.x) * route.x + (point.y - estimatedCenter.y) * route.y);
  const rectangleMin = Math.min(...projections);
  const rectangleMax = Math.max(...projections);
  const usefulBacktrack = 2.2 * Math.max(config.sigmaDatumX, config.sigmaDatumY) + radarRange;
  const minOffset = Math.max(rectangleMin, -usefulBacktrack);
  const maxOffset = rectangleMax;

  const offsets = [0];
  for (let offset = -trackSpacing; offset >= minOffset; offset -= trackSpacing) offsets.push(offset);
  for (let offset = trackSpacing; offset <= maxOffset; offset += trackSpacing) offsets.push(offset);
  offsets.sort((a, b) => a - b);

  const lines = offsets
    .map(offset => {
      const origin = {
        x: estimatedCenter.x + route.x * offset,
        y: estimatedCenter.y + route.y * offset,
      };
      return clipLineToRectangle(origin, leg, minX, maxX, minY, maxY);
    })
    .filter((line): line is [SweepPoint, SweepPoint] => !!line && Math.hypot(
      line[1].x - line[0].x,
      line[1].y - line[0].y,
    ) > radarRange);

  const waypoints: SweepPoint[] = [];
  let reverse = false;
  if (lines[0]) {
    const [a, b] = lines[0];
    reverse = Math.hypot(config.frigateX - b.x, config.frigateY - b.y)
      < Math.hypot(config.frigateX - a.x, config.frigateY - a.y);
  }
  lines.forEach((line, index) => {
    const swap = index % 2 === 0 ? reverse : !reverse;
    waypoints.push(...(swap ? [line[1], line[0]] : [line[0], line[1]]));
  });

  return {
    trackSpacing,
    sweepWidth,
    orientation: 'ROUTE_PERPENDICULAR',
    legHeadingDeg,
    estimatedCenter,
    waypoints,
  };
}
