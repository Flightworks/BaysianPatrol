import type { ScenarioConfig } from '../types/simulation.ts';

export interface SweepPoint {
  x: number;
  y: number;
}

export interface ParallelSweepPlan {
  trackSpacing: number;
  sweepWidth: number;
  orientation: 'EAST_WEST' | 'NORTH_SOUTH';
  waypoints: SweepPoint[];
}

function effectiveRadarRange(config: ScenarioConfig): number {
  const seaClutterPenalty = Math.min(0.35, (config.windSpeed / 50) * 0.3);
  return Math.max(0.5, config.radarBaseRange * (1 - seaClutterPenalty));
}

function buildTrackPositions(start: number, end: number, spacing: number): number[] {
  if (start === end) return [start];
  const direction = Math.sign(end - start);
  const positions = [start];
  let current = start;

  while (Math.abs(end - current) > spacing) {
    current += direction * spacing;
    positions.push(current);
  }

  if (Math.abs(end - positions[positions.length - 1]) > 1e-9) positions.push(end);
  return positions;
}

/**
 * Builds an IAMSAR-style parallel sweep (PS) inside the declared search rectangle.
 * The commence-search point is half a track spacing inside the nearest corner.
 * Long legs follow the major axis and adjacent tracks remain no farther apart than S.
 */
export function buildParallelSweepPlan(config: ScenarioConfig): ParallelSweepPlan {
  const sweepWidth = 2 * effectiveRadarRange(config);
  const nominalSpacing = 0.75 * sweepWidth;
  const orientation = config.searchAreaWidth >= config.searchAreaHeight ? 'EAST_WEST' : 'NORTH_SOUTH';
  const crossDimension = orientation === 'EAST_WEST' ? config.searchAreaHeight : config.searchAreaWidth;
  const alongDimension = orientation === 'EAST_WEST' ? config.searchAreaWidth : config.searchAreaHeight;
  const trackSpacing = Math.min(nominalSpacing, crossDimension / 2, alongDimension / 2);
  const inset = trackSpacing / 2;

  const minX = config.searchAreaCenterX - config.searchAreaWidth / 2;
  const maxX = config.searchAreaCenterX + config.searchAreaWidth / 2;
  const minY = config.searchAreaCenterY - config.searchAreaHeight / 2;
  const maxY = config.searchAreaCenterY + config.searchAreaHeight / 2;

  const nearMinX = Math.abs(config.frigateX - minX) <= Math.abs(config.frigateX - maxX);
  const nearMinY = Math.abs(config.frigateY - minY) <= Math.abs(config.frigateY - maxY);
  const waypoints: SweepPoint[] = [];

  if (orientation === 'EAST_WEST') {
    const xStart = nearMinX ? minX + inset : maxX - inset;
    const xEnd = nearMinX ? maxX - inset : minX + inset;
    const yStart = nearMinY ? minY + inset : maxY - inset;
    const yEnd = nearMinY ? maxY - inset : minY + inset;
    const tracks = buildTrackPositions(yStart, yEnd, trackSpacing);
    tracks.forEach((y, index) => {
      waypoints.push(
        { x: index % 2 === 0 ? xStart : xEnd, y },
        { x: index % 2 === 0 ? xEnd : xStart, y },
      );
    });
  } else {
    const yStart = nearMinY ? minY + inset : maxY - inset;
    const yEnd = nearMinY ? maxY - inset : minY + inset;
    const xStart = nearMinX ? minX + inset : maxX - inset;
    const xEnd = nearMinX ? maxX - inset : minX + inset;
    const tracks = buildTrackPositions(xStart, xEnd, trackSpacing);
    tracks.forEach((x, index) => {
      waypoints.push(
        { x, y: index % 2 === 0 ? yStart : yEnd },
        { x, y: index % 2 === 0 ? yEnd : yStart },
      );
    });
  }

  return { trackSpacing, sweepWidth, orientation, waypoints };
}
