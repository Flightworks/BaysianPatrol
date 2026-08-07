import React, { useRef, useEffect, useState } from 'react';
import type { ScenarioConfig, MonteCarloRunResult, GridMode } from '../types/simulation';
import { getRadarFootprintContour } from '../engine/radarModel';
import { Eye, EyeOff, Layers, ZoomIn, ZoomOut, RotateCcw, Activity } from 'lucide-react';
import { interpolatePathAtTime } from '../engine/playback';

const LAYER_INFO: Record<GridMode, { label: string; description: string }> = {
  CLASSICAL: {
    label: 'Prévision de route',
    description: 'Position probable issue du datum, de la route et de la dérive, sans effet des balayages radar.',
  },
  BAYESIAN_STANDARD: {
    label: 'Posterior radar',
    description: 'Prévision corrigée par les non-détections : une zone balayée perd de la probabilité.',
  },
  BAYESIAN_EVOLVED: {
    label: 'Posterior tactique',
    description: "Posterior radar tenant compte de l'efficacité du capteur selon l'angle d'approche.",
  },
};

interface TacticalCanvasProps {
  config: ScenarioConfig;
  selectedSigmaRun?: MonteCarloRunResult | null;
  selectedNaiveRun?: MonteCarloRunResult | null;
  selectedRlRun?: MonteCarloRunResult | null;
  currentPlaybackTime: number; // minutes
  showTargetGroundTruth: boolean;
  onToggleGroundTruth: () => void;
  allRuns?: MonteCarloRunResult[];
  showAllRunsOverlay: boolean;
  onToggleAllRunsOverlay: () => void;
}

export const TacticalCanvas: React.FC<TacticalCanvasProps> = ({
  config,
  selectedSigmaRun,
  selectedNaiveRun,
  selectedRlRun,
  currentPlaybackTime,
  showTargetGroundTruth,
  onToggleGroundTruth,
  allRuns,
  showAllRunsOverlay,
  onToggleAllRunsOverlay,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [gridMode, setGridMode] = useState<GridMode>('BAYESIAN_EVOLVED');
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Handle canvas sizing
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && containerRef.current) {
        canvasRef.current.width = containerRef.current.clientWidth;
        canvasRef.current.height = Math.max(500, containerRef.current.clientHeight);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Main Dual-Trajectory Render Loop (SIGMA vs Naive)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = (canvas.getContext('2d', {
      alpha: false,
      desynchronized: true,
      willReadFrequently: false,
      powerPreference: 'high-performance',
    } as any) || canvas.getContext('2d')) as CanvasRenderingContext2D;

    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Fast GPU fill background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Coordinate conversion NM -> Canvas pixels
    const baseScale = (Math.min(width, height) / (config.searchAreaWidth * 1.5)) * zoom;
    const centerX = width / 2 + pan.x;
    const centerY = height / 2 + pan.y;

    const nmToPxX = (nmX: number) => centerX + nmX * baseScale;
    const nmToPxY = (nmY: number) => centerY - nmY * baseScale;

    // Primary run used for grid heatmaps
    const activeSnapshotRun = selectedRlRun || selectedSigmaRun || selectedNaiveRun;
    const activeEnv = activeSnapshotRun?.runEnv || config;

    // 1. Draw Coordinate Grid Lines (Every 5 NM)
    ctx.strokeStyle = 'rgba(30, 58, 138, 0.25)';
    ctx.lineWidth = 1;
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';

    const stepNM = 5;
    const minNM = -80;
    const maxNM = 80;

    for (let x = minNM; x <= maxNM; x += stepNM) {
      const px = nmToPxX(x);
      if (px >= 0 && px <= width) {
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, height);
        ctx.stroke();
        ctx.fillText(`${x > 0 ? '+' : ''}${x} NM`, px + 2, height - 6);
      }
    }

    for (let y = minNM; y <= maxNM; y += stepNM) {
      const py = nmToPxY(y);
      if (py >= 0 && py <= height) {
        ctx.beginPath();
        ctx.moveTo(0, py);
        ctx.lineTo(width, py);
        ctx.stroke();
        ctx.fillText(`${y > 0 ? '+' : ''}${y} NM`, 6, py - 4);
      }
    }

    // 2. Draw Rectangular Search Area Boundary
    const searchLeft = nmToPxX(config.searchAreaCenterX - config.searchAreaWidth / 2);
    const searchRight = nmToPxX(config.searchAreaCenterX + config.searchAreaWidth / 2);
    const searchTop = nmToPxY(config.searchAreaCenterY + config.searchAreaHeight / 2);
    const searchBottom = nmToPxY(config.searchAreaCenterY - config.searchAreaHeight / 2);

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(searchLeft, searchTop, searchRight - searchLeft, searchBottom - searchTop);
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(56, 189, 248, 0.8)';
    ctx.fillText(`ZONE DE RECHERCHE (${config.searchAreaWidth}×${config.searchAreaHeight} NM)`, searchLeft + 8, searchTop + 16);

    // 3. Draw Probability Density Heatmap (Classical, Standard Bayesian, or Evolved Bayesian)
    if (activeSnapshotRun && activeSnapshotRun.heatmapSnapshots.length > 0) {
      let snapshot = activeSnapshotRun.heatmapSnapshots[0];
      for (const s of activeSnapshotRun.heatmapSnapshots) {
        if (s.t <= currentPlaybackTime) snapshot = s;
        else break;
      }

      let activeProbs = snapshot.probsBayesianEvolved;
      if (gridMode === 'CLASSICAL') {
        activeProbs = snapshot.probsClassical;
      } else if (gridMode === 'BAYESIAN_STANDARD') {
        activeProbs = snapshot.probsBayesianStandard;
      }

      if (activeProbs && activeProbs.length > 0) {
        const gridCellSize = config.gridCellSize;
        const widthCells = Math.ceil(config.searchAreaWidth / gridCellSize);
        const heightCells = Math.ceil(config.searchAreaHeight / gridCellSize);
        const minX = config.searchAreaCenterX - config.searchAreaWidth / 2;
        const minY = config.searchAreaCenterY - config.searchAreaHeight / 2;

        let maxProb = 0.0001;
        for (const p of activeProbs) {
          if (p > maxProb) maxProb = p;
        }

        const cellPxW = Math.max(2, gridCellSize * baseScale + 1.2);
        const cellPxH = Math.max(2, gridCellSize * baseScale + 1.2);

        let idx = 0;
        for (let j = 0; j < heightCells; j++) {
          const cellY = minY + (j + 0.5) * gridCellSize;
          for (let i = 0; i < widthCells; i++) {
            const cellX = minX + (i + 0.5) * gridCellSize;
            const prob = activeProbs[idx] || 0;
            idx++;

            if (prob > 0.00001) {
              const normP = Math.min(1.0, prob / (maxProb * 0.75));
              
              let r = 0, g = 0, b = 0, a = normP * 0.65;
              if (normP < 0.33) {
                g = Math.floor(255 * (normP / 0.33));
                b = Math.floor(180 * (1 - normP / 0.33));
              } else if (normP < 0.66) {
                r = Math.floor(255 * ((normP - 0.33) / 0.33));
                g = 255;
              } else {
                r = 255;
                g = Math.floor(255 * (1 - (normP - 0.66) / 0.34));
              }

              const pxX = nmToPxX(cellX - gridCellSize / 2);
              const pxY = nmToPxY(cellY + gridCellSize / 2);

              ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
              ctx.fillRect(pxX, pxY, cellPxW, cellPxH);
            }
          }
        }
      }
    }

    // 4. Draw Monte Carlo Ghost Overlay (All N runs faint target paths)
    if (showTargetGroundTruth && showAllRunsOverlay && allRuns && allRuns.length > 0) {
      ctx.lineWidth = 0.5;
      for (const run of allRuns) {
        if (run.targetPath.length > 0) {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.12)';
          ctx.beginPath();
          ctx.moveTo(nmToPxX(run.targetPath[0].x), nmToPxY(run.targetPath[0].y));
          for (let p = 1; p < run.targetPath.length; p++) {
            ctx.lineTo(nmToPxX(run.targetPath[p].x), nmToPxY(run.targetPath[p].y));
          }
          ctx.stroke();
        }

        if (run.interceptPoint) {
          const ipX = nmToPxX(run.interceptPoint.x);
          const ipY = nmToPxY(run.interceptPoint.y);
          ctx.fillStyle = run.strategy === 'SIGMA' ? 'rgba(16, 185, 129, 0.7)' : 'rgba(245, 158, 11, 0.7)';
          ctx.beginPath();
          ctx.arc(ipX, ipY, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // 5. DRAW DUAL TRAJECTORIES COMPARISON (SIGMA Green vs Naïve Amber)

    // A. NAÏVE TRAJECTORY (Classic Creeping Line Baseline - Amber/Gold)
    if (selectedNaiveRun) {
      const hPointsNaive = selectedNaiveRun.helicoPath.filter(p => p.t <= currentPlaybackTime);
      if (hPointsNaive.length > 0) {
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = '#f59e0b';
        ctx.beginPath();
        ctx.moveTo(nmToPxX(hPointsNaive[0].x), nmToPxY(hPointsNaive[0].y));
        for (let p = 1; p < hPointsNaive.length; p++) {
          ctx.lineTo(nmToPxX(hPointsNaive[p].x), nmToPxY(hPointsNaive[p].y));
        }
        ctx.stroke();

        const currentNaive = hPointsNaive[hPointsNaive.length - 1];
        const naivePxX = nmToPxX(currentNaive.x);
        const naivePxY = nmToPxY(currentNaive.y);

        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(naivePxX, naivePxY, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#fef3c7';
        ctx.font = 'bold 10px sans-serif';
        ctx.fillText(`RÂTEAU IAMSAR (${currentNaive.status})`, naivePxX + 10, naivePxY + 20);
      }
    }

    // B. SIGMA TRAJECTORY (Optimized Bayesian Interception - Emerald Green)
    if (selectedSigmaRun) {
      const hPointsSigma = selectedSigmaRun.helicoPath.filter(p => p.t <= currentPlaybackTime);
      if (hPointsSigma.length > 0) {
        ctx.lineWidth = 3.0;
        ctx.strokeStyle = '#10b981';
        ctx.beginPath();
        ctx.moveTo(nmToPxX(hPointsSigma[0].x), nmToPxY(hPointsSigma[0].y));
        for (let p = 1; p < hPointsSigma.length; p++) {
          ctx.lineTo(nmToPxX(hPointsSigma[p].x), nmToPxY(hPointsSigma[p].y));
        }
        ctx.stroke();

        const currentSigma = hPointsSigma[hPointsSigma.length - 1];
        const sigmaPxX = nmToPxX(currentSigma.x);
        const sigmaPxY = nmToPxY(currentSigma.y);

        const contour = getRadarFootprintContour(config.meanHeading, {
          baseRange: config.radarBaseRange,
          windSpeed: activeEnv.windSpeed ?? config.windSpeed,
          windDirection: activeEnv.windDirection ?? config.windDirection,
        }, 36);

        ctx.save();
        ctx.beginPath();
        if (contour.length > 0) {
          ctx.moveTo(nmToPxX(currentSigma.x + contour[0].dx), nmToPxY(currentSigma.y + contour[0].dy));
          for (let p = 1; p < contour.length; p++) {
            ctx.lineTo(nmToPxX(currentSigma.x + contour[p].dx), nmToPxY(currentSigma.y + contour[p].dy));
          }
        }
        ctx.closePath();

        ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(sigmaPxX, sigmaPxY, 6.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#d1fae5';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(`RECHERCHE BAYÉSIENNE (${currentSigma.status})`, sigmaPxX + 10, sigmaPxY - 14);
      }
    }

    // B2. RL MODEL TRAJECTORY (Neural Network Policy - Purple / Magenta)
    if (selectedRlRun) {
      const hPointsRl = selectedRlRun.helicoPath.filter(p => p.t <= currentPlaybackTime);
      if (hPointsRl.length > 0) {
        ctx.lineWidth = 3.0;
        ctx.strokeStyle = '#a855f7';
        ctx.beginPath();
        ctx.moveTo(nmToPxX(hPointsRl[0].x), nmToPxY(hPointsRl[0].y));
        for (let p = 1; p < hPointsRl.length; p++) {
          ctx.lineTo(nmToPxX(hPointsRl[p].x), nmToPxY(hPointsRl[p].y));
        }
        ctx.stroke();

        const currentRl = hPointsRl[hPointsRl.length - 1];
        const rlPxX = nmToPxX(currentRl.x);
        const rlPxY = nmToPxY(currentRl.y);

        const contour = getRadarFootprintContour(config.meanHeading, {
          baseRange: config.radarBaseRange,
          windSpeed: activeEnv.windSpeed ?? config.windSpeed,
          windDirection: activeEnv.windDirection ?? config.windDirection,
        }, 36);

        ctx.save();
        ctx.beginPath();
        if (contour.length > 0) {
          ctx.moveTo(nmToPxX(currentRl.x + contour[0].dx), nmToPxY(currentRl.y + contour[0].dy));
          for (let p = 1; p < contour.length; p++) {
            ctx.lineTo(nmToPxX(currentRl.x + contour[p].dx), nmToPxY(currentRl.y + contour[p].dy));
          }
        }
        ctx.closePath();

        ctx.fillStyle = 'rgba(168, 85, 247, 0.18)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.8)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = '#a855f7';
        ctx.beginPath();
        ctx.arc(rlPxX, rlPxY, 6.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#f3e8ff';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(`STRATÉGIE HYBRIDE (${currentRl.status})`, rlPxX + 10, rlPxY + 4);
      }
    }

    // C. Target Ground Truth Path (if unmasked)
    if (showTargetGroundTruth && activeSnapshotRun) {
      const tPoints = activeSnapshotRun.targetPath.filter(p => p.t < currentPlaybackTime);
      const interpolatedTarget = interpolatePathAtTime(activeSnapshotRun.targetPath, currentPlaybackTime);
      if (interpolatedTarget) tPoints.push(interpolatedTarget);
      if (tPoints.length > 0) {
        ctx.lineWidth = 2.0;
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#ef4444';
        ctx.beginPath();
        ctx.moveTo(nmToPxX(tPoints[0].x), nmToPxY(tPoints[0].y));
        for (let p = 1; p < tPoints.length; p++) {
          ctx.lineTo(nmToPxX(tPoints[p].x), nmToPxY(tPoints[p].y));
        }
        ctx.stroke();
        ctx.setLineDash([]);

        const currentT = tPoints[tPoints.length - 1];
        const tPxX = nmToPxX(currentT.x);
        const tPxY = nmToPxY(currentT.y);

        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(tPxX, tPxY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();

        ctx.fillStyle = '#fca5a5';
        ctx.font = 'bold 10px sans-serif';
        ctx.fillText('CIBLE RÉELLE', tPxX + 8, tPxY + 12);
      }
    }

    // 6. Draw Frigate Base Marker (Randomized per run)
    const frigateX = activeEnv.frigateX ?? config.frigateX;
    const frigateY = activeEnv.frigateY ?? config.frigateY;
    const frigPxX = nmToPxX(frigateX);
    const frigPxY = nmToPxY(frigateY);

    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(frigPxX - 5, frigPxY - 5, 10, 10);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.strokeRect(frigPxX - 5, frigPxY - 5, 10, 10);

    ctx.fillStyle = '#7dd3fc';
    ctx.font = 'bold 10px sans-serif';
    ctx.fillText('FRÉGATE (BASE)', frigPxX + 8, frigPxY + 12);

    // 7. Draw Datum T0 Marker
    const datumX = activeEnv.datumX ?? config.datumX;
    const datumY = activeEnv.datumY ?? config.datumY;
    const datumPxX = nmToPxX(datumX);
    const datumPxY = nmToPxY(datumY);

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(datumPxX, datumPxY, 7, 0, Math.PI * 2);
    ctx.moveTo(datumPxX - 10, datumPxY);
    ctx.lineTo(datumPxX + 10, datumPxY);
    ctx.moveTo(datumPxX, datumPxY - 10);
    ctx.lineTo(datumPxX + 10, datumPxY);
    ctx.stroke();

    ctx.fillStyle = '#fcd34d';
    ctx.font = 'bold 10px sans-serif';
    ctx.fillText(`DATUM T₀ (±${config.sigmaDatumX}NM)`, datumPxX + 10, datumPxY - 8);

    // 8. Draw Compass & Wind Rose Widget (Top Right Corner)
    const compassX = width - 55;
    const compassY = 55;
    const compassRadius = 25;

    const windDir = activeEnv.windDirection ?? config.windDirection;
    const windSpd = activeEnv.windSpeed ?? config.windSpeed;

    ctx.save();
    ctx.beginPath();
    ctx.arc(compassX, compassY, compassRadius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const windRad = ((windDir + 180) * Math.PI) / 180.0;
    const arrowX = compassX + (compassRadius - 6) * Math.sin(windRad);
    const arrowY = compassY - (compassRadius - 6) * Math.cos(windRad);

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(compassX, compassY);
    ctx.lineTo(arrowX, arrowY);
    ctx.stroke();

    ctx.fillStyle = '#38bdf8';
    ctx.font = '9px monospace';
    ctx.fillText(`VENT ${windSpd.toFixed(0)}kt`, compassX - 25, compassY + compassRadius + 14);
    ctx.restore();

  }, [
    config,
    selectedSigmaRun,
    selectedNaiveRun,
    selectedRlRun,
    currentPlaybackTime,
    showTargetGroundTruth,
    showAllRunsOverlay,
    allRuns,
    gridMode,
    zoom,
    pan,
  ]);

  // Mouse pan & zoom handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div ref={containerRef} className="tactical-canvas-shell">
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      />

      {/* Floating Canvas Controls: 3-Way Grid Selector */}
      <div className="tactical-toolbar absolute top-3 left-3 flex flex-wrap items-center gap-2 p-1.5 z-10 text-xs">
        <span className="text-[11px] font-bold text-slate-400 pl-1 flex items-center space-x-1">
          <Activity className="w-3.5 h-3.5 text-cyan-400" />
          <span>Couche:</span>
        </span>

        <button
          onClick={() => setGridMode('CLASSICAL')}
          className={`px-2.5 py-1 rounded font-semibold transition-all cursor-pointer ${
            gridMode === 'CLASSICAL'
              ? 'bg-amber-600 text-slate-950 font-bold shadow-md'
              : 'bg-slate-800 text-slate-400 hover:text-slate-200'
          }`}
          title={LAYER_INFO.CLASSICAL.description}
        >
          {LAYER_INFO.CLASSICAL.label}
        </button>

        <button
          onClick={() => setGridMode('BAYESIAN_STANDARD')}
          className={`px-2.5 py-1 rounded font-semibold transition-all cursor-pointer ${
            gridMode === 'BAYESIAN_STANDARD'
              ? 'bg-cyan-600 text-white font-bold shadow-md'
              : 'bg-slate-800 text-slate-400 hover:text-slate-200'
          }`}
          title={LAYER_INFO.BAYESIAN_STANDARD.description}
        >
          {LAYER_INFO.BAYESIAN_STANDARD.label}
        </button>

        <button
          onClick={() => setGridMode('BAYESIAN_EVOLVED')}
          className={`flex items-center space-x-1 px-2.5 py-1 rounded font-semibold transition-all cursor-pointer ${
            gridMode === 'BAYESIAN_EVOLVED'
              ? 'bg-cyan-700 text-white font-bold'
              : 'bg-slate-800 text-slate-400 hover:text-slate-200'
          }`}
          title={LAYER_INFO.BAYESIAN_EVOLVED.description}
        >
          <span>{LAYER_INFO.BAYESIAN_EVOLVED.label}</span>
        </button>

        <span className="tactical-layer-description">{LAYER_INFO[gridMode].description}</span>

        <div className="h-4 w-px bg-slate-800 mx-1" />

        <button
          onClick={onToggleGroundTruth}
          className={`flex items-center space-x-1 px-2 py-1 rounded transition-colors cursor-pointer ${
            showTargetGroundTruth
              ? 'bg-rose-950/80 border border-rose-600 text-rose-300'
              : 'bg-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          {showTargetGroundTruth ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          <span>Vérité Terrain</span>
        </button>

        <button
          onClick={onToggleAllRunsOverlay}
          disabled={!showTargetGroundTruth}
          title={showTargetGroundTruth ? 'Afficher la dispersion des trajectoires cibles Monte-Carlo' : 'Afficher d’abord la vérité terrain'}
          className={`flex items-center space-x-1 px-2 py-1 rounded transition-colors cursor-pointer ${
            showAllRunsOverlay
              ? 'bg-cyan-950/80 border border-cyan-500 text-cyan-300'
              : 'bg-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Dispersion des cibles</span>
        </button>
      </div>

      {/* Dynamic Comparison Legend Badge (SIGMA Green vs Naïve Amber) */}
      <div className="tactical-legend absolute bottom-3 left-3 z-10 flex items-center space-x-3 px-3 py-1.5 text-xs font-mono">
        <div className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
          <span className="text-purple-300 font-bold">Hybride 2027</span>
          {selectedRlRun && <span className="text-[10px] text-slate-400">({selectedRlRun.intercepted ? `${selectedRlRun.interceptionTime.toFixed(0)} min` : selectedRlRun.outcome})</span>}
        </div>
        <div className="h-3.5 w-px bg-slate-700" />
        <div className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
          <span className="text-emerald-300 font-bold">Bayésien</span>
          {selectedSigmaRun && (
            <span className="text-[10px] text-slate-400">
              ({selectedSigmaRun.intercepted ? `${selectedSigmaRun.interceptionTime.toFixed(0)} min` : 'Bingo'})
            </span>
          )}
        </div>

        <div className="h-3.5 w-px bg-slate-800" />

        <div className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
          <span className="text-amber-300 font-bold">Râteau IAMSAR</span>
          {selectedNaiveRun && (
            <span className="text-[10px] text-slate-400">
              ({selectedNaiveRun.intercepted ? `${selectedNaiveRun.interceptionTime.toFixed(0)} min` : 'Bingo'})
            </span>
          )}
        </div>
      </div>

      {/* Zoom / Pan Controls */}
      <div className="tactical-zoom absolute bottom-3 right-3 flex items-center space-x-1 p-1 z-10">
        <button
          onClick={() => setZoom((z) => Math.min(3.0, z * 1.2))}
          className="p-1.5 hover:bg-slate-800 text-slate-300 rounded cursor-pointer"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(0.4, z / 1.2))}
          className="p-1.5 hover:bg-slate-800 text-slate-300 rounded cursor-pointer"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => {
            setZoom(1.0);
            setPan({ x: 0, y: 0 });
          }}
          className="p-1.5 hover:bg-slate-800 text-slate-300 rounded cursor-pointer"
          title="Réinitialiser vue"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
