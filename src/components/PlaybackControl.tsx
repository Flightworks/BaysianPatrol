import React, { useEffect, useState } from 'react';
import type { MonteCarloRunResult } from '../types/simulation';
import { Play, Pause, SkipBack, SkipForward, Eye, EyeOff } from 'lucide-react';

interface PlaybackControlProps {
  runs: MonteCarloRunResult[];
  selectedRunIndex: number;
  onSelectRunIndex: (index: number) => void;
  currentTime: number; // minutes
  onSeekTime: React.Dispatch<React.SetStateAction<number>>;
  showGroundTruth: boolean;
  onToggleGroundTruth: () => void;
}

export const PlaybackControl: React.FC<PlaybackControlProps> = ({
  runs,
  selectedRunIndex,
  onSelectRunIndex,
  currentTime,
  onSeekTime,
  showGroundTruth,
  onToggleGroundTruth,
}) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(2); // 2x speed default

  const currentRun = runs[selectedRunIndex] || runs[0];
  const maxTime = currentRun
    ? Math.max(
        ...currentRun.helicoPath.map((p) => p.t),
        ...currentRun.targetPath.map((p) => p.t),
        120
      )
    : 180;

  // Playback timer tick
  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = 200 / playbackSpeed;
    const timer = setInterval(() => {
      onSeekTime((prev: number) => {
        if (prev >= maxTime) {
          setIsPlaying(false);
          return maxTime;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, maxTime, onSeekTime]);

  if (!currentRun) return null;

  // Current status info
  const hPoint = currentRun.helicoPath.find((p) => p.t >= currentTime) || currentRun.helicoPath[currentRun.helicoPath.length - 1];
  const tPoint = currentRun.targetPath.find((p) => p.t >= currentTime) || currentRun.targetPath[currentRun.targetPath.length - 1];
  const distToTarget = hPoint && tPoint ? Math.hypot(hPoint.x - tPoint.x, hPoint.y - tPoint.y) : null;

  return (
    <div className="glass-panel rounded-xl p-3 space-y-3">
      {/* Header & Run Selector */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-3">
          <span className="text-xs font-bold text-cyan-300 uppercase tracking-wider">
            Rejeu Tactique Pas-à-Pas
          </span>

          {/* Run Selector Dropdown */}
          <select
            value={selectedRunIndex}
            onChange={(e) => {
              onSelectRunIndex(parseInt(e.target.value));
              onSeekTime(0);
              setIsPlaying(false);
            }}
            className="bg-slate-950 text-cyan-200 text-xs border border-cyan-800 rounded px-2.5 py-1 font-mono focus:border-cyan-500"
          >
            {runs.slice(0, 100).map((r, idx) => (
              <option key={r.runId} value={idx}>
                Run #{r.runId} ({r.strategy}) - {r.intercepted ? '✅ Intercepté' : r.bingoTriggered ? '⚠️ Bingo' : '❌ Échec'} ({r.interceptionTime.toFixed(0)}m)
              </option>
            ))}
          </select>
        </div>

        {/* Status Pills */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 px-2 py-0.5 rounded">
            <span className="text-slate-400">Temps:</span>
            <span className="text-cyan-300 font-bold">{currentTime} min</span>
          </div>

          {distToTarget !== null && (
            <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 px-2 py-0.5 rounded">
              <span className="text-slate-400">Dist Cible:</span>
              <span className="text-emerald-400 font-bold">{distToTarget.toFixed(1)} NM</span>
            </div>
          )}

          {/* Ground Truth Reveal Toggle */}
          <button
            onClick={onToggleGroundTruth}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              showGroundTruth
                ? 'bg-rose-950/80 border border-rose-600 text-rose-300'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {showGroundTruth ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>{showGroundTruth ? 'Cible Démasquée' : 'Masquer Cible'}</span>
          </button>
        </div>
      </div>

      {/* Playback Controls & Slider */}
      <div className="flex items-center space-x-3">
        {/* Buttons */}
        <div className="flex items-center space-x-1">
          <button
            onClick={() => onSeekTime(0)}
            className="p-1.5 hover:bg-slate-800 text-slate-300 rounded cursor-pointer"
            title="Début"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-lg green-glow transition-all cursor-pointer"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-slate-950" />}
          </button>

          <button
            onClick={() => onSeekTime(Math.min(maxTime, currentTime + 5))}
            className="p-1.5 hover:bg-slate-800 text-slate-300 rounded cursor-pointer"
            title="+5 minutes"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Speed Selector */}
          <button
            onClick={() => setPlaybackSpeed((s) => (s === 1 ? 2 : s === 2 ? 5 : s === 5 ? 10 : 1))}
            className="bg-slate-900 border border-slate-800 text-cyan-300 font-mono text-xs px-2 py-1 rounded hover:bg-slate-800 cursor-pointer ml-1"
          >
            {playbackSpeed}x
          </button>
        </div>

        {/* Timeline Range Slider */}
        <div className="flex-1 flex items-center space-x-2">
          <span className="text-[10px] text-slate-500 font-mono">0m</span>
          <input
            type="range"
            min="0"
            max={maxTime}
            value={currentTime}
            onChange={(e) => {
              onSeekTime(parseInt(e.target.value));
              setIsPlaying(false);
            }}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <span className="text-[10px] text-slate-500 font-mono">{maxTime}m</span>
        </div>
      </div>
    </div>
  );
};
