import React, { useEffect, useState } from 'react';
import type { MonteCarloRunResult } from '../types/simulation';
import { Eye, EyeOff, Pause, Play, RotateCcw, SkipForward } from 'lucide-react';

interface PlaybackControlProps {
  runs: MonteCarloRunResult[];
  selectedRunIndex: number;
  onSelectRunIndex: (index: number) => void;
  currentTime: number;
  onSeekTime: React.Dispatch<React.SetStateAction<number>>;
  showGroundTruth: boolean;
  onToggleGroundTruth: () => void;
}

const outcomeLabel = (run: MonteCarloRunResult) => {
  if (run.intercepted) return `intercepté · ${run.interceptionTime.toFixed(0)} min`;
  if (run.safeReturn) return 'retour sûr';
  return run.outcome.toLowerCase().replaceAll('_', ' ');
};

export const PlaybackControl: React.FC<PlaybackControlProps> = ({
  runs,
  selectedRunIndex,
  onSelectRunIndex,
  currentTime,
  onSeekTime,
  showGroundTruth,
  onToggleGroundTruth,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2);
  const currentRun = runs[selectedRunIndex] ?? runs[0];
  const maxTime = currentRun ? Math.max(...currentRun.helicoPath.map((point) => point.t), ...currentRun.targetPath.map((point) => point.t), 120) : 180;

  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      onSeekTime((previous) => {
        if (previous >= maxTime) {
          setIsPlaying(false);
          return maxTime;
        }
        return previous + 1;
      });
    }, 200 / playbackSpeed);
    return () => window.clearInterval(timer);
  }, [isPlaying, maxTime, onSeekTime, playbackSpeed]);

  if (!currentRun) return null;
  const helico = currentRun.helicoPath.find((point) => point.t >= currentTime) ?? currentRun.helicoPath.at(-1);
  const target = currentRun.targetPath.find((point) => point.t >= currentTime) ?? currentRun.targetPath.at(-1);
  const distance = helico && target ? Math.hypot(helico.x - target.x, helico.y - target.y) : null;

  return (
    <div className="playback-panel">
      <div className="playback-meta">
        <label>
          <span>Tirage affiché</span>
          <select value={selectedRunIndex} onChange={(event) => {
            onSelectRunIndex(Number(event.target.value));
            onSeekTime(0);
            setIsPlaying(false);
          }}>
            {runs.slice(0, 100).map((run, index) => <option key={run.runId} value={index}>#{run.runId} · {outcomeLabel(run)}</option>)}
          </select>
        </label>
        <dl>
          <div><dt>Temps</dt><dd>{currentTime} min</dd></div>
          <div><dt>Distance cible</dt><dd>{distance === null ? '—' : `${distance.toFixed(1)} NM`}</dd></div>
        </dl>
        <button className={showGroundTruth ? 'truth-button active' : 'truth-button'} onClick={onToggleGroundTruth}>
          {showGroundTruth ? <Eye size={15} /> : <EyeOff size={15} />}
          {showGroundTruth ? 'Vérité terrain visible' : 'Afficher la vérité terrain'}
        </button>
      </div>

      <div className="playback-timeline">
        <div className="playback-actions">
          <button onClick={() => onSeekTime(0)} title="Revenir au début"><RotateCcw size={16} /></button>
          <button className="playback-primary" onClick={() => setIsPlaying((value) => !value)} title={isPlaying ? 'Pause' : 'Lecture'}>
            {isPlaying ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}
          </button>
          <button onClick={() => onSeekTime(Math.min(maxTime, currentTime + 5))} title="Avancer de cinq minutes"><SkipForward size={16} /></button>
          <button className="speed-button" onClick={() => setPlaybackSpeed((speed) => speed === 1 ? 2 : speed === 2 ? 5 : speed === 5 ? 10 : 1)}>{playbackSpeed}×</button>
        </div>
        <span>0</span>
        <input type="range" min={0} max={maxTime} value={currentTime} onChange={(event) => { onSeekTime(Number(event.target.value)); setIsPlaying(false); }} />
        <span>{maxTime} min</span>
      </div>
    </div>
  );
};
