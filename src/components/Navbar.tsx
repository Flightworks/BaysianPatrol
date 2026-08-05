import React, { useEffect, useState } from 'react';
import type { ScenarioConfig } from '../types/simulation';
import { PRESETS } from '../engine/presets';
import { detectGPUInfo, type GPUInfo } from '../engine/webglRenderer';
import { Play, Loader2, Compass, Layers, Cpu, CheckCircle2 } from 'lucide-react';

interface NavbarProps {
  config: ScenarioConfig;
  onSelectPreset: (presetId: string) => void;
  onRunSimulation: () => void;
  isRunning: boolean;
  progress: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  config,
  onSelectPreset,
  onRunSimulation,
  isRunning,
  progress,
}) => {
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);

  useEffect(() => {
    setGpuInfo(detectGPUInfo());
  }, []);

  return (
    <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur border-b border-slate-800 px-4 py-2.5 flex flex-wrap items-center justify-between gap-4">
      {/* Title & Logo */}
      <div className="flex items-center space-x-3">
        <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 text-slate-950 shadow-md">
          <Compass className="w-5 h-5 font-bold" />
        </div>
        <div>
          <h1 className="text-base font-black tracking-tight text-white flex items-center space-x-2">
            <span>BAYESIAN PATROL</span>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-cyan-950 border border-cyan-700/50 text-cyan-300">
              Monte-Carlo v2.5
            </span>
          </h1>
          <p className="text-[11px] text-slate-400">
            Simulateur de recherche maritime bayésienne & trajectoire d'interception optimale
          </p>
        </div>
      </div>

      {/* Center: Preset Scenario Selector & Clean Integrated GPU Badge */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2 bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-1.5 text-xs">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-400 font-semibold hidden sm:inline">Scénario:</span>
          <select
            onChange={(e) => onSelectPreset(e.target.value)}
            disabled={isRunning}
            className="bg-transparent text-slate-200 font-bold focus:outline-none cursor-pointer"
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id} className="bg-slate-900 text-slate-200">
                {p.name}
              </option>
            ))}
          </select>
        </div>

        {/* Clean GPU Status Badge */}
        {gpuInfo && (
          <div
            className={`hidden md:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg border text-[11px] font-mono transition-all ${
              gpuInfo.isDedicatedNvidia
                ? 'bg-emerald-950/80 border-emerald-500/50 text-emerald-300'
                : 'bg-slate-950/80 border-slate-800 text-slate-400'
            }`}
            title={`Processeur graphique actif : ${gpuInfo.renderer} (${gpuInfo.webglVersion})`}
          >
            <Cpu className={`w-3.5 h-3.5 ${gpuInfo.isDedicatedNvidia ? 'text-emerald-400 animate-pulse' : 'text-cyan-400'}`} />
            <span className="truncate max-w-[170px] font-semibold">
              {gpuInfo.cleanName}
            </span>
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          </div>
        )}
      </div>

      {/* Right Action: Monte-Carlo Trigger Button & Progress Bar */}
      <div className="flex items-center space-x-3">
        {isRunning && (
          <div className="flex flex-col items-end text-xs font-mono">
            <span className="text-cyan-400 font-bold">{progress.toFixed(0)}%</span>
            <div className="w-24 bg-slate-800 h-1.5 rounded-full overflow-hidden mt-0.5">
              <div
                className="bg-cyan-500 h-full transition-all duration-200"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        <button
          onClick={onRunSimulation}
          disabled={isRunning}
          className={`flex items-center space-x-2 px-5 py-2 rounded-lg text-xs font-black uppercase tracking-wider transition-all cursor-pointer shadow-lg ${
            isRunning
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
              : 'bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-600 text-slate-950 hover:brightness-110 active:scale-95 shadow-cyan-900/40 green-glow'
          }`}
        >
          {isRunning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span>Calcul en cours...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Lancer Simulation Monte-Carlo ({config.numIterations} runs)</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
