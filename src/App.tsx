import { useState, useEffect, useRef } from 'react';
import type { ScenarioConfig, GlobalSimulationResult } from './types/simulation';
import { PRESETS } from './engine/presets';
import { runMonteCarloSuite } from './engine/simulator';
import { Navbar } from './components/Navbar';
import { ControlPanel } from './components/ControlPanel';
import { TacticalCanvas } from './components/TacticalCanvas';
import { StatsDashboard } from './components/StatsDashboard';
import { PlaybackControl } from './components/PlaybackControl';
import { Map, BarChart3, Compass } from 'lucide-react';

export function App() {
  const [config, setConfig] = useState<ScenarioConfig>(PRESETS[0].config);
  const [simResult, setSimResult] = useState<GlobalSimulationResult | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);

  const [selectedRunIndex, setSelectedRunIndex] = useState<number>(0);
  const [playbackTime, setPlaybackTime] = useState<number>(0);
  const [showGroundTruth, setShowGroundTruth] = useState<boolean>(true);
  const [showAllRunsOverlay, setShowAllRunsOverlay] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'tactical' | 'stats'>('tactical');

  const activeWorkerRef = useRef<Worker | null>(null);

  useEffect(() => {
    return () => {
      if (activeWorkerRef.current) {
        activeWorkerRef.current.terminate();
      }
    };
  }, []);

  const handleSelectPreset = (presetId: string) => {
    const preset = PRESETS.find((p) => p.id === presetId);
    if (preset) {
      setConfig({ ...preset.config });
      setSimResult(null);
    }
  };

  const handleRunSimulation = () => {
    if (isRunning) return;

    setIsRunning(true);
    setProgress(0);

    if (activeWorkerRef.current) {
      activeWorkerRef.current.terminate();
      activeWorkerRef.current = null;
    }

    try {
      const worker = new Worker(new URL('./engine/mcWorker.ts', import.meta.url), {
        type: 'module',
      });
      activeWorkerRef.current = worker;

      worker.onmessage = (e: MessageEvent) => {
        const data = e.data;
        if (data.type === 'PROGRESS') {
          setProgress(data.progress);
        } else if (data.type === 'COMPLETE') {
          setSimResult(data.result);
          setSelectedRunIndex(0);
          setPlaybackTime(0);
          setProgress(100);
          setIsRunning(false);
          worker.terminate();
          activeWorkerRef.current = null;
        } else if (data.type === 'ERROR') {
          console.error('Worker simulation error:', data.error);
          runFallbackAsync();
        }
      };

      worker.onerror = (err) => {
        console.warn('Worker instantiation failed, falling back to async chunking:', err);
        worker.terminate();
        activeWorkerRef.current = null;
        runFallbackAsync();
      };

      worker.postMessage({ config });
    } catch (err) {
      console.warn('Worker error, running async fallback:', err);
      runFallbackAsync();
    }
  };

  const runFallbackAsync = () => {
    setTimeout(() => {
      try {
        const result = runMonteCarloSuite(config, (p) => setProgress(p));
        setSimResult(result);
        setSelectedRunIndex(0);
        setPlaybackTime(0);
        setProgress(100);
      } catch (err) {
        console.error('Simulation fallback error:', err);
      } finally {
        setIsRunning(false);
      }
    }, 50);
  };

  const sigmaRuns = simResult ? simResult.sigmaRuns : [];
  const naiveRuns = simResult ? simResult.naiveRuns : [];
  const rlRuns = simResult ? simResult.rlRuns : [];
  
  const selectedSigmaRun = sigmaRuns[selectedRunIndex] || null;
  const selectedNaiveRun = naiveRuns[selectedRunIndex] || null;
  const activeRunList = rlRuns.length > 0 ? rlRuns : sigmaRuns.length > 0 ? sigmaRuns : naiveRuns;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Header Bar */}
      <Navbar
        config={config}
        onSelectPreset={handleSelectPreset}
        onRunSimulation={handleRunSimulation}
        isRunning={isRunning}
        progress={progress}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Control Panel */}
        <aside className="lg:col-span-4 space-y-4">
          <ControlPanel config={config} onChange={setConfig} disabled={isRunning} />
        </aside>

        {/* Right Column: Interactive Tactical Map / Statistics & Replay */}
        <section className="lg:col-span-8 flex flex-col space-y-4">
          {/* Tab Navigation */}
          <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 rounded-xl p-1.5 glass-panel">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setActiveTab('tactical')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  activeTab === 'tactical'
                    ? 'bg-gradient-to-r from-cyan-600 to-sky-600 text-white shadow-md green-glow'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Map className="w-4 h-4" />
                <span>Carte Tactique & Visualisation Monte-Carlo</span>
              </button>

              <button
                onClick={() => setActiveTab('stats')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  activeTab === 'stats'
                    ? 'bg-gradient-to-r from-cyan-600 to-sky-600 text-white shadow-md green-glow'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                <span>Tableau de Bord Statistique ({config.numIterations} runs)</span>
              </button>
            </div>

            {simResult && (
              <span className="text-[11px] font-mono text-cyan-400 hidden sm:inline-block px-2">
                {simResult.trioStats ? (
                  <>🤖 SIGMA (RL): {simResult.rlStats?.successRate.toFixed(1) || 0}% | ⚡ POMDP: {simResult.sigmaStats?.successRate.toFixed(1) || 0}% | 📐 Naïf: {simResult.naiveStats?.successRate.toFixed(1) || 0}%</>
                ) : (
                  <>⚡ POMDP: {simResult.sigmaStats?.successRate.toFixed(1) || 0}% | 📐 Naïf: {simResult.naiveStats?.successRate.toFixed(1) || 0}%</>
                )}
              </span>
            )}
          </div>

          {/* Tab View Content */}
          {activeTab === 'tactical' ? (
            <div className="flex-1 flex flex-col space-y-4">
              <TacticalCanvas
                config={config}
                selectedSigmaRun={selectedSigmaRun}
                selectedNaiveRun={selectedNaiveRun}
                currentPlaybackTime={playbackTime}
                showTargetGroundTruth={showGroundTruth}
                onToggleGroundTruth={() => setShowGroundTruth(!showGroundTruth)}
                allRuns={activeRunList}
                showAllRunsOverlay={showAllRunsOverlay}
                onToggleAllRunsOverlay={() => setShowAllRunsOverlay(!showAllRunsOverlay)}
              />

              {/* Playback Control Footer */}
              {activeRunList.length > 0 && (
                <PlaybackControl
                  runs={activeRunList}
                  selectedRunIndex={selectedRunIndex}
                  onSelectRunIndex={setSelectedRunIndex}
                  currentTime={playbackTime}
                  onSeekTime={setPlaybackTime}
                  showGroundTruth={showGroundTruth}
                  onToggleGroundTruth={() => setShowGroundTruth(!showGroundTruth)}
                />
              )}
            </div>
          ) : (
            <div className="flex-1">
              {simResult ? (
                <StatsDashboard result={simResult} />
              ) : (
                <div className="glass-panel rounded-xl p-8 text-center text-slate-400">
                  <Compass className="w-12 h-12 mx-auto mb-2 text-cyan-400" />
                  <p>Cliquez sur "Lancer Simulation Monte-Carlo" pour démarrer les calculs.</p>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
