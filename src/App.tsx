import { useState, useEffect, useRef } from 'react';
import type { ScenarioConfig, GlobalSimulationResult } from './types/simulation';
import { PRESETS } from './engine/presets';
import { runMonteCarloSuite } from './engine/simulator';
import { Navbar } from './components/Navbar';
import { ControlPanel } from './components/ControlPanel';
import { TacticalCanvas } from './components/TacticalCanvas';
import { StatsDashboard } from './components/StatsDashboard';
import { PlaybackControl } from './components/PlaybackControl';
import { RlTrainingPanel } from './components/RlTrainingPanel';
import { Map, BarChart3, Compass, Cpu, Play } from 'lucide-react';

export function App() {
  const [config, setConfig] = useState<ScenarioConfig>(PRESETS[0].config);
  const [simResult, setSimResult] = useState<GlobalSimulationResult | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);

  const [selectedRunIndex, setSelectedRunIndex] = useState<number>(0);
  const [playbackTime, setPlaybackTime] = useState<number>(0);
  const [showGroundTruth, setShowGroundTruth] = useState<boolean>(true);
  const [showAllRunsOverlay, setShowAllRunsOverlay] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'training' | 'tactical' | 'stats'>('training');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

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
    setTimeout(async () => {
      try {
        const result = await runMonteCarloSuite(config, (p) => setProgress(p));
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
  const selectedRlRun = rlRuns[selectedRunIndex] || null;
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
        {/* Left Column: Control Panel (Hidden on RL Training Tab or when Collapsed) */}
        {!sidebarCollapsed && activeTab !== 'training' && (
          <aside className="lg:col-span-4 space-y-4">
            <ControlPanel config={config} onChange={setConfig} disabled={isRunning} />
          </aside>
        )}

        {/* Right Column: Interactive Tactical Map / Statistics & Replay */}
        <section className={`${sidebarCollapsed || activeTab === 'training' ? 'lg:col-span-12' : 'lg:col-span-8'} flex flex-col space-y-4`}>
          {/* Tab Navigation & Controls */}
          <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 rounded-xl p-1.5 glass-panel">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-all cursor-pointer mr-1"
                title={sidebarCollapsed ? "Déplier le panneau de configuration" : "Replier le panneau pour plein écran"}
              >
                <span>{sidebarCollapsed ? "⚙️ Afficher Paramètres" : "◀ Replier Panneau"}</span>
              </button>

              <button
                onClick={() => setActiveTab('training')}
                className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  activeTab === 'training'
                    ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md green-glow'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Cpu className="w-4 h-4 text-purple-300" />
                <span>🏋️ Entraînement & Recherche RL (Priorité 1)</span>
              </button>

              <button
                onClick={() => setActiveTab('tactical')}
                className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  activeTab === 'tactical'
                    ? 'bg-gradient-to-r from-cyan-600 to-sky-600 text-white shadow-md green-glow'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Map className="w-4 h-4" />
                <span>Carte Tactique</span>
              </button>

              <button
                onClick={() => setActiveTab('stats')}
                className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  activeTab === 'stats'
                    ? 'bg-gradient-to-r from-cyan-600 to-sky-600 text-white shadow-md green-glow'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                <span>Tableau de Bord ({config.numIterations} runs)</span>
              </button>
            </div>

            <div className="flex items-center space-x-3">
              {/* Monte-Carlo Launcher & Run Count Controls */}
              {activeTab !== 'training' && (
                <div className="flex items-center space-x-2 bg-slate-950/80 border border-slate-700/80 rounded-lg px-2 py-1">
                  <span className="text-xs text-slate-300 font-bold hidden md:inline">Runs:</span>
                  <input
                    type="number"
                    min={1}
                    max={1000}
                    value={config.numIterations}
                    onChange={(e) => setConfig({ ...config, numIterations: Math.max(1, parseInt(e.target.value) || 1) })}
                    disabled={isRunning}
                    className="w-16 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-cyan-300 font-mono font-bold text-center"
                    title="Nombre de tirages Monte-Carlo"
                  />
                  <button
                    onClick={handleRunSimulation}
                    disabled={isRunning}
                    className="flex items-center space-x-1.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold px-3 py-1.5 rounded-lg text-xs shadow-md transition-all cursor-pointer disabled:opacity-50"
                  >
                    <Play className="w-3.5 h-3.5 fill-white" />
                    <span>{isRunning ? 'Calculs...' : '▶️ Lancer Simulation Monte-Carlo'}</span>
                  </button>
                </div>
              )}

              {simResult && activeTab !== 'training' && (
                <span className="text-[11px] font-mono text-cyan-400 hidden xl:inline-block px-2">
                  {simResult.trioStats ? (
                    <>🤖 RL: {simResult.rlStats?.successRate.toFixed(1) || 0}% | ⚡ AMI: {simResult.sigmaStats?.successRate.toFixed(1) || 0}% | 📐 Naïf: {simResult.naiveStats?.successRate.toFixed(1) || 0}%</>
                  ) : (
                    <>⚡ SIGMA: {simResult.sigmaStats?.successRate.toFixed(1) || 0}% | 📐 Naïf: {simResult.naiveStats?.successRate.toFixed(1) || 0}%</>
                  )}
                </span>
              )}
            </div>
          </div>

          {/* Tab View Content */}
          {activeTab === 'tactical' ? (
            <div className="flex-1 flex flex-col space-y-4">
              <TacticalCanvas
                config={config}
                selectedSigmaRun={selectedSigmaRun}
                selectedNaiveRun={selectedNaiveRun}
                selectedRlRun={selectedRlRun}
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
          ) : activeTab === 'stats' ? (
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
          ) : (
            <div className="flex-1">
              <RlTrainingPanel onNavigateToTactical={() => setActiveTab('tactical')} />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
