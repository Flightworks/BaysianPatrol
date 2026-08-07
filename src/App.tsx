import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Play } from 'lucide-react';
import type { GlobalSimulationResult, ScenarioConfig } from './types/simulation';
import { PRESETS } from './engine/presets';
import { runMonteCarloSuite } from './engine/simulator';
import {
  RUN_HISTORY_STORAGE_KEY,
  createHistoryEntry,
  parseHistory,
  prependHistory,
  serializeHistory,
  type RunHistoryEntry,
} from './engine/runHistory';
import { Navbar, type AppTab } from './components/Navbar';
import { ControlPanel } from './components/ControlPanel';
import { TacticalCanvas } from './components/TacticalCanvas';
import { StatsDashboard } from './components/StatsDashboard';
import { PlaybackControl } from './components/PlaybackControl';
import { HistoryPanel } from './components/HistoryPanel';

export function App() {
  const [selectedPresetId, setSelectedPresetId] = useState(PRESETS[0].id);
  const [config, setConfig] = useState<ScenarioConfig>({ ...PRESETS[0].config, strategy: 'TRIO' });
  const [simResult, setSimResult] = useState<GlobalSimulationResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<AppTab>('comparison');
  const [history, setHistory] = useState<RunHistoryEntry[]>(() => {
    try {
      return parseHistory(window.localStorage.getItem(RUN_HISTORY_STORAGE_KEY));
    } catch {
      return [];
    }
  });

  const [selectedRunIndex, setSelectedRunIndex] = useState(0);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [showGroundTruth, setShowGroundTruth] = useState(false);
  const [showAllRunsOverlay, setShowAllRunsOverlay] = useState(false);
  const activeWorkerRef = useRef<Worker | null>(null);

  useEffect(() => () => activeWorkerRef.current?.terminate(), []);

  const selectedPreset = PRESETS.find((preset) => preset.id === selectedPresetId) ?? PRESETS[0];

  const updateHistory = (result: GlobalSimulationResult) => {
    const entry = createHistoryEntry(result, selectedPreset.name);
    setHistory((current) => {
      const updated = prependHistory(current, entry);
      try {
        window.localStorage.setItem(RUN_HISTORY_STORAGE_KEY, serializeHistory(updated));
      } catch (error) {
        console.warn('Historique local indisponible :', error);
      }
      return updated;
    });
  };

  const completeSimulation = (result: GlobalSimulationResult) => {
    setSimResult(result);
    setSelectedRunIndex(0);
    setPlaybackTime(0);
    setProgress(100);
    setIsRunning(false);
    updateHistory(result);
  };

  const handleSelectPreset = (presetId: string) => {
    const preset = PRESETS.find((item) => item.id === presetId);
    if (!preset) return;
    setSelectedPresetId(presetId);
    setConfig({ ...preset.config, strategy: 'TRIO' });
    setSimResult(null);
  };

  const handleConfigChange = (nextConfig: ScenarioConfig) => {
    setConfig({ ...nextConfig, strategy: 'TRIO' });
    setSimResult(null);
  };

  const runFallbackAsync = () => {
    window.setTimeout(async () => {
      try {
        completeSimulation(await runMonteCarloSuite(config, setProgress));
      } catch (error) {
        console.error('Erreur de simulation :', error);
        setIsRunning(false);
      }
    }, 50);
  };

  const handleRunSimulation = () => {
    if (isRunning) return;
    setIsRunning(true);
    setProgress(0);
    setActiveTab('comparison');
    activeWorkerRef.current?.terminate();

    try {
      const worker = new Worker(new URL('./engine/mcWorker.ts', import.meta.url), { type: 'module' });
      activeWorkerRef.current = worker;
      worker.onmessage = (event: MessageEvent) => {
        const data = event.data;
        if (data.type === 'PROGRESS') setProgress(data.progress);
        if (data.type === 'COMPLETE') {
          worker.terminate();
          activeWorkerRef.current = null;
          completeSimulation(data.result);
        }
        if (data.type === 'ERROR') {
          console.error('Erreur du Worker Monte-Carlo :', data.error);
          worker.terminate();
          activeWorkerRef.current = null;
          runFallbackAsync();
        }
      };
      worker.onerror = (event) => {
        console.error('Crash du Worker Monte-Carlo :', event.message, event.error);
        worker.terminate();
        activeWorkerRef.current = null;
        runFallbackAsync();
      };
      worker.postMessage({ config: { ...config, strategy: 'TRIO' } });
    } catch {
      runFallbackAsync();
    }
  };

  const activeRunList = simResult?.rlRuns.length
    ? simResult.rlRuns
    : simResult?.sigmaRuns.length
      ? simResult.sigmaRuns
      : simResult?.naiveRuns ?? [];

  return (
    <div className="app-frame">
      <Navbar activeTab={activeTab} onChangeTab={setActiveTab} historyCount={history.length} />

      <main className="app-shell">
        {activeTab === 'comparison' && (
          <>
            <section className="campaign-heading">
              <div>
                <p className="eyebrow">Évaluation opérationnelle</p>
                <h1>Comparer trois stratégies, sur les mêmes situations.</h1>
                <p className="campaign-intro">
                  Chaque tirage reproduit la même cible, la même météo et la même position frégate pour les trois méthodes.
                </p>
              </div>
              <div className="campaign-seed">
                <span>Stratégie active</span>
                <strong>Hybride · qualification 2027</strong>
              </div>
            </section>

            <section className="campaign-bar" aria-label="Configuration rapide de campagne">
              <label className="campaign-field campaign-field-wide">
                <span>Scénario</span>
                <select value={selectedPresetId} onChange={(event) => handleSelectPreset(event.target.value)} disabled={isRunning}>
                  {PRESETS.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
                </select>
              </label>
              <label className="campaign-field">
                <span>Tirages</span>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={config.numIterations}
                  onChange={(event) => handleConfigChange({ ...config, numIterations: Math.max(1, Number(event.target.value) || 1) })}
                  disabled={isRunning}
                />
              </label>
              <label className="campaign-field">
                <span>Seed Monte-Carlo</span>
                <input
                  type="number"
                  value={config.monteCarloSeed ?? 2026}
                  onChange={(event) => handleConfigChange({ ...config, monteCarloSeed: Number(event.target.value) || 2026 })}
                  disabled={isRunning}
                />
              </label>
              <button className="run-button" onClick={handleRunSimulation} disabled={isRunning}>
                <Play size={17} fill="currentColor" />
                <span>{isRunning ? `Calcul ${progress.toFixed(0)} %` : 'Lancer la campagne'}</span>
              </button>
              {isRunning && <div className="campaign-progress" style={{ width: `${progress}%` }} />}
            </section>

            <details className="parameter-drawer">
              <summary><ChevronDown size={16} /> Paramètres détaillés <span>{selectedPreset.description}</span></summary>
              <ControlPanel config={config} onChange={handleConfigChange} disabled={isRunning} />
            </details>

            {simResult ? (
              <StatsDashboard result={simResult} />
            ) : (
              <section className="empty-comparison">
                <span>01</span>
                <div>
                  <h2>Prêt pour la démonstration</h2>
                  <p>Lancez une campagne pour comparer la stratégie hybride, la recherche bayésienne et le balayage parallèle IAMSAR.</p>
                </div>
              </section>
            )}
          </>
        )}

        {activeTab === 'tactical' && (
          <section className="tactical-workspace">
            <header className="section-heading">
              <div><p className="eyebrow">Rejeu d'un tirage</p><h1>Carte tactique</h1></div>
              <p>Les trois trajectoires sont superposées sur une situation strictement identique.</p>
            </header>
            {simResult ? (
              <>
                <TacticalCanvas
                  config={config}
                  selectedSigmaRun={simResult.sigmaRuns[selectedRunIndex] ?? null}
                  selectedNaiveRun={simResult.naiveRuns[selectedRunIndex] ?? null}
                  selectedRlRun={simResult.rlRuns[selectedRunIndex] ?? null}
                  currentPlaybackTime={playbackTime}
                  showTargetGroundTruth={showGroundTruth}
                  onToggleGroundTruth={() => setShowGroundTruth((value) => !value)}
                  allRuns={activeRunList}
                  showAllRunsOverlay={showAllRunsOverlay}
                  onToggleAllRunsOverlay={() => setShowAllRunsOverlay((value) => !value)}
                />
                <PlaybackControl
                  runs={activeRunList}
                  selectedRunIndex={selectedRunIndex}
                  onSelectRunIndex={setSelectedRunIndex}
                  currentTime={playbackTime}
                  onSeekTime={setPlaybackTime}
                  showGroundTruth={showGroundTruth}
                  onToggleGroundTruth={() => setShowGroundTruth((value) => !value)}
                />
              </>
            ) : (
              <section className="empty-comparison"><span>02</span><div><h2>Aucun tirage à rejouer</h2><p>Lancez d’abord une campagne depuis l’onglet Comparaison.</p></div></section>
            )}
          </section>
        )}

        {activeTab === 'history' && (
          <HistoryPanel history={history} onClear={() => {
            setHistory([]);
            window.localStorage.removeItem(RUN_HISTORY_STORAGE_KEY);
          }} />
        )}
      </main>
    </div>
  );
}

export default App;
