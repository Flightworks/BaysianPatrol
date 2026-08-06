import React, { useState, useEffect } from 'react';
import { Play, Square, Cpu, RefreshCw, CheckCircle2, AlertCircle, Info, HelpCircle, Check, Sparkles, Sliders, Bot, Trophy, Flame, Trash2, Rocket, ExternalLink, Zap } from 'lucide-react';

interface RlStatus {
  is_training: boolean;
  progress_percent: number;
  current_timestep: number;
  total_timesteps: number;
  mean_reward: number;
  status_message: string;
}

interface ModelItem {
  filename: string;
  title: string;
  type: string;
  size_mb: number;
  date: string;
}

interface AutoResearchHistoryItem {
  experiment: number;
  name: string;
  timestamp: string;
  score: number;
  mean_reward: number;
  success_rate: number;
  mean_flight_length: number;
  hyperparams: {
    lr: number;
    r_det: number;
    r_prob: number;
    p_time: number;
    p_bingo: number;
    gamma: number;
  };
  status: 'CHAMPION_KEPT' | 'DISCARDED';
}

interface AutoResearchData {
  best_score: number;
  best_params: {
    lr: number;
    r_det: number;
    r_prob: number;
    p_time: number;
    p_bingo: number;
    gamma: number;
  } | null;
  history: AutoResearchHistoryItem[];
}

interface RlTrainingPanelProps {
  onNavigateToTactical?: () => void;
}

export const RlTrainingPanel: React.FC<RlTrainingPanelProps> = ({ onNavigateToTactical }) => {
  const [status, setStatus] = useState<RlStatus>({
    is_training: false,
    progress_percent: 0,
    current_timestep: 0,
    total_timesteps: 50000,
    mean_reward: 0,
    status_message: "Serveur d'entraînement prêt.",
  });

  const [timesteps, setTimesteps] = useState<number>(50000);
  const [customModelName, setCustomModelName] = useState<string>('PPO_8');
  
  // Custom Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.0003);
  const [rewardDetection, setRewardDetection] = useState<number>(500.0);
  const [rewardExploration, setRewardExploration] = useState<number>(10.0);
  const [penaltyTime, setPenaltyTime] = useState<number>(0.1);
  const [penaltyBingo, setPenaltyBingo] = useState<number>(1000.0);

  const [availableModels, setAvailableModels] = useState<ModelItem[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [activeModelConfirmed, setActiveModelConfirmed] = useState<string>('');
  const [serverOnline, setServerOnline] = useState<boolean>(false);
  const [showGuide, setShowGuide] = useState<boolean>(false);
  const [actionNotice, setActionNotice] = useState<string>('');
  const [isActivating, setIsActivating] = useState<boolean>(false);
  const [isClearingLogs, setIsClearingLogs] = useState<boolean>(false);
  
  // AutoResearch State
  const [autoResearchData, setAutoResearchData] = useState<AutoResearchData | null>(null);
  const [autoExpCount, setAutoExpCount] = useState<number>(5);

  const fetchModels = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/models');
      if (res.ok) {
        const data = await res.json();
        const modelsList = data.models || [];
        setAvailableModels(modelsList);
        if (modelsList.length > 0 && !selectedModel) {
          setSelectedModel(modelsList[0].filename);
          setActiveModelConfirmed(data.active || modelsList[0].filename);
        }
      }
    } catch {
      // Fallback
    }
  };

  const fetchAutoResearchData = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/autoresearch/results');
      if (res.ok) {
        const data = await res.json();
        setAutoResearchData(data);
      }
    } catch {
      // Fallback
    }
  };

  useEffect(() => {
    let wasTraining = false;

    const fetchStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/status');
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
          setServerOnline(true);

          if (wasTraining && !data.is_training) {
            await fetchModels();
            await fetchAutoResearchData();
            setActionNotice("🎉 Opération terminée ! Modèles et registres d'expérience mis à jour.");
          }
          wasTraining = data.is_training;
        } else {
          setServerOnline(false);
        }
      } catch {
        setServerOnline(false);
      }
    };

    fetchStatus();
    fetchModels();
    fetchAutoResearchData();

    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          total_timesteps: timesteps,
          model_name: customModelName,
          learning_rate: learningRate,
          reward_detection: rewardDetection,
          reward_exploration: rewardExploration,
          penalty_time: penaltyTime,
          penalty_bingo: penaltyBingo,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.model_name) {
          setCustomModelName(data.model_name);
        }
        setStatus(prev => ({ ...prev, is_training: true, status_message: `Démarrage de l'entraînement ${data.model_name || customModelName}...` }));
      }
    } catch (err) {
      console.error("Erreur démarrage entraînement:", err);
    }
  };

  const handleStartAutoResearch = async () => {
    try {
      const res = await fetch(`http://localhost:8000/api/train/autoresearch?experiments=${autoExpCount}&timesteps=25000`, {
        method: 'POST',
      });
      if (res.ok) {
        setStatus(prev => ({ ...prev, is_training: true, status_message: `🤖 Recherche AutoResearch lancée (${autoExpCount} expériences)...` }));
      }
    } catch (err) {
      console.error("Erreur démarrage AutoResearch:", err);
    }
  };

  const handleStopTraining = async () => {
    try {
      await fetch('http://localhost:8000/api/train/stop', { method: 'POST' });
    } catch (err) {
      console.error("Erreur arrêt:", err);
    }
  };

  const handleClearTensorboardLogs = async () => {
    if (!window.confirm("Voulez-vous réinitialiser l'historique TensorBoard ? Les anciens graphiques seront effacés pour une parfaite lisibilité.")) {
      return;
    }
    setIsClearingLogs(true);
    try {
      const res = await fetch('http://localhost:8000/api/tensorboard/clear', { method: 'POST' });
      if (res.ok) {
        setActionNotice("🗑️ Logs TensorBoard effacés avec succès. Graphiques réinitialisés.");
      }
    } catch (err) {
      console.error('Erreur effacement TensorBoard:', err);
    } finally {
      setIsClearingLogs(false);
    }
  };

  const handleApplyChampionParamsForLongRun = () => {
    if (autoResearchData && autoResearchData.best_params) {
      const p = autoResearchData.best_params;
      setLearningRate(p.lr || 0.0003);
      setRewardDetection(p.r_det || 600.0);
      setRewardExploration(p.r_prob || 20.0);
      setPenaltyTime(p.p_time || 0.1);
      setPenaltyBingo(p.p_bingo || 1000.0);
      setTimesteps(250000);
      setCustomModelName('PPO_Champion_Long_250k');
      setActionNotice("🚀 Paramètres du Modèle Champion chargés ! Vous pouvez lancer l'entraînement long de 250 000 pas.");
    }
  };

  const handleApplyModelSelection = async (filename: string) => {
    setIsActivating(true);
    setActionNotice(`Activation du modèle ${filename}...`);
    try {
      const res = await fetch('http://localhost:8000/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_filename: filename }),
      });
      if (res.ok) {
        const data = await res.json();
        setActiveModelConfirmed(filename);
        setActionNotice(data.message || `✅ Modèle ${filename} activé avec succès dans la simulation !`);
        fetchModels();
      }
    } catch (err) {
      console.error('Erreur sélection modèle:', err);
      setActionNotice(`Erreur lors de l'activation de ${filename}`);
    } finally {
      setIsActivating(false);
    }
  };

  const championParams = autoResearchData?.best_params;

  return (
    <div className="space-y-5 text-slate-100">
      {/* Top Navigation & Status Header */}
      <div className="glass-panel rounded-xl p-4 border-l-4 border-purple-500 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            <span>Laboratoire de Recherche & Entraînement RL (PPO)</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Optimisez la politique de patrouille par recherche autonome (Karpathy Loop) ou par réglages manuels.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleClearTensorboardLogs}
            disabled={isClearingLogs || status.is_training}
            className="flex items-center space-x-1.5 bg-slate-800 hover:bg-rose-950/80 hover:border-rose-500/50 text-slate-300 hover:text-rose-300 text-xs px-3 py-1.5 rounded-lg border border-slate-700 font-bold transition-all cursor-pointer disabled:opacity-50"
            title="Effacer les anciens logs pour repartir de zéro dans TensorBoard"
          >
            <Trash2 className="w-3.5 h-3.5 text-rose-400" />
            <span>{isClearingLogs ? 'Purge...' : '🗑️ Purger Logs TensorBoard'}</span>
          </button>

          <a
            href="http://localhost:6006"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-1.5 bg-purple-950/80 border border-purple-500/50 text-purple-300 text-xs px-3 py-1.5 rounded-lg font-mono font-bold hover:bg-purple-900/80 transition-all"
          >
            <ExternalLink className="w-3.5 h-3.5 text-purple-400" />
            <span>TensorBoard (6006)</span>
          </a>

          {serverOnline ? (
            <span className="flex items-center space-x-1.5 bg-emerald-950/80 border border-emerald-500/50 text-emerald-300 text-xs px-3 py-1.5 rounded-lg font-mono font-bold">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>API Connectée</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1.5 bg-amber-950/80 border border-amber-500/50 text-amber-300 text-xs px-3 py-1.5 rounded-lg font-mono font-bold">
              <AlertCircle className="w-4 h-4 text-amber-400" />
              <span>python python/server.py</span>
            </span>
          )}
        </div>
      </div>

      {/* Live Training Progress Bar */}
      {status.is_training && (
        <div className="glass-panel rounded-xl p-4 border-l-4 border-cyan-500 space-y-2 animate-pulse">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-cyan-300 font-bold flex items-center space-x-2">
              <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
              <span>{status.status_message}</span>
            </span>
            <span className="text-cyan-400 font-bold">{status.progress_percent.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-800">
            <div
              className="bg-gradient-to-r from-purple-500 via-cyan-500 to-emerald-500 h-full transition-all duration-500"
              style={{ width: `${Math.max(5, status.progress_percent)}%` }}
            />
          </div>
        </div>
      )}

      {/* TIER 1: CHAMPION MODEL & OPTIMAL DISCOVERED HYPERPARAMETERS CARD */}
      <div className="glass-panel rounded-xl p-5 border-l-4 border-emerald-500 space-y-4 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <Trophy className="w-5 h-5 text-amber-400" />
            <span>Modèle Actif dans la Simulation & Paramètres Optimaux Découverts</span>
          </h3>

          {activeModelConfirmed && (
            <span className="bg-emerald-950/80 border border-emerald-500/50 text-emerald-300 text-xs px-3 py-1 rounded font-mono font-bold flex items-center space-x-1">
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span>Modèle Sélectionné : {activeModelConfirmed}</span>
            </span>
          )}
        </div>

        {/* Champion Hyperparameters Badges Grid */}
        {championParams ? (
          <div className="bg-slate-950/80 border border-amber-500/30 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-amber-400 flex items-center space-x-1.5">
                <Sparkles className="w-4 h-4 text-amber-400" />
                <span>Hyperparamètres Optimaux Identifiés par AutoResearch (Score: {autoResearchData?.best_score.toFixed(1)} pts) :</span>
              </span>

              <button
                onClick={handleApplyChampionParamsForLongRun}
                disabled={status.is_training}
                className="flex items-center space-x-1.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold py-1.5 px-3 rounded-lg text-xs transition-all shadow-md cursor-pointer disabled:opacity-50"
              >
                <Rocket className="w-3.5 h-3.5 text-white" />
                <span>🚀 Lancer un Entraînement Poussé avec ces Paramètres (250k)</span>
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 text-xs font-mono">
              <div className="bg-slate-900 border border-amber-500/40 p-2.5 rounded-lg text-center">
                <span className="text-slate-400 text-[10px] block uppercase">Learning Rate</span>
                <span className="text-sm font-bold text-amber-300">{championParams.lr}</span>
              </div>

              <div className="bg-slate-900 border border-emerald-500/40 p-2.5 rounded-lg text-center">
                <span className="text-slate-400 text-[10px] block uppercase">Bonus Détection</span>
                <span className="text-sm font-bold text-emerald-300">+{championParams.r_det}</span>
              </div>

              <div className="bg-slate-900 border border-cyan-500/40 p-2.5 rounded-lg text-center">
                <span className="text-slate-400 text-[10px] block uppercase">Exploration Bayésienne</span>
                <span className="text-sm font-bold text-cyan-300">+{championParams.r_prob} × Gain</span>
              </div>

              <div className="bg-slate-900 border border-rose-500/40 p-2.5 rounded-lg text-center">
                <span className="text-slate-400 text-[10px] block uppercase">Pénalité Bingo</span>
                <span className="text-sm font-bold text-rose-300">-{championParams.p_bingo}</span>
              </div>

              <div className="bg-slate-900 border border-blue-500/40 p-2.5 rounded-lg text-center">
                <span className="text-slate-400 text-[10px] block uppercase">Pénalité Temps</span>
                <span className="text-sm font-bold text-blue-300">-{championParams.p_time} / min</span>
              </div>

              <div className="bg-slate-900 border border-purple-500/40 p-2.5 rounded-lg text-center">
                <span className="text-slate-400 text-[10px] block uppercase">Gamma Escompte</span>
                <span className="text-sm font-bold text-purple-300">{championParams.gamma}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-lg text-xs text-slate-400 flex items-center space-x-2">
            <Info className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <span>Lancez la Recherche Autonome AutoResearch ci-dessous pour que le système identifie automatiquement les hyperparamètres optimaux.</span>
          </div>
        )}

        {/* Model Selection Dropdown */}
        <div className="space-y-2 text-xs">
          <label className="text-slate-300 font-bold block">
            Choisir le modèle RL actif dans les simulations Monte-Carlo :
          </label>
          <div className="flex items-center space-x-3">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2.5 text-emerald-300 font-mono font-bold cursor-pointer"
            >
              {availableModels.map((m) => (
                <option key={m.filename} value={m.filename}>
                  {m.title}
                </option>
              ))}
            </select>

            <button
              onClick={() => handleApplyModelSelection(selectedModel)}
              disabled={isActivating || !selectedModel}
              className="flex items-center space-x-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold py-2.5 px-4 rounded-lg shadow-md cursor-pointer transition-all disabled:opacity-50"
            >
              {isActivating ? (
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-white" />
              )}
              <span>Activer ce Modèle</span>
            </button>
          </div>
        </div>

        {actionNotice && (
          <div className="bg-emerald-950/90 border border-emerald-500/80 p-3 rounded-lg text-xs font-mono text-emerald-200 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <span>{actionNotice}</span>
            </div>

            {onNavigateToTactical && (
              <button
                onClick={onNavigateToTactical}
                className="bg-emerald-600 hover:bg-emerald-500 text-white px-3.5 py-1 rounded text-[11px] font-bold cursor-pointer transition-all"
              >
                🗺️ Voir sur la Carte Tactique
              </button>
            )}
          </div>
        )}
      </div>

      {/* TIER 2: AUTORESEARCH (KARPATHY LOOP) AUTONOMOUS SEARCH */}
      <div className="glass-panel rounded-xl p-5 border-l-4 border-amber-500 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2">
            <Bot className="w-4 h-4 text-amber-400" />
            <span>1. Recherche Autonome d'Hyperparamètres (Mode AutoResearch Karpathy)</span>
          </h3>

          {autoResearchData && autoResearchData.best_score > -900000 && (
            <span className="text-xs font-mono text-amber-400 font-bold">
              Meilleur Score : {autoResearchData.best_score.toFixed(1)} pts
            </span>
          )}
        </div>

        <p className="text-xs text-slate-300">
          Inspiré par le projet <strong>AutoResearch d'Andrej Karpathy</strong>, le système formule des hypothèses d'hyperparamètres (LR, Récompenses, Gamma), entraîne de courts modèles, évalue le taux d'interception et <u>conserve automatiquement le modèle Champion</u> (`PPO_CHAMPION_AUTORESEARCH`).
        </p>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <div className="flex items-center space-x-3 text-xs">
            <span className="text-slate-400 font-bold">Nombre d'Expérimentations Autonomes :</span>
            <select
              value={autoExpCount}
              onChange={(e) => setAutoExpCount(parseInt(e.target.value))}
              disabled={status.is_training}
              className="bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-amber-300 font-mono font-bold"
            >
              <option value={3}>3 Expérimentations (~3 min)</option>
              <option value={5}>5 Expérimentations (~5 min - Recommandé)</option>
              <option value={10}>10 Expérimentations (~10 min - Exploration Profonde)</option>
            </select>
          </div>

          <button
            onClick={handleStartAutoResearch}
            disabled={status.is_training || !serverOnline}
            className="flex items-center space-x-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white font-bold py-2 px-4 rounded-lg shadow-lg cursor-pointer transition-all disabled:opacity-50 text-xs"
          >
            <Flame className="w-4 h-4 fill-white" />
            <span>▶️ Lancer la Recherche Autonome AutoResearch</span>
          </button>
        </div>

        {/* Experiment History Ledger Table */}
        {autoResearchData && autoResearchData.history && autoResearchData.history.length > 0 && (
          <div className="pt-3 border-t border-slate-800/80 space-y-2">
            <h4 className="text-xs font-bold text-amber-400 flex items-center space-x-1">
              <span>📋 Registre des Expérimentations Autonomes (AutoResearch Ledger) :</span>
            </h4>
            <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1 font-mono text-[11px]">
              {autoResearchData.history.slice().reverse().map((exp, idx) => (
                <div
                  key={idx}
                  className={`p-2.5 rounded-lg border flex flex-wrap items-center justify-between gap-2 ${
                    exp.status === 'CHAMPION_KEPT'
                      ? 'bg-amber-950/70 border-amber-500/60 text-amber-200 font-bold'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-slate-500 font-bold">#{exp.experiment}</span>
                    <span>{exp.name}</span>
                    <span className="text-slate-500">[{exp.timestamp}]</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span>Récompense: {exp.mean_reward}</span>
                    <span>Succès: {exp.success_rate}%</span>
                    <span className="text-slate-400 text-[10px]">
                      (LR:{exp.hyperparams.lr}, Det:+{exp.hyperparams.r_det}, Prob:+{exp.hyperparams.r_prob})
                    </span>
                    <span className={exp.status === 'CHAMPION_KEPT' ? 'text-amber-400 font-bold' : 'text-slate-500'}>
                      {exp.status === 'CHAMPION_KEPT' ? '🏆 CHAMPION CONSERVÉ' : 'REJETÉ'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* TIER 3: MANUAL HYPERPARAMETER CONTROLS & LAUNCHER */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Panel 1: Hyperparameter Sliders */}
        <div className="glass-panel rounded-xl p-4 space-y-3">
          <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2 border-b border-slate-800 pb-2">
            <Sliders className="w-4 h-4 text-purple-400" />
            <span>2. Réglage Manuel des Hyperparamètres PPO</span>
          </h3>

          <div className="space-y-3 text-xs">
            <div>
              <div className="flex justify-between text-slate-300 font-bold mb-1">
                <span>Taux d'Apprentissage (Learning Rate PPO)</span>
                <span className="font-mono text-amber-400">{learningRate}</span>
              </div>
              <select
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                disabled={status.is_training}
                className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-amber-300 font-mono font-bold"
              >
                <option value={0.0001}>0.0001 (Apprentissage Lent & Stable)</option>
                <option value={0.0003}>0.0003 (Standard SB3 - Recommandé)</option>
                <option value={0.001}>0.001 (Apprentissage Rapide & Agressif)</option>
              </select>
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1 font-bold">
                <span>Bonus Détection Cible</span>
                <span className="font-mono text-emerald-400">+{rewardDetection} pts</span>
              </div>
              <input
                type="range"
                min={100}
                max={2000}
                step={50}
                value={rewardDetection}
                onChange={(e) => setRewardDetection(parseFloat(e.target.value))}
                disabled={status.is_training}
                className="w-full accent-emerald-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1 font-bold">
                <span>Bonus Prospection Bayésienne</span>
                <span className="font-mono text-cyan-400">+{rewardExploration} × Gain</span>
              </div>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={rewardExploration}
                onChange={(e) => setRewardExploration(parseFloat(e.target.value))}
                disabled={status.is_training}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1 font-bold">
                <span>Pénalité Temps de Vol</span>
                <span className="font-mono text-blue-400">-{penaltyTime} pt / min</span>
              </div>
              <input
                type="range"
                min={0.01}
                max={1.0}
                step={0.05}
                value={penaltyTime}
                onChange={(e) => setPenaltyTime(parseFloat(e.target.value))}
                disabled={status.is_training}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1 font-bold">
                <span>Pénalité Bingo Fuel Crash</span>
                <span className="font-mono text-rose-400">-{penaltyBingo} pts</span>
              </div>
              <input
                type="range"
                min={100}
                max={3000}
                step={100}
                value={penaltyBingo}
                onChange={(e) => setPenaltyBingo(parseFloat(e.target.value))}
                disabled={status.is_training}
                className="w-full accent-rose-500 cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Panel 2: Training Launcher */}
        <div className="glass-panel rounded-xl p-4 space-y-3">
          <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2 border-b border-slate-800 pb-2">
            <Play className="w-4 h-4 text-purple-400" />
            <span>Lancement Manuel d'un Run PPO_X</span>
          </h3>

          <div className="space-y-3 text-xs">
            <div>
              <label className="text-slate-400 block mb-1">Nom du Run (ex: PPO_8, PPO_Custom)</label>
              <input
                type="text"
                value={customModelName}
                onChange={(e) => setCustomModelName(e.target.value)}
                placeholder="PPO_8"
                disabled={status.is_training}
                className="w-full bg-slate-900 border border-slate-800 rounded px-3 py-2 text-purple-300 font-mono font-bold"
              />
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Pas de Temps (Timesteps)</label>
              <select
                value={timesteps}
                onChange={(e) => setTimesteps(parseInt(e.target.value))}
                disabled={status.is_training}
                className="w-full bg-slate-900 border border-slate-800 rounded px-3 py-2 text-slate-100 font-mono font-bold"
              >
                <option value={10000}>10 000 timesteps (~30 sec - Test Rapide)</option>
                <option value={50000}>50 000 timesteps (~2 min - Standard)</option>
                <option value={100000}>100 000 timesteps (~4 min - Entraînement Poussé)</option>
                <option value={250000}>250 000 timesteps (~10 min - Haute Précision)</option>
              </select>
            </div>

            <div className="pt-2 flex items-center space-x-3">
              {!status.is_training ? (
                <button
                  onClick={handleStartTraining}
                  disabled={!serverOnline}
                  className="flex-1 flex items-center justify-center space-x-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold py-2.5 px-4 rounded-lg shadow-lg cursor-pointer transition-all disabled:opacity-50"
                >
                  <Play className="w-4 h-4 fill-white" />
                  <span>Démarrer {customModelName || 'PPO'}</span>
                </button>
              ) : (
                <button
                  onClick={handleStopTraining}
                  className="flex-1 flex items-center justify-center space-x-2 bg-rose-600 hover:bg-rose-500 text-white font-bold py-2.5 px-4 rounded-lg shadow-lg cursor-pointer transition-all"
                >
                  <Square className="w-4 h-4 fill-white" />
                  <span>Interrompre l'Entraînement</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Guide Section */}
      <div className="glass-panel rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2">
            <Info className="w-4 h-4 text-cyan-400" />
            <span>Guide d'Interprétation TensorBoard</span>
          </h3>

          <button
            onClick={() => setShowGuide(!showGuide)}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold flex items-center space-x-1 cursor-pointer"
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>{showGuide ? 'Masquer le Guide' : 'Afficher le Guide'}</span>
          </button>
        </div>

        {showGuide && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="bg-slate-900/90 border border-emerald-500/30 p-3.5 rounded-lg space-y-1.5">
              <div className="font-bold text-emerald-400 text-sm flex items-center justify-between">
                <span>1. rollout/ep_rew_mean</span>
                <span className="text-[10px] bg-emerald-950 text-emerald-300 px-2 py-0.5 rounded border border-emerald-800">LE KPI N°1</span>
              </div>
              <p className="text-slate-300">
                Score accumulé par l'aéronef. Si le score stagne ou chute, augmentez le bonus Détection ou baissez la pénalité de temps.
              </p>
            </div>

            <div className="bg-slate-900/90 border border-cyan-500/30 p-3.5 rounded-lg space-y-1.5">
              <div className="font-bold text-cyan-400 text-sm flex items-center justify-between">
                <span>2. rollout/ep_len_mean</span>
                <span className="text-[10px] bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800">DURÉE VOL</span>
              </div>
              <p className="text-slate-300">
                Doit diminuer au fur et à mesure de l'entraînement, prouvant des interceptions plus rapides.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
