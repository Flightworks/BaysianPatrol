import React from 'react';
import type { GlobalSimulationResult } from '../types/simulation';
import { Trophy, Clock, Fuel, ShieldAlert, Zap, TrendingUp } from 'lucide-react';

interface StatsDashboardProps {
  result: GlobalSimulationResult;
}

export const StatsDashboard: React.FC<StatsDashboardProps> = ({ result }) => {
  const { sigmaStats, naiveStats, rlStats, pairedStats, trioStats, executionTimeMs, config } = result;

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-wrap items-center justify-between gap-4 glass-panel rounded-xl p-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <Trophy className="w-5 h-5 text-amber-400" />
            <span>Résultats de l'Expérimentation Monte-Carlo</span>
          </h2>
          <p className="text-xs text-slate-400">
            {config.numIterations} tirages stochastiques sous variabilité environnementale (Vent ±{config.sigmaWindSpeed || 4}kt, Direction ±{config.sigmaWindDirection || 15}°) | Calcul exécuté en {executionTimeMs.toFixed(0)} ms
          </p>
        </div>

        {trioStats && (
          <div className="flex items-center space-x-3 bg-slate-900/90 border border-purple-500/40 rounded-lg px-4 py-2 text-xs">
            <Zap className="w-4 h-4 text-purple-400 animate-pulse" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-bold">Meilleure Stratégie du Trio :</span>
              <span className="text-purple-300 font-extrabold text-sm">
                {trioStats.bestStrategy === 'RL_MODEL' ? '🤖 Modèle RL / PPO' : trioStats.bestStrategy === 'SIGMA' ? '⚡ Algorithme SIGMA' : '📐 Râteau Naïf'}
              </span>
            </div>
          </div>
        )}

        {!trioStats && pairedStats && (
          <div className="flex items-center space-x-3 bg-slate-900/90 border border-emerald-500/40 rounded-lg px-4 py-2 text-xs">
            <Zap className="w-4 h-4 text-emerald-400 animate-pulse" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-bold">Domination Algorithme SIGMA:</span>
              <span className="text-emerald-300 font-extrabold text-sm">
                SIGMA gagne dans {pairedStats.sigmaWinRate.toFixed(1)}% des scénarios
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Trio Comparison KPI Grid */}
      {trioStats && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="glass-panel rounded-xl p-4 border-l-4 border-amber-500">
            <div className="text-xs text-slate-400 font-semibold">1. Râteau Naïf (IAMSAR)</div>
            <div className="text-2xl font-black text-amber-300 mt-1">{trioStats.naiveSuccessRate.toFixed(1)}%</div>
            <div className="text-xs text-slate-400 mt-1">Victoires directes : {trioStats.naiveWins} runs</div>
          </div>

          <div className="glass-panel rounded-xl p-4 border-l-4 border-emerald-500">
            <div className="text-xs text-slate-400 font-semibold">2. Algorithme Bayésien (POMDP)</div>
            <div className="text-2xl font-black text-emerald-300 mt-1">{trioStats.sigmaSuccessRate.toFixed(1)}%</div>
            <div className="text-xs text-slate-400 mt-1">Victoires directes : {trioStats.sigmaWins} runs</div>
          </div>

          <div className="glass-panel rounded-xl p-4 border-l-4 border-purple-500">
            <div className="text-xs text-slate-400 font-semibold">3. Agent RL SIGMA (Neurones)</div>
            <div className="text-2xl font-black text-purple-300 mt-1">{trioStats.rlSuccessRate.toFixed(1)}%</div>
            <div className="text-xs text-slate-400 mt-1">Victoires directes : {trioStats.rlWins} runs</div>
          </div>
        </div>
      )}

      {/* Paired Comparison KPI Grid */}
      {!trioStats && pairedStats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="glass-panel rounded-xl p-4 border-l-4 border-emerald-500">
            <div className="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span>Victoires Algorithme POMDP</span>
              <Trophy className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-black text-white mt-1">
              {pairedStats.sigmaWins} <span className="text-xs font-normal text-slate-400">/ {config.numIterations} runs</span>
            </div>
            <div className="text-xs text-emerald-400 font-bold mt-1">
              Taux de succès relatif : {pairedStats.sigmaWinRate.toFixed(1)}%
            </div>
          </div>

          <div className="glass-panel rounded-xl p-4 border-l-4 border-cyan-500">
            <div className="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span>Temps Moyen Gagné</span>
              <Clock className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-2xl font-black text-white mt-1">
              {pairedStats.meanTimeSavedMinutes > 0 ? `+${pairedStats.meanTimeSavedMinutes.toFixed(1)}` : pairedStats.meanTimeSavedMinutes.toFixed(1)}{' '}
              <span className="text-xs font-normal text-slate-400">min</span>
            </div>
            <div className="text-xs text-cyan-400 font-bold mt-1">
              Interception plus rapide par rapport au râteau classique
            </div>
          </div>

          <div className="glass-panel rounded-xl p-4 border-l-4 border-blue-500">
            <div className="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span>Économie de Carburant</span>
              <Fuel className="w-4 h-4 text-blue-400" />
            </div>
            <div className="text-2xl font-black text-white mt-1">
              {pairedStats.meanFuelSavedMinutes > 0 ? `+${pairedStats.meanFuelSavedMinutes.toFixed(1)}` : pairedStats.meanFuelSavedMinutes.toFixed(1)}{' '}
              <span className="text-xs font-normal text-slate-400">min de vol</span>
            </div>
            <div className="text-xs text-blue-400 font-bold mt-1">
              Marge de sécurité supplémentaire avant Bingo Fuel
            </div>
          </div>

          <div className="glass-panel rounded-xl p-4 border-l-4 border-rose-500">
            <div className="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span>Échecs Bingo Fuel</span>
              <ShieldAlert className="w-4 h-4 text-rose-400" />
            </div>
            <div className="text-xl font-bold text-white mt-1">
              POMDP: <span className="text-emerald-400">{sigmaStats?.bingoRate.toFixed(1) || 0}%</span> | Naïf: <span className="text-rose-400">{naiveStats?.bingoRate.toFixed(1) || 0}%</span>
            </div>
            <div className="text-xs text-rose-400 font-bold mt-1">
              Réduction nette des retours bredouilles sur panne d'autonomie
            </div>
          </div>
        </div>
      )}

      {/* Comparison Table */}
      <div className="glass-panel rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <span>Tableau Comparatif Global</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] font-mono">
                <th className="py-2.5 px-3">Stratégie de Recherche</th>
                <th className="py-2.5 px-3">Taux Détection (%)</th>
                <th className="py-2.5 px-3">Temps Moyen (min)</th>
                <th className="py-2.5 px-3">Conso. Carburant (min)</th>
                <th className="py-2.5 px-3">Retours sûrs (%)</th>
                <th className="py-2.5 px-3">Violations Bingo (%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {rlStats && (
                <tr className="hover:bg-slate-800/40 text-purple-300 font-semibold">
                  <td className="py-3 px-3 flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-purple-500 inline-block" />
                    <span>Agent RL SIGMA (PPO / ONNX)</span>
                  </td>
                  <td className="py-3 px-3 text-sm font-bold text-purple-400">
                    {rlStats.successRate.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3">{rlStats.meanInterceptionTime.toFixed(1)} min</td>
                  <td className="py-3 px-3">{rlStats.meanFuelConsumed.toFixed(1)} min</td>
                  <td className="py-3 px-3">{rlStats.safeReturnRate.toFixed(1)}%</td>
                  <td className="py-3 px-3">{rlStats.bingoRate.toFixed(1)}%</td>
                </tr>
              )}

              {sigmaStats && (
                <tr className="hover:bg-slate-800/40 text-emerald-300 font-semibold">
                  <td className="py-3 px-3 flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
                    <span>Algorithme Bayésien POMDP</span>
                  </td>
                  <td className="py-3 px-3 text-sm font-bold text-emerald-400">
                    {sigmaStats.successRate.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3">{sigmaStats.meanInterceptionTime.toFixed(1)} min</td>
                  <td className="py-3 px-3">{sigmaStats.meanFuelConsumed.toFixed(1)} min</td>
                  <td className="py-3 px-3">{sigmaStats.safeReturnRate.toFixed(1)}%</td>
                  <td className="py-3 px-3">{sigmaStats.bingoRate.toFixed(1)}%</td>
                </tr>
              )}

              {naiveStats && (
                <tr className="hover:bg-slate-800/40 text-amber-300 font-semibold">
                  <td className="py-3 px-3 flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
                    <span>Râteau Classique Naïf (Base)</span>
                  </td>
                  <td className="py-3 px-3 text-sm font-bold text-amber-400">
                    {naiveStats.successRate.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3">{naiveStats.meanInterceptionTime.toFixed(1)} min</td>
                  <td className="py-3 px-3">{naiveStats.meanFuelConsumed.toFixed(1)} min</td>
                  <td className="py-3 px-3">{naiveStats.safeReturnRate.toFixed(1)}%</td>
                  <td className="py-3 px-3">{naiveStats.bingoRate.toFixed(1)}%</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
