import React from 'react';
import type { GlobalSimulationResult, StrategyStats } from '../types/simulation';

interface StatsDashboardProps {
  result: GlobalSimulationResult;
}

interface StrategyColumn {
  key: string;
  name: string;
  detail: string;
  stats?: StrategyStats;
  featured?: boolean;
}

const value = (stats: StrategyStats | undefined, field: keyof StrategyStats, suffix: string) => {
  const raw = stats?.[field];
  return typeof raw === 'number' ? `${raw.toFixed(1)}${suffix}` : '—';
};

export const StatsDashboard: React.FC<StatsDashboardProps> = ({ result }) => {
  const columns: StrategyColumn[] = [
    { key: 'hybrid', name: 'Stratégie hybride', detail: 'Politique qualifiée 2027', stats: result.rlStats, featured: true },
    { key: 'bayesian', name: 'Recherche bayésienne', detail: 'Planification POMDP', stats: result.sigmaStats },
    { key: 'naive', name: 'Balayage parallèle', detail: 'Référence IAMSAR', stats: result.naiveStats },
  ];
  const hybrid = result.rlStats;
  const naive = result.naiveStats;
  const timeSaved = hybrid && naive ? naive.meanInterceptionTime - hybrid.meanInterceptionTime : 0;
  const sameOrBetterDetection = hybrid && naive ? hybrid.successRate >= naive.successRate : false;

  return (
    <section className="results-section">
      <header className="results-header">
        <div>
          <p className="eyebrow">Résultat de campagne</p>
          <h2>{result.config.numIterations} situation{result.config.numIterations > 1 ? 's' : ''} appariée{result.config.numIterations > 1 ? 's' : ''}</h2>
          <p>Calcul réalisé en {(result.executionTimeMs / 1000).toFixed(1)} s · seed {result.config.monteCarloSeed ?? 2026}</p>
        </div>
        {hybrid && (
          <div className="campaign-verdict">
            <span>{sameOrBetterDetection && timeSaved > 0 ? 'Avantage hybride' : 'Résultat à examiner'}</span>
            <strong>{timeSaved > 0 ? `${timeSaved.toFixed(1)} min gagnées` : `${hybrid.successRate.toFixed(1)} % détectés`}</strong>
            <small>par rapport au balayage IAMSAR</small>
          </div>
        )}
      </header>

      <div className="comparison-matrix" role="table" aria-label="Comparaison des trois stratégies">
        <div className="matrix-row matrix-head" role="row">
          <div role="columnheader">Indicateur</div>
          {columns.map((column) => (
            <div key={column.key} className={column.featured ? 'featured' : ''} role="columnheader">
              <strong>{column.name}</strong><span>{column.detail}</span>
            </div>
          ))}
        </div>
        {[
          ['Détection', 'successRate', ' %'],
          ['Temps moyen d’interception', 'meanInterceptionTime', ' min'],
          ['Carburant consommé', 'meanFuelConsumed', ' min'],
          ['Retours sûrs', 'safeReturnRate', ' %'],
          ['Violations Bingo', 'bingoRate', ' %'],
        ].map(([label, field, suffix]) => (
          <div className="matrix-row" role="row" key={field}>
            <div role="rowheader">{label}</div>
            {columns.map((column) => (
              <div key={column.key} className={column.featured ? 'featured matrix-value' : 'matrix-value'} role="cell">
                {value(column.stats, field as keyof StrategyStats, suffix)}
              </div>
            ))}
          </div>
        ))}
      </div>

      <footer className="results-footnote">
        <span>Lecture</span>
        <p>Une détection élevée ne suffit pas : la stratégie doit aussi revenir sans violation carburant et réduire le temps d’exposition.</p>
      </footer>
    </section>
  );
};
