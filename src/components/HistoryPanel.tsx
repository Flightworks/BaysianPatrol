import React, { useState } from 'react';
import { Trash2 } from 'lucide-react';
import type { RunHistoryEntry } from '../engine/runHistory';

interface HistoryPanelProps {
  history: RunHistoryEntry[];
  onClear: () => void;
}

const formatDate = (iso: string) => new Intl.DateTimeFormat('fr-FR', {
  dateStyle: 'short',
  timeStyle: 'short',
}).format(new Date(iso));

const rate = (value?: number) => value === undefined ? '—' : `${value.toFixed(1)} %`;
const minutes = (value?: number) => value === undefined ? '—' : `${value.toFixed(1)} min`;

export const HistoryPanel: React.FC<HistoryPanelProps> = ({ history, onClear }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <section className="history-workspace">
      <header className="section-heading history-heading">
        <div>
          <p className="eyebrow">Mémoire locale</p>
          <h1>Historique des campagnes</h1>
          <p>Les vingt dernières comparaisons sont conservées sur cet appareil, sans stocker les trajectoires volumineuses.</p>
        </div>
        <button
          className="secondary-button danger-button"
          disabled={history.length === 0}
          onClick={() => {
            if (window.confirm('Effacer tout l’historique des campagnes sur cet appareil ?')) onClear();
          }}
        >
          <Trash2 size={15} /> Effacer l’historique
        </button>
      </header>

      {history.length === 0 ? (
        <section className="empty-comparison"><span>03</span><div><h2>Aucune campagne conservée</h2><p>Le prochain calcul apparaîtra ici automatiquement.</p></div></section>
      ) : (
        <div className="history-table-wrap">
          <table className="history-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Scénario</th>
                <th>Tirages</th>
                <th>Hybride</th>
                <th>Bayésien</th>
                <th>Râteau IAMSAR</th>
                <th>Sécurité hybride</th>
              </tr>
            </thead>
            <tbody>
              {history.map((entry) => (
                <React.Fragment key={entry.id}>
                  <tr className="history-row" onClick={() => setExpandedId(expandedId === entry.id ? null : entry.id)}>
                    <td>{formatDate(entry.createdAt)}</td>
                    <td><strong>{entry.scenarioName}</strong><small>seed {entry.config.monteCarloSeed ?? 2026}</small></td>
                    <td>{entry.config.numIterations}</td>
                    <td className="strategy-hybrid"><strong>{rate(entry.strategies.hybrid?.successRate)}</strong><small>{minutes(entry.strategies.hybrid?.meanInterceptionTime)}</small></td>
                    <td><strong>{rate(entry.strategies.bayesian?.successRate)}</strong><small>{minutes(entry.strategies.bayesian?.meanInterceptionTime)}</small></td>
                    <td><strong>{rate(entry.strategies.naive?.successRate)}</strong><small>{minutes(entry.strategies.naive?.meanInterceptionTime)}</small></td>
                    <td>{rate(entry.strategies.hybrid?.bingoRate)} Bingo</td>
                  </tr>
                  {expandedId === entry.id && (
                    <tr className="history-details-row">
                      <td colSpan={7}>
                        <dl className="history-parameters">
                          <div><dt>Zone</dt><dd>{entry.config.searchAreaWidth} × {entry.config.searchAreaHeight} NM</dd></div>
                          <div><dt>Datum</dt><dd>±{entry.config.sigmaDatumX} NM</dd></div>
                          <div><dt>Cible</dt><dd>{entry.config.meanHeading}° · {entry.config.meanSpeed} kt</dd></div>
                          <div><dt>Vent</dt><dd>{entry.config.windDirection}° · {entry.config.windSpeed} kt</dd></div>
                          <div><dt>Radar</dt><dd>{entry.config.radarBaseRange} NM</dd></div>
                          <div><dt>Autonomie</dt><dd>{entry.config.helicoEndurance} min</dd></div>
                          <div><dt>Calcul</dt><dd>{(entry.executionTimeMs / 1000).toFixed(1)} s</dd></div>
                        </dl>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};
