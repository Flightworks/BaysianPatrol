import React from 'react';
import type { ScenarioConfig } from '../types/simulation';
import { Sliders, Wind, Navigation, Fuel, Cpu, Anchor } from 'lucide-react';

interface ControlPanelProps {
  config: ScenarioConfig;
  onChange: (newConfig: ScenarioConfig) => void;
  disabled: boolean;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({ config, onChange, disabled }) => {
  const updateField = <K extends keyof ScenarioConfig>(field: K, value: ScenarioConfig[K]) => {
    onChange({ ...config, [field]: value });
  };

  return (
    <div className="glass-panel rounded-xl p-4 space-y-5 text-xs">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h2 className="text-sm font-bold text-white flex items-center space-x-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span>Configuration de la Simulation Stochastique</span>
        </h2>
      </div>

      {/* Section 1: Target Kinematics & Datum Uncertainty */}
      <div className="space-y-3">
        <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
          <Navigation className="w-3.5 h-3.5" />
          <span>Incertitude Datum & Cinématique Cible</span>
        </h3>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-slate-400 block mb-1">Cap Moyen Cible (°)</label>
            <input
              type="number"
              value={config.meanHeading}
              onChange={(e) => updateField('meanHeading', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Vitesse Moyenne (kts)</label>
            <input
              type="number"
              value={config.meanSpeed}
              onChange={(e) => updateField('meanSpeed', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Incertitude Spatiale Datum (±NM)</label>
            <input
              type="number"
              value={config.sigmaDatumX}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 1;
                onChange({ ...config, sigmaDatumX: val, sigmaDatumY: val });
              }}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono text-amber-400 font-bold"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Incertitude Temps T₀ (min)</label>
            <input
              type="number"
              value={config.sigmaT}
              onChange={(e) => updateField('sigmaT', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>
        </div>
      </div>

      {/* Section 2: Frigate Base & Helicopter Departure Variability */}
      <div className="space-y-3 pt-2 border-t border-slate-800/80">
        <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
          <Anchor className="w-3.5 h-3.5 text-sky-400" />
          <span>Variabilité Frégate & Hélicoptère par Run</span>
        </h3>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-slate-400 block mb-1">Position Frégate Nominal X/Y (NM)</label>
            <input
              type="number"
              value={config.frigateX}
              onChange={(e) => updateField('frigateX', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Var. Pos. Frégate par Run (±NM)</label>
            <input
              type="number"
              value={config.sigmaFrigatePosition || 3}
              onChange={(e) => updateField('sigmaFrigatePosition', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono text-sky-400 font-bold"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Vitesse Hélico Moyenne (kt)</label>
            <input
              type="number"
              value={config.helicoMaxSpeed}
              onChange={(e) => updateField('helicoMaxSpeed', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Var. Vitesse Hélico (±kt)</label>
            <input
              type="number"
              value={config.sigmaHelicoSpeed || 5}
              onChange={(e) => updateField('sigmaHelicoSpeed', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono text-sky-400 font-bold"
            />
          </div>
        </div>
      </div>

      {/* Section 3: Environmental Variability Per Run */}
      <div className="space-y-3 pt-2 border-t border-slate-800/80">
        <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
          <Wind className="w-3.5 h-3.5" />
          <span>Variabilité Météo par Run</span>
        </h3>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-slate-400 block mb-1">Vitesse Vent Moyenne (kt)</label>
            <input
              type="number"
              value={config.windSpeed}
              onChange={(e) => updateField('windSpeed', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Variation Vent par Run (±kt)</label>
            <input
              type="number"
              value={config.sigmaWindSpeed || 4}
              onChange={(e) => updateField('sigmaWindSpeed', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono text-cyan-400 font-bold"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Direction Vent Moyenne (°)</label>
            <input
              type="number"
              value={config.windDirection}
              onChange={(e) => updateField('windDirection', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Variation Dir. par Run (±°)</label>
            <input
              type="number"
              value={config.sigmaWindDirection || 15}
              onChange={(e) => updateField('sigmaWindDirection', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono text-cyan-400 font-bold"
            />
          </div>
        </div>
      </div>

      {/* Section 4: Radar Specs & Endurance */}
      <div className="space-y-3 pt-2 border-t border-slate-800/80">
        <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
          <Fuel className="w-3.5 h-3.5" />
          <span>Radar & Réserve Autonomie</span>
        </h3>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-slate-400 block mb-1">Portée Radar R₀ (NM)</label>
            <input
              type="number"
              value={config.radarBaseRange}
              onChange={(e) => updateField('radarBaseRange', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Autonomie Vol (min)</label>
            <input
              type="number"
              value={config.helicoEndurance}
              onChange={(e) => updateField('helicoEndurance', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono"
            />
          </div>
        </div>
      </div>

      {/* Section 5: Monte-Carlo Controls */}
      <div className="space-y-3 pt-2 border-t border-slate-800/80">
        <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
          <Cpu className="w-3.5 h-3.5" />
          <span>Contrôles Monte-Carlo</span>
        </h3>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-slate-400 block mb-1">Nombre de Runs N</label>
            <input
              type="number"
              value={config.numIterations}
              onChange={(e) => updateField('numIterations', parseInt(e.target.value) || 100)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-100 font-mono font-bold text-cyan-400"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Comparaison Stratégie</label>
            <select
              value={config.strategy}
              onChange={(e) => updateField('strategy', e.target.value as any)}
              disabled={disabled}
              className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-slate-100 font-bold cursor-pointer"
            >
              <option value="BOTH">SIGMA vs NAÏVE (Apparié)</option>
              <option value="SIGMA">Seul SIGMA (Bayésien)</option>
              <option value="NAIVE">Seule Râteau Naïf</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};
