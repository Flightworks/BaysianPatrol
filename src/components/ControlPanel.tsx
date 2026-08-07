import React from 'react';
import type { ScenarioConfig } from '../types/simulation';

interface ControlPanelProps {
  config: ScenarioConfig;
  onChange: (newConfig: ScenarioConfig) => void;
  disabled: boolean;
}

interface NumberFieldProps {
  label: string;
  value: number;
  unit: string;
  disabled: boolean;
  onChange: (value: number) => void;
}

const NumberField: React.FC<NumberFieldProps> = ({ label, value, unit, disabled, onChange }) => (
  <label className="parameter-field">
    <span>{label}</span>
    <div><input type="number" value={value} disabled={disabled} onChange={(event) => onChange(Number(event.target.value) || 0)} /><small>{unit}</small></div>
  </label>
);

export const ControlPanel: React.FC<ControlPanelProps> = ({ config, onChange, disabled }) => {
  const update = <K extends keyof ScenarioConfig>(key: K, value: ScenarioConfig[K]) => onChange({ ...config, [key]: value });

  return (
    <div className="parameter-grid">
      <fieldset>
        <legend>Cible et datum</legend>
        <NumberField label="Cap cible moyen" value={config.meanHeading} unit="°" disabled={disabled} onChange={(v) => update('meanHeading', v)} />
        <NumberField label="Vitesse cible" value={config.meanSpeed} unit="kt" disabled={disabled} onChange={(v) => update('meanSpeed', v)} />
        <NumberField label="Incertitude datum" value={config.sigmaDatumX} unit="± NM" disabled={disabled} onChange={(v) => onChange({ ...config, sigmaDatumX: v, sigmaDatumY: v })} />
        <NumberField label="Incertitude T₀" value={config.sigmaT} unit="min" disabled={disabled} onChange={(v) => update('sigmaT', v)} />
      </fieldset>

      <fieldset>
        <legend>Frégate et aéronef</legend>
        <NumberField label="Position frégate X" value={config.frigateX} unit="NM" disabled={disabled} onChange={(v) => update('frigateX', v)} />
        <NumberField label="Position frégate Y" value={config.frigateY} unit="NM" disabled={disabled} onChange={(v) => update('frigateY', v)} />
        <NumberField label="Dispersion frégate" value={config.sigmaFrigatePosition} unit="± NM" disabled={disabled} onChange={(v) => update('sigmaFrigatePosition', v)} />
        <NumberField label="Vitesse hélicoptère" value={config.helicoMaxSpeed} unit="kt" disabled={disabled} onChange={(v) => update('helicoMaxSpeed', v)} />
        <NumberField label="Autonomie" value={config.helicoEndurance} unit="min" disabled={disabled} onChange={(v) => update('helicoEndurance', v)} />
        <NumberField label="Réserve Bingo" value={config.bingoFuelBuffer} unit="min" disabled={disabled} onChange={(v) => update('bingoFuelBuffer', v)} />
      </fieldset>

      <fieldset>
        <legend>Environnement</legend>
        <NumberField label="Vent moyen" value={config.windSpeed} unit="kt" disabled={disabled} onChange={(v) => update('windSpeed', v)} />
        <NumberField label="Direction du vent" value={config.windDirection} unit="°" disabled={disabled} onChange={(v) => update('windDirection', v)} />
        <NumberField label="Variation vitesse" value={config.sigmaWindSpeed} unit="± kt" disabled={disabled} onChange={(v) => update('sigmaWindSpeed', v)} />
        <NumberField label="Variation direction" value={config.sigmaWindDirection} unit="± °" disabled={disabled} onChange={(v) => update('sigmaWindDirection', v)} />
      </fieldset>

      <fieldset>
        <legend>Capteur et zone</legend>
        <NumberField label="Portée radar" value={config.radarBaseRange} unit="NM" disabled={disabled} onChange={(v) => update('radarBaseRange', v)} />
        <NumberField label="Largeur de zone" value={config.searchAreaWidth} unit="NM" disabled={disabled} onChange={(v) => update('searchAreaWidth', v)} />
        <NumberField label="Hauteur de zone" value={config.searchAreaHeight} unit="NM" disabled={disabled} onChange={(v) => update('searchAreaHeight', v)} />
        <NumberField label="Maille de calcul" value={config.gridCellSize} unit="NM" disabled={disabled} onChange={(v) => update('gridCellSize', v)} />
      </fieldset>
    </div>
  );
};
