import React from 'react';
import './styles/forecastControls.css';

interface ForecastControlsProps {
  timeRange: 'semanal' | 'mensual' | 'anual';
  onTimeRangeChange: (range: 'semanal' | 'mensual' | 'anual') => void;
  activeBrand: string;
  onBrandChange: (brand: string) => void;
}

export const ForecastControls: React.FC<ForecastControlsProps> = ({
  timeRange,
  onTimeRangeChange,
  activeBrand,
  onBrandChange
}) => {
  return (
    <div className="forecast-controls">
      <div className="time-range-selector">
        <button 
          className={timeRange === 'semanal' ? 'active' : ''}
          onClick={() => onTimeRangeChange('semanal')}
        >
          Semanal
        </button>
        <button
          className={timeRange === 'mensual' ? 'active' : ''}
          onClick={() => onTimeRangeChange('mensual')}
        >
          Mensual
        </button>
        <button
          className={timeRange === 'anual' ? 'active' : ''}
          onClick={() => onTimeRangeChange('anual')}
        >
          Anual
        </button>
      </div>

      <div className="brand-selector">
        <select 
          value={activeBrand}
          onChange={(e) => onBrandChange(e.target.value)}
        >
          <option value="Mejora">Mejora</option>
          <option value="OtraMarca">Otra Marca</option>
        </select>
      </div>
    </div>
  );
};