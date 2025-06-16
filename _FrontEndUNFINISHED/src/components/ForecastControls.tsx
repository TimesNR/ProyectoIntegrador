import React from 'react';
import './ForecastControls.css';

interface ForecastControlsProps {
  selectedModel: 'ARIMA' | 'Geometric';
  onModelSelected: (model: 'ARIMA' | 'Geometric') => void;
}

export const ForecastControls: React.FC<ForecastControlsProps> = ({
  selectedModel,
  onModelSelected
}) => {
  return (
    <div className="model-controls">
      <button 
        className={`model-button ${selectedModel === 'ARIMA' ? 'active' : ''}`}
        onClick={() => onModelSelected('ARIMA')}
      >
        ARIMA
      </button>
      <button
        className={`model-button ${selectedModel === 'Geometric' ? 'active' : ''}`}
        onClick={() => onModelSelected('Geometric')}
      >
        Geometric Proy.
      </button>
    </div>
  );
};


// import React from 'react';
// import './ForecastControls.css';

// interface ForecastControlsProps {
//   selectedModel: 'ARIMA' | 'Geometric';
//   onModelSelected: (model: 'ARIMA' | 'Geometric') => void;
//   timeRange: 'semanal' | 'mensual' | 'anual'; // <-- Agregado
//   onTimeRangeChange: (range: 'semanal' | 'mensual' | 'anual') => void; // <-- Agregado si lo necesitas
//   // activeBrand: string; // <-- También agrégalo si se usa
//   // onBrandChange: (brand: string) => void; // <-- También agrégalo si se usa
// }

// export const ForecastControls: React.FC<ForecastControlsProps> = ({
//   selectedModel,
//   onModelSelected
// }) => {
//   return (
//     <div className="model-controls">
//       <button 
//         className={`model-button ${selectedModel === 'ARIMA' ? 'active' : ''}`}
//         onClick={() => onModelSelected('ARIMA')}
//       >
//         ARIMA
//       </button>
//       <button
//         className={`model-button ${selectedModel === 'Geometric' ? 'active' : ''}`}
//         onClick={() => onModelSelected('Geometric')}
//       >
//         Geometric Brownian
//       </button>
//     </div>
//   );
// };