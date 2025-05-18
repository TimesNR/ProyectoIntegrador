// import React, { useState } from 'react';
// import { ForecastChart } from '../components/ForecastChart';
// import { MetricCard } from '../components/MetricCard';
// import { ForecastControls } from '../components/ForecastControls';

// interface ForecastData {
//   semanal: {
//     labels: string[];
//     datasets: {
//       label: string;
//       data: number[];
//       backgroundColor: string;
//     }[];
//   };
//   mensual: {
//     labels: string[];
//     datasets: {
//       label: string;
//       data: number[];
//       backgroundColor: string;
//     }[];
//   };
//   anual: {
//     labels: string[];
//     datasets: {
//       label: string;
//       data: number[];
//       backgroundColor: string;
//     }[];
//   };
// }

// const forecastData: ForecastData = {
//   semanal: {
//     labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie'],
//     datasets: [
//       {
//         label: 'Visitas',
//         data: [10, 20, 15, 30, 25],
//         backgroundColor: 'rgba(75,192,192,0.4)',
//       },
//     ],
//   },
//   mensual: {
//     labels: ['Semana 1', 'Semana 2', 'Semana 3', 'Semana 4'],
//     datasets: [
//       {
//         label: 'Visitas',
//         data: [100, 150, 120, 180],
//         backgroundColor: 'rgba(153,102,255,0.6)',
//       },
//     ],
//   },
//   anual: {
//     labels: ['Ene', 'Feb', 'Mar', 'Abr'],
//     datasets: [
//       {
//         label: 'Visitas',
//         data: [400, 420, 500, 470],
//         backgroundColor: 'rgba(255,159,64,0.6)',
//       },
//     ],
//   },
// };

// const ForecastPage = () => {
//   const [timeRange, setTimeRange] = useState<'semanal' | 'mensual' | 'anual'>('semanal');
//   const [modelSelected, setModelSelected] = useState<'ARIMA' | 'Geometric'>('ARIMA');
//   // const [activeBrand, setActiveBrand] = useState<string>('Mejora');

//   const handleTimeRangeChange = (range: 'semanal' | 'mensual' | 'anual') => {
//     setTimeRange(range);
//   };

//   const handleModelSelected = (range: "ARIMA" | "Geometric") =>{
//     setModelSelected(range);
//   };

//   // const handleBrandChange = (brand: string) => {
//   //   setActiveBrand(brand);
//   // };

//   return (
//     <div>
//       <h1>Forecast Page</h1>
//       <ForecastControls 
//         timeRange={timeRange}
//         onTimeRangeChange={handleTimeRangeChange}
//         selectedModel={modelSelected}
//         onModelSelected = {handleModelSelected}
//         // activeBrand={activeBrand}
//         // onBrandChange={handleBrandChange}
//       />
//       <ForecastChart data={forecastData[timeRange]} />
//       <MetricCard title="Resumen" value="1234" />
//     </div>
//   );
// };

// export default ForecastPage;
import React, { useState } from 'react';
import { ForecastChart } from '../components/ForecastChart';
import { ForecastControls } from '../components/ForecastControls';
import MetricsPanel from '../components/MetricsPanel';
import { Metric } from '../components/MetricsPanel';
import './ForecastPage.css';
// import forecastImage from '../assets/team/compPLOT.jpg';
import forecastImage from '../assets/compPLOT.jpg';


const ForecastPage = () => {
  const [currentModel, setCurrentModel] = useState<'ARIMA' | 'Geometric'>('ARIMA');
  const [selectedMaterial, setSelectedMaterial] = useState<'black' | 'cardjolote' | 'ocean-plastic'>('black');

  const getDataByMaterial = (material: string, model: string) => {
    if (model === 'ARIMA') {
      switch(material) {
        case 'black': return [400, 420, 500, 470, 490, 520];
        case 'cardjolote': return [350, 380, 420, 400, 430, 450];
        case 'ocean-plastic': return [300, 320, 350, 340, 360, 380];
        default: return [];
      }
    } else {
      switch(material) {
        case 'black': return [380, 410, 490, 450, 480, 510];
        case 'cardjolote': return [330, 360, 400, 380, 410, 430];
        case 'ocean-plastic': return [280, 300, 330, 320, 340, 360];
        default: return [];
      }
    }
  };

  const getBackgroundColor = (material: string) => {
    switch(material) {
      case 'black': return 'rgba(0, 0, 0, 0.6)';
      case 'cardjolote': return 'rgba(139, 69, 19, 0.6)';
      case 'ocean-plastic': return 'rgba(0, 100, 150, 0.6)';
      default: return 'rgb(255, 0, 0)';
    }
  };

  const getBorderColor = (material: string) => {
    switch(material) {
      case 'black': return '#000000';
      case 'cardjolote': return '#8B4513';
      case 'ocean-plastic': return '#006496';
      default: return '#4BC0C0';
    }
  };

  const getMetricsData = (material: string, model: string): Metric[] => {
    const baseMetrics = [
      { 
        name: "Precisión", 
        value: model === 'ARIMA' ? 0.8923 : 0.8567, 
        improvement: model === 'ARIMA',
        format: 'percentage' as const
      },
      { 
        name: "Error (MSE)", 
        value: model === 'ARIMA' ? 0.045 : 0.051, 
        improvement: model === 'ARIMA',
        format: 'decimal' as const
      },
      {
        name: "Tiempo ejecución",
        value: model === 'ARIMA' ? "2.4s" : "1.8s"
      }
    ];

    return baseMetrics.map(metric => {
      if (metric.name === "Precisión") {
        return {
          ...metric,
          value: (metric.value as number) + (material === 'black' ? 0.02 : material === 'cardjolote' ? 0.01 : 0)
        };
      }
      return metric;
    });
  };

  const forecastData = {
    labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun'],
    datasets: [
      {
        label: `Modelo ${currentModel} - ${selectedMaterial}`,
        data: getDataByMaterial(selectedMaterial, currentModel),
        backgroundColor: getBackgroundColor(selectedMaterial),
        borderColor: getBorderColor(selectedMaterial),
        borderWidth: 2,
      },
    ],
  };

  const mockMetrics = getMetricsData(selectedMaterial, currentModel);

  return (
    <div className="forecast-page">
      <header className="page-header">
        <h1 className="page-title">Análisis Predictivo</h1>
      </header>

      <div className="controls-section">
        <ForecastControls 
          selectedModel={currentModel}
          onModelSelected={setCurrentModel}
        />
        
        <div className="material-selector">
          <label>Material:</label>
          <select 
            value={selectedMaterial}
            onChange={(e) => setSelectedMaterial(e.target.value as any)}
          >
            <option value="black">Black</option>
            <option value="cardjolote">Cardjolote</option>
            <option value="ocean-plastic">Ocean Plastic</option>
          </select>
        </div>
      </div>

      <div className="metrics-section">
        <MetricsPanel metrics={mockMetrics} />
      </div>

      <div className="content-container">
        <div className="chart-section">
          <ForecastChart data={forecastData} />
        </div>
        
        <div className="image-section">
          <img 
            src={forecastImage} 
            alt="Visualización de pronóstico" 
            className="forecast-image" 
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              const placeholder = document.querySelector('.image-placeholder') as HTMLElement | null;
              if (placeholder) {
                placeholder.style.display = 'block';
              }
            }}
          />
          <div className="image-placeholder" style={{ display: 'none' }}>
          </div>
        </div>
      </div>

    </div>
  );
};

export default ForecastPage;