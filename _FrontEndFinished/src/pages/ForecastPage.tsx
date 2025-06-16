import React, { useState, useEffect } from 'react';
import { ForecastChart } from '../components/ForecastChart';
import MetricsPanel, { Metric } from '../components/MetricsPanel';
import './ForecastPage.css';



const constantMetrics: { [key: string]: Metric[] } = {
  "rappi_entregas_black": [
    { name: "mse", value: 6683513.5749, improvement: true, format: "decimal" },
    { name: "mae", value: 2355.2222, improvement: true, format: "decimal" },
    { name: "smape", value: 50.1472, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_cincita": [
    { name: "mse", value: 12327.9714, improvement: true, format: "decimal" },
    { name: "mae", value: 99.8597, improvement: true, format: "decimal" },
    { name: "smape", value: 112.7996, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_onix": [
    { name: "mse", value: 0.4249, improvement: true, format: "decimal" },
    { name: "mae", value: 0.5507, improvement: true, format: "decimal" },
    { name: "smape", value: 200.0, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_cincita_reemplazo": [
    { name: "mse", value: 1072.8175, improvement: true, format: "decimal" },
    { name: "mae", value: 30.3157, improvement: true, format: "decimal" },
    { name: "smape", value: 118.658, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_pride_2022": [
    { name: "mse", value: 182.7679, improvement: true, format: "decimal" },
    { name: "mae", value: 13.405, improvement: true, format: "decimal" },
    { name: "smape", value: 185.0493, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_mundial": [
    { name: "mse", value: 37772.2324, improvement: true, format: "decimal" },
    { name: "mae", value: 183.539, improvement: true, format: "decimal" },
    { name: "smape", value: 179.0604, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_pride_hologlam": [
    { name: "mse", value: 0.0348, improvement: true, format: "decimal" },
    { name: "mae", value: 0.1375, improvement: true, format: "decimal" },
    { name: "smape", value: 200.0, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_pride_colors": [
    { name: "mse", value: 1.0096, improvement: true, format: "decimal" },
    { name: "mae", value: 0.9022, improvement: true, format: "decimal" },
    { name: "smape", value: 200.0, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_cardlaveritas": [
    { name: "mse", value: 18.1841, improvement: true, format: "decimal" },
    { name: "mae", value: 2.4651, improvement: true, format: "decimal" },
    { name: "smape", value: 145.3945, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_cardtrinas": [
    { name: "mse", value: 14.1051, improvement: true, format: "decimal" },
    { name: "mae", value: 2.9757, improvement: true, format: "decimal" },
    { name: "smape", value: 100.1279, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_cardjolote": [
    { name: "mse", value: 73.3961, improvement: true, format: "decimal" },
    { name: "mae", value: 7.4112, improvement: true, format: "decimal" },
    { name: "smape", value: 106.6898, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_cardjolote_black": [
    { name: "mse", value: 1973062.0975, improvement: true, format: "decimal" },
    { name: "mae", value: 1043.1753, improvement: true, format: "decimal" },
    { name: "smape", value: 52.1372, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_ocean_plastic": [
    { name: "mse", value: 680126.8092, improvement: true, format: "decimal" },
    { name: "mae", value: 622.9862, improvement: true, format: "decimal" },
    { name: "smape", value: 32.1236, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_playcard": [
    { name: "mse", value: 697926.1825, improvement: true, format: "decimal" },
    { name: "mae", value: 664.9272, improvement: true, format: "decimal" },
    { name: "smape", value: 101.0793, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_gotica": [
    { name: "mse", value: 488.4295, improvement: true, format: "decimal" },
    { name: "mae", value: 22.0166, improvement: true, format: "decimal" },
    { name: "smape", value: 107.0931, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_pride_2024": [
    { name: "mse", value: 6080.3516, improvement: true, format: "decimal" },
    { name: "mae", value: 58.9534, improvement: true, format: "decimal" },
    { name: "smape", value: 162.2832, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_minion_rayo": [
    { name: "mse", value: 997.8451, improvement: true, format: "decimal" },
    { name: "mae", value: 23.6012, improvement: true, format: "decimal" },
    { name: "smape", value: 25.4122, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_minion_mega": [
    { name: "mse", value: 30757.0047, improvement: true, format: "decimal" },
    { name: "mae", value: 145.9242, improvement: true, format: "decimal" },
    { name: "smape", value: 200.0, improvement: true, format: "percentage" }
  ],
  "rappi_entregas_cardlaveritas_2024": [
    { name: "mse", value: 68185.4715, improvement: true, format: "decimal" },
    { name: "mae", value: 261.1235, improvement: true, format: "decimal" },
    { name: "smape", value: 143.033, improvement: true, format: "percentage" }
  ]
};

const forecastValues: { [key: string]: (number | null)[] } = {
  "rappi_entregas_black": [5100.65, 5128.28, 5200.35,5250.11,5280.08,5300.02],
  "rappi_entregas_cincita": [196.62, 196.62, 196.62, 196.62, 196.62, 196.62],
  "rappi_entregas_onix": [106.35, 106.35, 106.35, 106.35, 106.35, 106.35],
  "rappi_entregas_cincita_reemplazo": [45.04, 45.04, 45.04, 45.04, 45.04, 45.04],
  "rappi_entregas_pride_2022": [179.65, 179.65, 179.65, 179.65, 179.65, 179.65],
  "rappi_entregas_mundial": [283.88, 283.88, 283.88, 283.88, 283.88, 283.88],
  "rappi_entregas_pride_hologlam": [337.9, 337.9, 337.9, 337.9, 337.9, 337.9],
  "rappi_entregas_pride_colors": [168.05, 168.05, 182.48, 168.05, 168.05, 168.05],
  "rappi_entregas_cardlaveritas": [560.29, 560.29, 560.29, 560.29, 560.29, 560.29],
  "rappi_entregas_cardtrinas": [198.88, 198.88, 198.88, 198.88, 198.88, 198.88],
  "rappi_entregas_cardjolote": [297.88, 297.88, 297.88, 297.88, 297.88, 297.88],
  "rappi_entregas_cardjolote_black": [2464.62, 2610.85, 2464.62, 2464.62, 2464.62, 2464.62],
  "rappi_entregas_ocean_plastic": [1470.31, 1470.31, 1470.31, 2048.06, 2048.06, 2009.06],
  "rappi_entregas_playcard": [1634.58, 11608.28, 1794.98, 1634.58, 1634.58, 1634.58],
  "rappi_entregas_gotica": [922.7, 922.7, 922.7, 922.7, 922.7, 922.7],
  "rappi_entregas_pride_2024": [1035.67, 1035.67, 1035.67, 1035.67, 1035.67, 1035.67],
  "rappi_entregas_minion_rayo": [539.62, 539.62, 539.62, 539.62, 539.62, 539.62],
  "rappi_entregas_minion_mega": [508.5, 508.5, 508.5, 508.5, 508.5, 508.5],
  "rappi_entregas_cardlaveritas_2024": [1769.0, 1769.0, 1769.0, 1769.0, 1769.0, 1769.0]
};


const historicalValues: { [key: string]: (number | null)[] } = {
  "rappi_entregas_black": [6926, 7981, 9155, 7257, 4313, 4427, 3468, 2403, 1661],
  "rappi_entregas_cincita": [320, 228, 182, 251, 102, 52, 0, 0, 0],
  "rappi_entregas_onix": [123, 106, 115, 103, 99, 104, 101, 96, 90],
  "rappi_entregas_cincita_reemplazo": [44, 30, 22, 22, 27, 27, 0, 0, 0],
  "rappi_entregas_pride_2022": [null, null, null, null, 104, 149, 149, 143, 148],
  "rappi_entregas_mundial": [null, null, null, null, 94, 139, 139, 150, 146],
  "rappi_entregas_pride_hologlam": [null, null, null, null, null, 15, 43, 52, 56],
  "rappi_entregas_pride_colors": [null, null, null, null, null, 16, 44, 50, 52],
  "rappi_entregas_cardlaveritas": [null, null, null, null, null, null, 206, 251, 294],
  "rappi_entregas_cardtrinas": [null, null, null, null, null, null, 64, 102, 126],
  "rappi_entregas_cardjolote": [null, null, null, null, null, null, 128, 168, 205],
  "rappi_entregas_cardjolote_black": [172, 84, 48, 6157, 3512, 2490, 864, 433, 1853],
  "rappi_entregas_ocean_plastic": [3574, 2996, 2066, 2724, 1859, 2951, 1945, 1343, 1135],
  "rappi_entregas_playcard": [null, null, null, null, null, null, 2009, 2566, 2709],
  "rappi_entregas_gotica": [null, null, null, null, null, null, 197, 228, 266],
  "rappi_entregas_pride_2024": [null, null, null, null, null, null, 291, 354, 390],
  "rappi_entregas_minion_rayo": [null, null, null, null, null, null, 391, 444, 456],
  "rappi_entregas_minion_mega": [null, null, null, null, null, null, 274, 296, 319],
  "rappi_entregas_cardlaveritas_2024": [null, null, null, null, null, null, 844, 1000, 1075]
};





const seriesNombres = Object.keys(constantMetrics);

const ForecastPage = () => {
  const [selectedSerie, setSelectedSerie] = useState(seriesNombres[0]);
  const [forecastData, setForecastData] = useState<(number | null)[]>([]);
  const [historicalData, setHistoricalData] = useState<(number | null)[]>([]);

  const splitIndex = 8;
  const labels = [
    '01-Jun', '01-Jul', '01-Ago', '01-Sep', '01-Oct', '01-Nov', '01-Dic', '01-Ene', '01-Feb',
    '01-Mar', '01-Abr', '01-Mayo', '01-Jun', '01-Jul', '01-Ago'
  ];


  useEffect(() => {
    const fetchData = async () => {
      try {
        const forecast = forecastValues[selectedSerie] || [];
        const paddedForecast = Array(9).fill(null).concat(forecast.slice(0, 6));
        setForecastData(paddedForecast);
        setHistoricalData(historicalValues[selectedSerie] || Array(9).fill(null));
      } catch (err) {
        console.error("Error al cargar pronóstico", err);
      }
    };
    fetchData();
  }, [selectedSerie]);






  const chartData = {
    labels,
    datasets: [
      {
        label: 'Histórico',
        data: historicalData,
        borderColor: '#4FC3F7',
        backgroundColor: 'rgba(79, 195, 247, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        pointBackgroundColor: '#4FC3F7',
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: true
      },
      {
        label: `Pronóstico - ${selectedSerie}`,
        data: forecastData,
        borderColor: '#000080',
        backgroundColor: 'rgba(0, 0, 128, 0.2)',
        borderWidth: 3,
        borderDash: [5, 5],
        tension: 0.4,
        pointBackgroundColor: '#000080',
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: true
      }
    ]
  };


  return (
      <div className="forecast-page">
        <header className="page-header">
          <h1 className="page-title">Proyección Geométrica</h1>
          <p className="page-subtitle">Visualización de datos históricos y predicción</p>
        </header>

        <div className="controls-section">
          <label htmlFor="serie-select">Serie:</label>
          <select
              id="serie-select"
              value={selectedSerie}
              onChange={(e) => setSelectedSerie(e.target.value)}
              className="material-select"
          >
            {seriesNombres.map(name => (
                <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>

        <div className="metrics-section">
          <MetricsPanel metrics={constantMetrics[selectedSerie]} />
        </div>

        <div className="chart-container">
          <ForecastChart data={chartData} splitIndex={splitIndex} />
        </div>
      </div>
  );
};

export default ForecastPage;
