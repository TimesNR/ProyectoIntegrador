import React, { useState } from 'react';
import { ForecastChart } from '../components/ForecastChart';
import { MetricCard } from '../components/MetricCard';
import { ForecastControls } from '../components/ForecastControls';

interface ForecastData {
  semanal: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor: string;
    }[];
  };
  mensual: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor: string;
    }[];
  };
  anual: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor: string;
    }[];
  };
}

const forecastData: ForecastData = {
  semanal: {
    labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie'],
    datasets: [
      {
        label: 'Visitas',
        data: [10, 20, 15, 30, 25],
        backgroundColor: 'rgba(75,192,192,0.4)',
      },
    ],
  },
  mensual: {
    labels: ['Semana 1', 'Semana 2', 'Semana 3', 'Semana 4'],
    datasets: [
      {
        label: 'Visitas',
        data: [100, 150, 120, 180],
        backgroundColor: 'rgba(153,102,255,0.6)',
      },
    ],
  },
  anual: {
    labels: ['Ene', 'Feb', 'Mar', 'Abr'],
    datasets: [
      {
        label: 'Visitas',
        data: [400, 420, 500, 470],
        backgroundColor: 'rgba(255,159,64,0.6)',
      },
    ],
  },
};

const ForecastPage = () => {
  const [timeRange, setTimeRange] = useState<'semanal' | 'mensual' | 'anual'>('semanal');
  const [activeBrand, setActiveBrand] = useState<string>('Mejora');

  const handleTimeRangeChange = (range: 'semanal' | 'mensual' | 'anual') => {
    setTimeRange(range);
  };

  const handleBrandChange = (brand: string) => {
    setActiveBrand(brand);
  };

  return (
    <div>
      <h1>Forecast Page</h1>
      <ForecastControls 
        timeRange={timeRange}
        onTimeRangeChange={handleTimeRangeChange}
        activeBrand={activeBrand}
        onBrandChange={handleBrandChange}
      />
      <ForecastChart data={forecastData[timeRange]} />
      <MetricCard title="Resumen" value="1234" />
    </div>
  );
};

export default ForecastPage;
