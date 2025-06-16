import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

interface Dataset {
  label: string;
  data: (number | null)[];
  backgroundColor?: string;
  borderColor?: string;
  borderWidth?: number;
  fill?: boolean;
  tension?: number;
  borderDash?: number[];
  pointBackgroundColor?: string;
  pointRadius?: number;
  pointHoverRadius?: number;
}

interface ForecastChartProps {
  data: {
    labels: string[];
    datasets: Dataset[];
  };
  splitIndex?: number;
}

export const ForecastChart: React.FC<ForecastChartProps> = ({ data, splitIndex }) => {
  const options = {
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#000080',
          font: {
            size: 14,
            weight: 'bold' as const
          },
          padding: 20,
          usePointStyle: true,
          boxWidth: 12,
          boxHeight: 12
        }
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(0, 0, 128, 0.9)',
        titleColor: '#FFFFFF',
        bodyColor: '#FFFFFF',
        borderColor: '#4FC3F7',
        borderWidth: 1,
        padding: 12,
        usePointStyle: true,
        displayColors: true,
        bodyFont: {
          size: 14
        },
        titleFont: {
          size: 16,
          weight: 'bold' as const
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(0, 0, 128, 0.1)',
          drawBorder: false
        },
        ticks: {
          color: '#000080',
          font: {
            weight: 'bold' as const,
            size: 12
          }
        },
        border: {
          display: false
        }
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 128, 0.1)',
          drawBorder: false
        },
        ticks: {
          color: '#000080',
          font: {
            weight: 'bold' as const,
            size: 12
          }
        },
        border: {
          display: false
        }
      }
    }
  };

  return (
      <div style={{
        height: '500px',
        width: '100%',
        position: 'relative',
        margin: '20px 0'
      }}>
        <Line
            data={data}
            options={options}
            style={{
              width: '100%',
              height: '100%'
            }}
        />

        {splitIndex !== undefined && (
            <div style={{
              position: 'absolute',
              left: `calc(${(splitIndex / data.labels.length) * 100}% - 1px)`,
              top: '5%',
              bottom: '5%',
              width: '2px',
              borderLeft: '2px dashed #000080',
              opacity: 0.7,
              zIndex: 2,
              pointerEvents: 'none'
            }} />
        )}
      </div>
  );
};