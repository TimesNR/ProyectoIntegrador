import React from 'react'; 
import './MetricsPanel.css'; 

// Exporta la interfaz Metric
export interface Metric {
  name: string;
  value: number | string;
  improvement?: boolean;
  format?: 'percentage' | 'decimal';
}

interface MetricsPanelProps {
  metrics: Metric[];
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics }) => {
  const formatValue = (metric: Metric) => {
    if (metric.format === 'percentage') {
      return typeof metric.value === 'number' ? `${metric.value.toFixed(2)}%` : metric.value;
    }
    if (metric.format === 'decimal') {
      return typeof metric.value === 'number' ? metric.value.toFixed(4) : metric.value;
    }
    return metric.value;
  };

  return (
    <div className="metrics-panel">
      {metrics.map((metric, index) => (
        <div key={index} className={`metric-card ${metric.improvement !== undefined ? 
          (metric.improvement ? 'improvement' : 'worsening') : ''}`}>
          <h3>{metric.name}</h3>
          <div className="metric-value">{formatValue(metric)}</div>
          {metric.improvement !== undefined && (
            <div className={`trend-icon ${
              metric.improvement ? 'up' : 'down'
            }`}>
              {metric.improvement ? '↑' : '↓'}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default MetricsPanel;
