import React from 'react';
import './styles/metricCard.css'; 

interface MetricCardProps {
  title: string;
  value: string;
  trend?: 'up' | 'down' | 'neutral';
  description?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  trend = 'neutral',
  description
}) => {
  const trendIcons = {
    up: '↑',
    down: '↓',
    neutral: '→'
  };

  return (
    <div className={`metric-card ${trend}`}>
      <h3>{title}</h3>
      <div className="metric-value">
        {value} 
        <span className="trend-icon">{trendIcons[trend]}</span>
      </div>
      {description && <p className="metric-description">{description}</p>}
    </div>
  );
};