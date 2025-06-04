import React from 'react';
import { Link } from 'react-router-dom';

export const Header: React.FC = () => {
  return (
    <header id="header" className="alt">
      <nav>
        <ul>
          <li>
            <Link to="/">Forecast</Link>
          </li>
          <li>
            <Link to="/base-datos">Base de Datos</Link>
          </li>
          <li>
            <Link to="/seguridad">Seguridad</Link>
          </li>
          <li>
            <Link to="/info">Info Proyecto</Link>
          </li>
        </ul>
      </nav>
    </header>
  );
};