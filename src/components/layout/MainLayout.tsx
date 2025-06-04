import { Outlet, Link } from 'react-router-dom';
import { ReactNode } from 'react';

interface MainLayoutProps {
  children?: ReactNode;
}

const MainLayout = ({ children }: MainLayoutProps) => {
  return (
    <div id="page-wrapper">
      <header id="header" className="alt">
        <nav>
          <ul>
            <li><Link to="/">Forecast</Link></li>
            <li><Link to="/base-datos">Base de Datos</Link></li>
            <li><Link to="/seguridad">Seguridad</Link></li>
            <li><Link to="/info">Info Proyecto</Link></li>
          </ul>
        </nav>
      </header>

      {/* Esto renderiza la subruta */}
      <Outlet />
      {/* Esto renderiza cualquier contenido pasado como children */}
      {children}

      <footer id="footer">
        <p>© Forecast Rappi {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
};

export default MainLayout;
