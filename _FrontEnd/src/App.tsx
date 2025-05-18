import { BrowserRouter, Routes, Route } from 'react-router-dom';
import ForecastPage from './pages/ForecastPage';
import DatabasePage from './pages/DatabasePage'; 
import SecurityPage from './pages/SecurityPage';
import InfoPage from './pages/InfoPage'; 
import MainLayout from './components/layout/MainLayout'; 
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'; 
import "bootstrap"

const App = () => {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<ForecastPage />} />
          <Route path="/base-datos" element={<DatabasePage />} />
          <Route path="/seguridad" element={<SecurityPage />} />
          <Route path="/info" element={<InfoPage />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
};

export default App;