import { BrowserRouter, Routes, Route } from 'react-router-dom';
import ForecastPage from './pages/ForecastPage';
import DatabasePage from './pages/DatabasePage'; 
import SecurityPage from './pages/SecurityPage';
import InfoPage from './pages/InfoPage'; 
import MainLayout from './components/layout/MainLayout'; 

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