import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import ForecastPage from './pages/ForecastPage';
import DatabasePage from './pages/DatabasePage';
import InfoPage from './pages/InfoPage';
import LoginPage from './pages/LoginPage';
import MainLayout from './components/layout/MainLayout';

import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';

const App = () => {
    const [autenticado, setAutenticado] = useState<boolean>(false);

    useEffect(() => {
        const sesionGuardada = localStorage.getItem('autenticado');
        if (sesionGuardada === 'true') {
            setAutenticado(true);
        }
    }, []);

    const manejarLoginExitoso = () => {
        localStorage.setItem('autenticado', 'true');
        setAutenticado(true);
    };

    const cerrarSesion = () => {
        localStorage.removeItem('autenticado');
        window.location.reload();
    };

    if (!autenticado) {
        return <LoginPage onLoginSuccess={manejarLoginExitoso} />;
    }

    return (
        <BrowserRouter>
            <MainLayout>
                <Routes>
                    <Route path="/" element={<ForecastPage />} />
                    <Route path="/base-datos" element={<DatabasePage />} />
                    <Route path="/info" element={<InfoPage />} />
                </Routes>
            </MainLayout>
        </BrowserRouter>
    );
};

export default App;
