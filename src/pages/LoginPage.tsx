import React, { useState, useEffect } from 'react';

interface Props {
    onLoginSuccess: () => void;
}

const LoginPage: React.FC<Props> = ({ onLoginSuccess }) => {
    const [usuario, setUsuario] = useState('');
    const [contrasena, setContrasena] = useState('');
    const [error, setError] = useState('');
    const [sesionActiva, setSesionActiva] = useState(false);

    useEffect(() => {
        const estaAutenticado = localStorage.getItem('autenticado') === 'true';
        setSesionActiva(estaAutenticado);
    }, []);

    const handleLogin = () => {
        const userValido = 'admin';
        const passValido = '1234';

        if (usuario === userValido && contrasena === passValido) {
            localStorage.setItem('autenticado', 'true');
            onLoginSuccess();
        } else {
            setError('Usuario o contraseña incorrectos');
        }
    };

    const cerrarSesion = () => {
        localStorage.removeItem('autenticado');
        window.location.reload();
    };

    return (
        <div style={{ padding: '30px', textAlign: 'center', fontFamily: 'Arial, sans-serif' }}>
            <h2>Iniciar sesión</h2>

            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '10px' }}>
                <input
                    type="text"
                    placeholder="Usuario"
                    value={usuario}
                    onChange={(e) => setUsuario(e.target.value)}
                    style={{
                        padding: '10px',
                        width: '200px',
                        fontSize: '14px'
                    }}
                />
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '10px' }}>
                <input
                    type="password"
                    placeholder="Contraseña"
                    value={contrasena}
                    onChange={(e) => setContrasena(e.target.value)}
                    style={{
                        padding: '10px',
                        width: '200px',
                        fontSize: '14px'
                    }}
                />
            </div>

            <button
                onClick={handleLogin}
                style={{
                    backgroundColor: '#007bff',
                    color: 'white',
                    padding: '10px 20px',
                    fontSize: '14px',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    marginTop: '10px'
                }}
            >
                Ingresar
            </button>

            {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}

            {sesionActiva && (
                <div style={{ marginTop: '20px' }}>
                    <button
                        onClick={cerrarSesion}
                        style={{
                            backgroundColor: '#dc3545',
                            color: 'white',
                            padding: '10px 20px',
                            fontSize: '14px',
                            border: 'none',
                            borderRadius: '5px',
                            cursor: 'pointer'
                        }}
                    >
                        Cerrar sesión
                    </button>
                </div>
            )}
        </div>
    );
};

export default LoginPage;
