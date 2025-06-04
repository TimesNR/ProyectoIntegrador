// src/components/BotonBD1.tsx
import React, { useState, useEffect } from 'react';
import { Modal, Button, Form } from 'react-bootstrap';
import './BotonBD1.css';

const BotonBD1: React.FC = () => {
    const [show, setShow] = useState<string | null>(null);
    console.log("Valor de show:", show);
    const [registroData, setRegistroData] = useState<Record<string, string | number>>({});
    const [nombreColumna, setNombreColumna] = useState('');
    const [archivo, setArchivo] = useState<File | null>(null);
    const [columnas, setColumnas] = useState<string[]>([]);
    const [fechaEliminar, setFechaEliminar] = useState(''); //ultima hora


    const handleClose = () => setShow(null);

    //Cambios necesarios para arreglar los campos de la ventana emergente del boton actualizar

    useEffect(() => {
        if (show === 'registro') {
            console.log("Cargando columnas...");
            fetch('/columnas')
                .then(res => res.json())
                .then(data => {
                    console.log("Columnas recibidas:", data);
                    setColumnas(data);
                })
                .catch(err => console.error('Error al cargar columnas', err));
        }
    }, [show]);

    const handleInputChange = (col: string, value: string) => {
        setRegistroData(prev => ({
            ...prev,
            [col]: isNaN(Number(value)) ? value : Number(value)
        }));
    };
//Hasta aquí

    const handleAgregarRegistro = async () => {
        const res = await fetch('/agregar_registro', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(registroData),
        });
        const result = await res.json();
        alert(result.mensaje || result.error);
        handleClose();
    };

    const handleAgregarTarjeta = async () => {
        const res = await fetch('/agregar_tarjeta', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nombre_columna: nombreColumna }),
        });
        const result = await res.json();
        alert(result.mensaje || result.error);
        handleClose();
    };

    const handleQuitarTarjeta = async () => {
        const res = await fetch('/quitar_tarjeta', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nombre_columna: nombreColumna }),
        });
        const result = await res.json();
        alert(result.mensaje || result.error);
        handleClose();
    };

    const handleActualizarConTabla = async () => {
        if (!archivo) return alert('Selecciona un archivo');

        const formData = new FormData();
        formData.append('file', archivo);

        const res = await fetch('/actualizar_con_tabla', {
            method: 'POST',
            body: formData,
        });
        const result = await res.json();
        alert(result.mensaje || result.error);
        handleClose();
    };

    //ultima hora
    const handleEliminarFila = async () => {
        if (!fechaEliminar) {
            alert("Por favor ingresa una fecha.");
            return;
        }

        const res = await fetch('/eliminar_por_fecha', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fecha: fechaEliminar }),
        });

        const result = await res.json();
        alert(result.mensaje || result.error);
        handleClose();
    };


    return (
        <div className="container mt-4 text-center">
            <button className="btn btn-primary btn-lg oval-button" onClick={() => setShow('registro')}>Actualizar</button>
            <button className="btn btn-primary btn-lg oval-button" onClick={() => setShow('nueva')}>Nueva Tarjeta</button>
            <button className="btn btn-primary btn-lg oval-button" onClick={() => setShow('cambio')}>Actualizar con tabla</button>
            <button className="btn btn-danger btn-lg oval-button" onClick={() => setShow('borrar')}>Eliminar Tarjeta</button>
            <button className="btn btn-danger btn-lg oval-button" onClick={() => setShow('eliminar')}>Eliminar Fila</button>

            {/* Modal Agregar Registro */}
            <Modal show={show === 'registro'} onHide={handleClose}>
                <Modal.Header closeButton>
                    <Modal.Title>Agregar Registro</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <Form>
                        {columnas.map((col) => (
                            <Form.Group key={col} className="mb-3">
                                <Form.Label>{col}</Form.Label>
                                <Form.Control
                                    type="text"
                                    onChange={(e) => handleInputChange(col, e.target.value)}
                                />
                            </Form.Group>
                        ))}
                    </Form>
                </Modal.Body>
                <Modal.Footer>
                    <Button onClick={handleAgregarRegistro}>Enviar</Button>
                </Modal.Footer>
            </Modal>


            {/* Modal Nueva Tarjeta */}
            <Modal show={show === 'nueva'} onHide={handleClose}>
                <Modal.Header closeButton><Modal.Title>Agregar Tarjeta</Modal.Title></Modal.Header>
                <Modal.Body>
                    <Form.Control placeholder="Nombre de columna" onChange={(e) => setNombreColumna(e.target.value)} />
                </Modal.Body>
                <Modal.Footer>
                    <Button onClick={handleAgregarTarjeta}>Agregar</Button>
                </Modal.Footer>
            </Modal>

            {/* Modal Borrar Tarjeta */}
            <Modal show={show === 'borrar'} onHide={handleClose}>
                <Modal.Header closeButton><Modal.Title>Eliminar Tarjeta</Modal.Title></Modal.Header>
                <Modal.Body>
                    <Form.Control placeholder="Nombre de columna" onChange={(e) => setNombreColumna(e.target.value)} />
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="danger" onClick={handleQuitarTarjeta}>Borrar</Button>
                </Modal.Footer>
            </Modal>

            {/* Modal Eliminar Fila */}
            <Modal show={show === 'eliminar'} onHide={handleClose}>
                <Modal.Header closeButton><Modal.Title>Eliminar Fila por Fecha</Modal.Title></Modal.Header>
                <Modal.Body>
                    <Form.Control
                        placeholder="Fecha a eliminar (YYYY-MM-DD)"
                        onChange={(e) => setFechaEliminar(e.target.value)}
                    />
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="danger" onClick={handleEliminarFila}>Eliminar</Button>
                </Modal.Footer>
            </Modal>


            {/* Modal Actualizar con Excel */}
            <Modal show={show === 'cambio'} onHide={handleClose}>
                <Modal.Header closeButton><Modal.Title>Actualizar con Excel</Modal.Title></Modal.Header>
                <Modal.Body>
                    <Form.Control
                        type="file"
                        accept=".xlsx, .xls"
                        onChange={(e) => {
                            const file = (e.target as HTMLInputElement).files?.[0] || null;
                            setArchivo(file);
                        }}
                    />
                </Modal.Body>
                <Modal.Footer>
                    <Button onClick={handleActualizarConTabla}>Subir</Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
};

export default BotonBD1;