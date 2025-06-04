import React, { useEffect, useState } from 'react';
import ArrayToTable from './ArrayToTable';

const Tabla = () => {
    const [data, setData] = useState<Record<string, any>[]>([]);
    const [columnas, setColumnas] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/registros')
            .then(res => res.json())
            .then(json => {
                setData(json);
                setLoading(false);
            });

        fetch('/columnas')
            .then(res => res.json())
            .then(colData => {
                // Filtrar solo columnas con "entregas" o "demanda_total_tarjetas"
                const columnasFiltradas = colData.filter((col: string) =>
                    col.includes('entregas') || col === 'fecha'
                );
                setColumnas(columnasFiltradas);
            });
    }, []);


    if (loading) return <p>Cargando datos...</p>;

    return (
        <div className="container mt-4">
            <ArrayToTable data={data} columns={columnas} />
        </div>
    );
};

export default Tabla;
