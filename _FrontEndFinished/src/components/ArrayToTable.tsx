import React from 'react';
import './ArrayToTable.css';

interface DataTableProps {
  data: Record<string, any>[];
  columns: string[];
}

const ArrayToTable: React.FC<DataTableProps> = ({ data, columns }) => {
  if (!data || data.length === 0 || columns.length === 0) return <p>No hay datos para mostrar.</p>;

  return (
      <div className="table-responsive">
        <table className="table table-striped table-bordered table-sm compact-table">
          <thead className="thead-dark">
          <tr>
              {columns.map((col) => {
                  // Transformar el nombre para mostrar
                  const displayName = col
                      .replace(/_/g, ' ')              // reemplaza _ por espacio
                      .replace(/\b\w/g, (c) => c.toUpperCase()); // capitaliza cada palabra
                  return (
                      <th key={col}>
                          <span style={{ color: 'black', fontSize: '0.9rem' }}>{displayName}</span>
                      </th>
                  );
              })}
          </tr>
          </thead>
            <tbody>
            {data.map((row, idx) => (
                <tr key={idx}>
                    {columns.map((col) => {
                        let value = row[col];

                        // Si es una fecha, formatearla como MM/YYYY
                        if (col === 'fecha' && value) {
                            // Si es una fecha, formatearla como MM/YYYY
                            if (col === 'fecha' && value) {
                                const dateObj = new Date(value);
                                const month = (dateObj.getMonth() + 1).toString().padStart(2, '0');
                                const year = dateObj.getFullYear();
                                value = `${month}/${year}`;
                            }

                        }

                        return <td key={col}>{value}</td>;
                    })}
                </tr>
            ))}
            </tbody>
        </table>
      </div>
  );
};

export default ArrayToTable;