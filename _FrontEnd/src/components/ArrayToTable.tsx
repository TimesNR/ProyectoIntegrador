// // src/components/DataTable.tsx

import React from 'react';

interface DataTableProps {
  data: Record<string, any>[];
}

const ArrayToTable: React.FC <DataTableProps>= ({ data }) => {
  if (data.length === 0) return <p>No hay datos para mostrar.</p>;

  const columns = Object.keys(data[0]);

  return (
    <div className="table-responsive">
      <table className="table table-striped">
        <thead className = "thead-dark">
          <tr>
            {columns.map((col) => (
              <th key={col}> <span style={{color: "black"}}> {col}</span> </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col}>{row[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// export default DataTable;
export default ArrayToTable;
