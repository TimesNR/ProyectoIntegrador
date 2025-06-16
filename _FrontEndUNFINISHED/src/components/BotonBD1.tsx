// src/components/BotonBD1.tsx
import React from 'react';
import './BotonBD1.css'; // Archivo CSS personalizado

const BotonBD1: React.FC = () => {
  return (
    <div className="container mt-4 text-center"> {/* text-center centra los botones */}
      <button type="button" className="btn btn-secondary btn-lg oval-button">Actualizar</button>
      <button type="button" className="btn btn-secondary btn-lg oval-button">Nueva Tarjeta</button>
      <button type="button" className="btn btn-secondary btn-lg oval-button">Cambio Total Datos</button>
      <button type="button" className="btn btn-secondary btn-lg oval-button">Borrar Tarjeta</button>
    </div>
  );
};

export default BotonBD1;





// const BotonBD1: React.FC = () => {
//   return (
//     <div className="container d-flex justify-content-center align-items-center vh-100">
//       <div className="card shadow-lg" style={{ width: '700px' }}>
//         <div className="card-body">
//             <div className = "card mx-auto text-center" > 
//                 <h2 className="card-title text-center mb-4">Selecciona las funciones:</h2>

//                 <button type="button" className="btn btn-primary btn-block mb-2">
//                     Actualizar
//                 </button>
//                 <button type="button" className="btn btn-secondary btn-block mb-2">
//                     Nueva Tarjeta
//                 </button>
//                 <button type="button" className="btn btn-warning btn-block mb-2">
//                     Cambio Total Datos
//                 </button>
//                 <button type="button" className="btn btn-danger btn-block">
//                     Borrar Tarjeta
//                 </button>
//             </div>
//         </div>
//       </div>
//     </div>
//   );
// };
