import React from 'react';
import BotonBD1  from '../components/BotonBD1';
import CarruselBD1 from '../components/CarruselBD1';
import BotonPrueba from '../components/BotonPrueba';
// import { ForecastChart } from '../components/ForecastChart';
const DatabasePage = () => {
  return (
    <div>
      
      {/* <CarruselBD1>   </CarruselBD1> */}
      <CarruselBD1></CarruselBD1>
      <h1>Base de Datos</h1>
      <BotonBD1></BotonBD1>
    </div>
  );
};

export default DatabasePage;