import React from 'react';
import BotonBD1  from '../components/BotonBD1';
import CarruselBD1 from '../components/CarruselBD1';
import BotonPrueba from '../components/BotonPrueba';
import Tabla from '../components/Tabla';
import BatchTable from '../components/BatchTable';
// import { ForecastChart } from '../components/ForecastChart';
const DatabasePage = () => {
  return (
    <div>
      
      {/* <CarruselBD1>   </CarruselBD1> */}
      <CarruselBD1></CarruselBD1>
      <h1>.</h1>
      <BotonBD1></BotonBD1>
      <h1>.</h1>
      <Tabla></Tabla>   
    </div>
  );
};

export default DatabasePage;