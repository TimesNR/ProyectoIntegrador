import React from 'react';
import img1 from '../assets/c_IMG1.jpg';
import img2 from '../assets/c_IMG2.jpg';
import img3 from '../assets/c_IMG3.jpg';

const CarruselBD1: React.FC = () => {
  return (
    <div id="carouselExampleIndicators" className="carousel slide" data-bs-ride="carousel">
      <ol className="carousel-indicators">
        <li data-bs-target="#carouselExampleIndicators" data-bs-slide-to="0" className="active"></li>
        <li data-bs-target="#carouselExampleIndicators" data-bs-slide-to="1"></li>
        <li data-bs-target="#carouselExampleIndicators" data-bs-slide-to="2"></li>
      </ol>

      <div className="carousel-inner">
        <div className="carousel-item active">
          <img className="d-block w-100" src={img1} alt="Actualización" />
          <div className="carousel-caption d-none d-md-block"
              style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center",
              width: "80%",
            }}>
            <h1> <span style={{ fontSize: "250%"}}>  Actualizar</span></h1>
            <p> Sube los datos más recientes</p>
          </div>
        </div>

        <div className="carousel-item">
          <img className="d-block w-100" src={img2} alt="Borrar o agregar tarjetas" />
          <div className="carousel-caption d-none d-md-block"
              style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center",
              width: "80%",
            }}>
            <h1> <span style={{ fontSize: "250%"}}>  Nueva Tarjeta</span></h1>
            <p> Agrega la ultima tarjeta o borra una fuera de stock</p>
          </div>
        </div>

        <div className="carousel-item">
          <img className="d-block w-100" src={img3} alt="Cambio total DF" />
          <div className="carousel-caption d-none d-md-block"
              style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center",
              width: "80%",
            }}>
            <h1> <span style={{ fontSize: "250%"}}>  Nueva Data</span></h1>
            <p> Modifica toda la data disponible</p>
          </div>
        </div>
      </div>

      <a className="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-bs-slide="prev">
        <span className="carousel-control-prev-icon" aria-hidden="true"></span>
        <span className="visually-hidden">Previous</span>
      </a>
      <a className="carousel-control-next" href="#carouselExampleIndicators" role="button" data-bs-slide="next">
        <span className="carousel-control-next-icon" aria-hidden="true"></span>
        <span className="visually-hidden">Next</span>
      </a>
    </div>
  );
};

export default CarruselBD1;
