import React from 'react';
import Ongay from '../assets/comp.jpg';
import Alan from '../assets/comp0.jpg';
import JP from '../assets/comp1.jpg';
import Bali from '../assets/comp2.jpg';
import Rafa from '../assets/comp3.jpg';
import Xime from '../assets/comp4.jpg';
import Ale from '../assets/comp5.jpg';
import '../components/styles/InfoPage.css';

const teamMembers = [
  {
    name: 'Ivan Ongay Valverde',
    role: 'Profesor líder del proyecto',
    image: Ongay,
  },
  {
    name: 'Alan Uriel Merlan Esquivel - A01656612',
    role: 'Scrum Master',
    image: Alan,
  },
  {
    name: 'Juan Pablo Rodriguez - A01662334',
    role: 'Frontend',
    image: JP,
  },
  {
    name: 'Iker Sebastian Bali Elizalde - A01656437',
    role: 'Modelado',
    image: Bali,
  },
  {
    name: 'Rafael Barroso Portugal - A01662031',
    role: 'Modelado',
    image: Rafa,
  },
  {
    name: 'Maria Ximena Rocha Valle - A01706707',
    role: 'Base de Datos',
    image: Xime,
  },
  {
    name: 'Alejandro Diaz Ruiz - A01655911',
    role: 'Backend',
    image: Ale,
  },
];

const InfoPage = () => {
  return (
    <div className="info-page">
      <div className="info-header">
        <h1 className="info-title">Información del Proyecto</h1>
        <p className="info-description">
          Este proyecto tiene como objetivo optimizar el stock basándonos en una predicción 
          de las tendencias de las tarjetas de Rappi.
        </p>
      </div>

      <div className="info-section">
        <h2 className="section-title">Equipo de trabajo</h2>
        <div className="team-grid">
          {teamMembers.map((member, index) => (
            <div key={index} className="team-card">
              <div className="photo-container">
                <img src={member.image} alt={member.name} className="team-photo" />
              </div>
              <div className="member-info">
                <h3 className="member-name">{member.name}</h3>
                <p className="member-role">{member.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="info-section">
        <h2 className="section-title">Tecnologías usadas</h2>
        <div className="tech-grid">
          <div className="tech-card">
            <div className="tech-icon react-icon"></div>
            <p>React + TypeScript</p>
          </div>
          <div className="tech-card">
            <div className="tech-icon python-icon"></div>
            <p>Python</p>
          </div>
          <div className="tech-card">
            <div className="tech-icon sql-icon"></div>
            <p>SQL</p>
          </div>
          <div className="tech-card">
            <div className="tech-icon aws-icon"></div>
            <p>Amazon Web Services</p>
          </div>
          <div className="tech-card">
            <div className="tech-icon chartjs-icon"></div>
            <p>Chart.js</p>
          </div>
        </div>
      </div>

      <div className="info-section contact-section">
        <h2 className="section-title">Contacto</h2>
        <p className="contact-info">
          Tecnológico de Monterrey · Escuela de Ingeniería y Ciencias
        </p>
      </div>
    </div>
  );
};

export default InfoPage;

// import React from 'react';

// const teamMembers = [
//   {
//     name: 'Ivan Ongay Valverde',
//     role: 'Profesor líder del proyecto',
//     image: '/images/comp.jpg',
//   },
//   {
//     name: 'Alan Uriel Merlan Esqeuivel - A01656612',
//     role: 'Scrum Master',
//     image: '/images/comp0.jpg',
//   },
//   {
//     name: 'Juan Pablo Rodriguez - A01662334',
//     role: 'Frontend',
//     image: '/images/comp1.jpg',
//   },
//   {
//     name: 'Iker Sebastian Bali Elizalde - A01656437',
//     role: 'Modelado',
//     image: '/images/comp2.jpg',
//   },
//   {
//     name: 'Rafael Barroso Portugal - A01662031',
//     role: 'Modelado',
//     image: '/images/comp3.jpg',
//   },
//   {
//     name: 'Maria Ximena Rocha Valle - A01706707',
//     role: 'Base de Datos',
//     image: '/images/comp4.jpg',
//   },
//   {
//     name: 'Alejandro Diaz Ruiz - A01655911',
//     role: 'Backend',
//     image: '/images/comp5.jpg',
//   },
// ];

// const InfoPage = () => {
//   return (
//     <div className="info-container">
//       <h1>Información del Proyecto</h1>
//       <p>
//         Este proyecto tiene como objetivo optimizar el stock basandonos en una prediccion de las tendencias de las tarjetas de Rappi.
//       </p>

//       <h2>Equipo de trabajo</h2>
//       <div className="team-grid">
//         {teamMembers.map((member, index) => (
//           <div key={index} className="team-member">
//             <img src={member.image} alt={member.name} className="team-photo" />
//             <h3>{member.name}</h3>
//             <p>{member.role}</p>
//           </div>
//         ))}
//       </div>

//       <h2>Tecnologías usadas</h2>
//       <ul>
//         <li>React + TypeScript</li>
//         <li>Python</li>
//         <li>Chart.js para visualización</li>
//       </ul>

//       <h2>Contacto</h2>
//       <p>Tecnologico de Monterrey · Escuela de Ingenieria y Ciencias</p>
//     </div>
//   );
// };

// export default InfoPage;
