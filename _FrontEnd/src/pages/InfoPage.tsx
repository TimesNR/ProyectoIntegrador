import React from 'react';

const teamMembers = [
  {
    name: 'Ivan Ongay Valverde',
    role: 'Profesor líder del proyecto',
    image: '/images/comp.jpg',
  },
  {
    name: 'Alan Uriel Merlan Esqeuivel - A01656612',
    role: 'Scrum Master',
    image: '/images/comp0.jpg',
  },
  {
    name: 'Juan Pablo Rodriguez - A01662334',
    role: 'Frontend',
    image: '/images/comp1.jpg',
  },
  {
    name: 'Iker Sebastian Bali Elizalde - A01656437',
    role: 'Modelado',
    image: '/images/comp2.jpg',
  },
  {
    name: 'Rafael Barroso Portugal - A01662031',
    role: 'Modelado',
    image: '/images/comp3.jpg',
  },
  {
    name: 'Maria Ximena Rocha Valle - A01706707',
    role: 'Base de Datos',
    image: '/images/comp4.jpg',
  },
  {
    name: 'Alejandro Diaz Ruiz - A01655911',
    role: 'Backend',
    image: '/images/comp5.jpg',
  },
];

const InfoPage = () => {
  return (
    <div className="info-container">
      <h1>Información del Proyecto</h1>
      <p>
        Este proyecto tiene como objetivo optimizar el stock basandonos en una prediccion de las tendencias de las tarjetas de Rappi.
      </p>

      <h2>Equipo de trabajo</h2>
      <div className="team-grid">
        {teamMembers.map((member, index) => (
          <div key={index} className="team-member">
            <img src={member.image} alt={member.name} className="team-photo" />
            <h3>{member.name}</h3>
            <p>{member.role}</p>
          </div>
        ))}
      </div>

      <h2>Tecnologías usadas</h2>
      <ul>
        <li>React + TypeScript</li>
        <li>Python</li>
        <li>Chart.js para visualización</li>
      </ul>

      <h2>Contacto</h2>
      <p>Tecnologico de Monterrey · Escuela de Ingenieria y Ciencias</p>
    </div>
  );
};

export default InfoPage;
