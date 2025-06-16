import React from 'react';
import { MouseEvent } from 'react';

const BotonPrueba: React.FC = () => {
  return (
    <div className="container mt-4">
        <a className="btn btn-primary" href="#" role="button">Link</a>
        <button className="btn btn-primary" type="submit">Button</button>
        <input className="btn btn-primary" type="button" value="Input"/>
        <input className="btn btn-primary" type="submit" value="Submit"/>
        <input className="btn btn-primary" type="reset" value="Reset"/>
    </div>
  );
};

export default BotonPrueba;