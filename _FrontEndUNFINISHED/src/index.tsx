import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

import $ from 'jquery';

declare global {
  interface Window {
    $: typeof $;
    jQuery: typeof $;
  }
}

window.$ = window.jQuery = $;

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);