// dashboard/frontend/src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Global reset
document.body.style.margin   = '0';
document.body.style.padding  = '0';
document.body.style.background = '#0f172a';
document.body.style.color      = '#e2e8f0';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
