// dashboard/frontend/src/App.jsx
import React, { useState } from 'react';
import Overview   from './pages/Overview';
import Privacy    from './pages/Privacy';
import Explain    from './pages/Explain';
import Config     from './pages/Config';

const NAV = [
  { id: 'overview',  label: '⬡  Overview',      component: Overview  },
  { id: 'privacy',   label: '🔒 Privacy',        component: Privacy   },
  { id: 'explain',   label: '🔍 Explainability', component: Explain   },
  { id: 'config',    label: '⚙  Config',         component: Config    },
];

const STYLES = {
  app: {
    minHeight: '100vh',
    background: '#0f172a',
    display: 'flex',
    fontFamily: "'Inter', -apple-system, sans-serif",
  },
  sidebar: {
    width: 220,
    background: '#1e293b',
    borderRight: '1px solid #334155',
    display: 'flex',
    flexDirection: 'column',
    padding: '24px 0',
    flexShrink: 0,
  },
  logo: {
    color: '#6366f1',
    fontSize: 22,
    fontWeight: 800,
    padding: '0 20px 24px',
    borderBottom: '1px solid #334155',
    marginBottom: 16,
    letterSpacing: '-0.5px',
  },
  navItem: (active) => ({
    padding: '10px 20px',
    cursor: 'pointer',
    color:  active ? '#e2e8f0' : '#64748b',
    background: active ? '#334155' : 'transparent',
    borderLeft: `3px solid ${active ? '#6366f1' : 'transparent'}`,
    fontSize: 14,
    fontWeight: active ? 600 : 400,
    transition: 'all 0.15s',
    userSelect: 'none',
  }),
  main: {
    flex: 1,
    overflowY: 'auto',
    background: '#0f172a',
  },
  topbar: {
    background: '#1e293b',
    borderBottom: '1px solid #334155',
    padding: '14px 24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
};

export default function App() {
  const [active, setActive] = useState('overview');
  const Page = NAV.find(n => n.id === active)?.component || Overview;

  return (
    <div style={STYLES.app}>
      <aside style={STYLES.sidebar}>
        <div style={STYLES.logo}>FALCON</div>
        {NAV.map(n => (
          <div key={n.id}
               style={STYLES.navItem(active === n.id)}
               onClick={() => setActive(n.id)}>
            {n.label}
          </div>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{ padding: '0 20px', color: '#334155', fontSize: 11 }}>
          picograd v0.1.0
        </div>
      </aside>

      <main style={STYLES.main}>
        <div style={STYLES.topbar}>
          <div style={{ color: '#e2e8f0', fontWeight: 600 }}>
            {NAV.find(n => n.id === active)?.label}
          </div>
          <div style={{ color: '#475569', fontSize: 12 }}>
            FALCON — Federated Adaptive Learning
          </div>
        </div>
        <Page />
      </main>
    </div>
  );
}
