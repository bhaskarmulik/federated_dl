// dashboard/frontend/src/pages/Config.jsx
import React, { useEffect, useState } from 'react';
import { fetchConfig, updateConfig } from '../api/client';

export default function Config() {
  const [cfg, setCfg] = useState({
    strategy: 'fedavg', secure_agg: false, dp_enabled: false,
    total_rounds: 10, noise_multiplier: 1.0, max_grad_norm: 1.0,
    learning_rate: 1e-3, local_epochs: 1,
  });
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetchConfig().then(c => setCfg(prev => ({...prev, ...c}))).catch(console.error);
  }, []);

  const save = () => {
    updateConfig(cfg)
      .then(() => { setSaved(true); setTimeout(() => setSaved(false), 2000); })
      .catch(console.error);
  };

  const Field = ({ label, k, type='text', options=null }) => (
    <div style={{ marginBottom: 18 }}>
      <div style={{ color: '#94a3b8', fontSize: 13, marginBottom: 5 }}>{label}</div>
      {options ? (
        <select value={cfg[k]} onChange={e => setCfg(c => ({...c, [k]: e.target.value}))}
          style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6,
                   padding: '8px 12px', color: '#e2e8f0', fontSize: 14, width: '100%' }}>
          {options.map(o => <option key={o}>{o}</option>)}
        </select>
      ) : type === 'checkbox' ? (
        <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
          <input type="checkbox" checked={cfg[k]}
                 onChange={e => setCfg(c => ({...c, [k]: e.target.checked}))}
                 style={{ width: 16, height: 16 }} />
          <span style={{ color: '#e2e8f0', fontSize: 14 }}>
            {cfg[k] ? 'Enabled' : 'Disabled'}
          </span>
        </label>
      ) : (
        <input type={type} value={cfg[k]}
               onChange={e => setCfg(c => ({...c, [k]: type==='number' ? +e.target.value : e.target.value}))}
               style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6,
                        padding: '8px 12px', color: '#e2e8f0', fontSize: 14, width: '100%',
                        boxSizing: 'border-box' }} />
      )}
    </div>
  );

  return (
    <div style={{ padding: 24 }}>
      <div style={{ color: '#e2e8f0', fontSize: 20, fontWeight: 700, marginBottom: 24 }}>
        Configuration
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32,
                    maxWidth: 800 }}>
        {/* FL Settings */}
        <div style={{ background: '#1e293b', borderRadius: 12, padding: 24,
                      border: '1px solid #334155' }}>
          <div style={{ color: '#6366f1', fontWeight: 600, marginBottom: 16 }}>
            Federated Learning
          </div>
          <Field label="Aggregation Strategy" k="strategy"
                 options={['fedavg','fedprox','fedbn']} />
          <Field label="Total Rounds" k="total_rounds" type="number" />
          <Field label="Local Epochs" k="local_epochs" type="number" />
          <Field label="Learning Rate" k="learning_rate" type="number" />
          <Field label="Secure Aggregation" k="secure_agg" type="checkbox" />
        </div>

        {/* DP Settings */}
        <div style={{ background: '#1e293b', borderRadius: 12, padding: 24,
                      border: '1px solid #334155' }}>
          <div style={{ color: '#10b981', fontWeight: 600, marginBottom: 16 }}>
            Differential Privacy
          </div>
          <Field label="Enable DP" k="dp_enabled" type="checkbox" />
          <Field label="Noise Multiplier (σ)" k="noise_multiplier" type="number" />
          <Field label="Max Grad Norm (C)" k="max_grad_norm" type="number" />
        </div>
      </div>

      <button onClick={save} style={{
        marginTop: 24, background: saved ? '#10b981' : '#6366f1',
        color: '#fff', border: 'none', borderRadius: 8, padding: '12px 32px',
        fontWeight: 700, fontSize: 15, cursor: 'pointer', transition: 'background 0.2s',
      }}>
        {saved ? '✓ Saved' : 'Save Configuration'}
      </button>
    </div>
  );
}
