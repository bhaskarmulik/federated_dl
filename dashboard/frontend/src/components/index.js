// dashboard/frontend/src/components/index.js
// =============================================
// All reusable components for the FALCON dashboard.

import React, { useRef, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar,
} from 'recharts';


// ── StatusCard ────────────────────────────────────────────────────────────

export function StatusCard({ title, value, sub, color = '#6366f1', icon }) {
  return (
    <div style={{
      background: '#1e293b', border: '1px solid #334155',
      borderRadius: 12, padding: '20px 24px',
      display: 'flex', flexDirection: 'column', gap: 4, minWidth: 160,
    }}>
      <div style={{ color: '#94a3b8', fontSize: 13, fontWeight: 500 }}>{title}</div>
      <div style={{ color, fontSize: 28, fontWeight: 700 }}>{value ?? '—'}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 12 }}>{sub}</div>}
    </div>
  );
}


// ── TrainingChart ─────────────────────────────────────────────────────────

export function TrainingChart({ rounds = [], height = 280 }) {
  const data = rounds.map(r => ({
    round: r.round,
    loss:  r.global_loss != null ? +r.global_loss.toFixed(4) : null,
    acc:   r.global_acc  != null ? +r.global_acc.toFixed(4)  : null,
  }));

  return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 20,
                  border: '1px solid #334155' }}>
      <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 12 }}>
        Training Progress
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #334155',
                            borderRadius: 8, color: '#e2e8f0' }}
          />
          <Legend />
          <Line type="monotone" dataKey="loss" stroke="#6366f1"
                strokeWidth={2} dot={false} name="Global Loss" />
          {data.some(d => d.acc) && (
            <Line type="monotone" dataKey="acc" stroke="#10b981"
                  strokeWidth={2} dot={false} name="Accuracy" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}


// ── ClientCard ────────────────────────────────────────────────────────────

export function ClientCard({ client }) {
  const { client_id, status, online, train_loss, epsilon, round, n_samples } = client;
  const statusColor = online ? '#10b981' : '#ef4444';

  return (
    <div style={{
      background: '#1e293b', border: `1px solid ${online ? '#334155' : '#7f1d1d'}`,
      borderRadius: 10, padding: '16px 20px', minWidth: 200,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ color: '#e2e8f0', fontWeight: 600, fontSize: 14 }}>
          {client_id}
        </div>
        <div style={{
          background: statusColor + '22', color: statusColor,
          padding: '2px 8px', borderRadius: 20, fontSize: 11, fontWeight: 600,
        }}>
          {online ? 'online' : 'offline'}
        </div>
      </div>
      <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[
          ['Round',    round      ?? '—'],
          ['Loss',     train_loss != null ? train_loss.toFixed(4) : '—'],
          ['ε spent',  epsilon    != null ? epsilon.toFixed(3)    : '—'],
          ['Samples',  n_samples  ?? '—'],
        ].map(([k, v]) => (
          <div key={k}>
            <div style={{ color: '#64748b', fontSize: 11 }}>{k}</div>
            <div style={{ color: '#cbd5e1', fontSize: 13, fontWeight: 500 }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}


// ── PrivacyGauge ──────────────────────────────────────────────────────────

export function PrivacyGauge({ epsilon = 0, target = 8.0 }) {
  const pct     = Math.min(epsilon / target, 1);
  const degrees = pct * 180;
  const r = 70, cx = 100, cy = 100;

  // SVG arc path for gauge
  const polarToXY = (deg) => {
    const rad = (deg - 180) * Math.PI / 180;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  };
  const [sx, sy] = polarToXY(0);
  const [ex, ey] = polarToXY(degrees);
  const large    = degrees > 180 ? 1 : 0;

  const bgArc   = `M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`;
  const fillArc = degrees > 0
    ? `M ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`
    : null;

  const color = pct < 0.5 ? '#10b981' : pct < 0.8 ? '#f59e0b' : '#ef4444';

  return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 20,
                  border: '1px solid #334155', textAlign: 'center' }}>
      <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 8 }}>
        Privacy Budget
      </div>
      <svg width={200} height={120} viewBox="0 0 200 110">
        <path d={bgArc} fill="none" stroke="#334155" strokeWidth={14}
              strokeLinecap="round" />
        {fillArc && (
          <path d={fillArc} fill="none" stroke={color} strokeWidth={14}
                strokeLinecap="round" />
        )}
        <text x={cx} y={cy - 8} textAnchor="middle"
              fill={color} fontSize={22} fontWeight={700}>
          ε = {epsilon.toFixed(2)}
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle"
              fill="#64748b" fontSize={12}>
          target: {target.toFixed(1)}
        </text>
      </svg>
      <div style={{ color: '#64748b', fontSize: 12 }}>
        {(pct * 100).toFixed(0)}% budget consumed
      </div>
    </div>
  );
}


// ── GradCAMCard ───────────────────────────────────────────────────────────

export function GradCAMCard({ clientId, data }) {
  if (!data) return (
    <div style={{ background: '#1e293b', borderRadius: 10, padding: 20,
                  border: '1px solid #334155', color: '#64748b', textAlign: 'center' }}>
      No Grad-CAM data for {clientId}
    </div>
  );

  const { heatmap_b64, original_b64, anomaly_score } = data;
  const scoreColor = anomaly_score > 0.05 ? '#ef4444' : '#10b981';

  return (
    <div style={{ background: '#1e293b', borderRadius: 10, padding: 20,
                  border: '1px solid #334155' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginBottom: 12 }}>
        <div style={{ color: '#e2e8f0', fontWeight: 600 }}>
          Grad-CAM — {clientId}
        </div>
        <div style={{ color: scoreColor, fontWeight: 700, fontSize: 14 }}>
          score: {anomaly_score?.toFixed(4) ?? '—'}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        {original_b64 && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>Original</div>
            <img src={`data:image/png;base64,${original_b64}`}
                 alt="original" style={{ width: 112, height: 112, imageRendering: 'pixelated',
                 borderRadius: 6, border: '1px solid #334155' }} />
          </div>
        )}
        {heatmap_b64 && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>Anomaly Map</div>
            <img src={`data:image/png;base64,${heatmap_b64}`}
                 alt="heatmap" style={{ width: 112, height: 112, imageRendering: 'pixelated',
                 borderRadius: 6, border: '1px solid #334155' }} />
          </div>
        )}
      </div>
    </div>
  );
}


// ── WeightHeatmap ─────────────────────────────────────────────────────────

export function WeightHeatmap({ heatmapData = {} }) {
  // heatmapData: {layer_name: [(round, delta), ...]}
  const layers = Object.keys(heatmapData).slice(0, 8);  // top 8 layers

  if (!layers.length) return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 20,
                  border: '1px solid #334155', color: '#64748b' }}>
      No weight data yet
    </div>
  );

  // Build recharts data: [{round, layer1, layer2, ...}, ...]
  const roundSet = new Set();
  layers.forEach(l => heatmapData[l].forEach(([r]) => roundSet.add(r)));
  const roundArr = Array.from(roundSet).sort((a,b) => a-b);
  const chartData = roundArr.map(r => {
    const entry = { round: r };
    layers.forEach(l => {
      const found = heatmapData[l].find(([rr]) => rr === r);
      entry[l.split('.')[0]] = found ? +found[1].toFixed(5) : null;
    });
    return entry;
  });

  const COLORS = ['#6366f1','#10b981','#f59e0b','#ef4444',
                  '#8b5cf6','#06b6d4','#ec4899','#84cc16'];

  return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 20,
                  border: '1px solid #334155' }}>
      <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 12 }}>
        Weight Delta per Layer
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 10 }} />
          <YAxis stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={v => v.toFixed(4)} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155',
                                   borderRadius: 8, color: '#e2e8f0', fontSize: 11 }} />
          {layers.map((l, i) => (
            <Line key={l} type="monotone" dataKey={l.split('.')[0]}
                  stroke={COLORS[i % COLORS.length]} strokeWidth={1.5}
                  dot={false} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}


// ── LogViewer ─────────────────────────────────────────────────────────────

export function LogViewer({ log = [] }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [log]);

  return (
    <div style={{ background: '#0f172a', borderRadius: 10, padding: 16,
                  border: '1px solid #334155' }}>
      <div style={{ color: '#94a3b8', fontWeight: 600, fontSize: 13,
                    marginBottom: 8 }}>Event Log</div>
      <div ref={ref} style={{ height: 180, overflowY: 'auto',
                               fontFamily: 'monospace', fontSize: 11 }}>
        {log.map((e, i) => {
          const ts  = new Date(e.ts * 1000).toLocaleTimeString();
          const col = e.level === 'error' ? '#ef4444' :
                      e.level === 'warn'  ? '#f59e0b' : '#94a3b8';
          return (
            <div key={i} style={{ color: col, lineHeight: 1.6 }}>
              <span style={{ color: '#475569' }}>{ts} </span>
              {e.message}
            </div>
          );
        })}
      </div>
    </div>
  );
}


// ── DockerLaunchModal ─────────────────────────────────────────────────────

export function DockerLaunchModal({ onLaunch, onClose }) {
  const [form, setForm] = React.useState({
    client_id: `client_${Date.now()}`,
    dataset:   'BreastMNIST',
    alpha:     0.5,
  });

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
    }}>
      <div style={{ background: '#1e293b', borderRadius: 16, padding: 32,
                    border: '1px solid #334155', width: 380 }}>
        <div style={{ color: '#e2e8f0', fontSize: 18, fontWeight: 700, marginBottom: 20 }}>
          Launch FL Client
        </div>
        {[
          ['Client ID',   'client_id', 'text'],
          ['Dataset',     'dataset',   'text'],
          ['Non-IID α',   'alpha',     'number'],
        ].map(([label, key, type]) => (
          <div key={key} style={{ marginBottom: 16 }}>
            <div style={{ color: '#94a3b8', fontSize: 13, marginBottom: 4 }}>{label}</div>
            <input type={type} value={form[key]}
              onChange={e => setForm(f => ({...f, [key]: e.target.value}))}
              style={{ width: '100%', background: '#0f172a', border: '1px solid #334155',
                       borderRadius: 6, padding: '8px 12px', color: '#e2e8f0',
                       fontSize: 14, boxSizing: 'border-box' }} />
          </div>
        ))}
        <div style={{ display: 'flex', gap: 12, marginTop: 24 }}>
          <button onClick={() => onLaunch(form)}
            style={{ flex: 1, background: '#6366f1', color: '#fff', border: 'none',
                     borderRadius: 8, padding: '10px 0', fontWeight: 600,
                     cursor: 'pointer', fontSize: 14 }}>
            Launch
          </button>
          <button onClick={onClose}
            style={{ flex: 1, background: '#334155', color: '#e2e8f0', border: 'none',
                     borderRadius: 8, padding: '10px 0', fontWeight: 600,
                     cursor: 'pointer', fontSize: 14 }}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
