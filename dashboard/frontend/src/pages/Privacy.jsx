// dashboard/frontend/src/pages/Privacy.jsx
import React, { useEffect, useState } from 'react';
import { fetchPrivacy } from '../api/client';
import { useWebSocket } from '../hooks/useWebSocket';
import { PrivacyGauge, StatusCard } from '../components';
import { LineChart, Line, XAxis, YAxis, CartesianGrid,
         Tooltip, ResponsiveContainer } from 'recharts';

export default function Privacy() {
  const [privacy, setPrivacy] = useState({ timeline: [], current_eps: 0 });
  const { privacy_update } = useWebSocket(['privacy_update']);

  useEffect(() => { fetchPrivacy().then(setPrivacy).catch(console.error); }, []);
  useEffect(() => {
    if (privacy_update) {
      setPrivacy(prev => ({
        current_eps: privacy_update.epsilon,
        timeline:    [...prev.timeline,
                      { round: privacy_update.round, epsilon: privacy_update.epsilon }],
      }));
    }
  }, [privacy_update]);

  return (
    <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <PrivacyGauge epsilon={privacy.current_eps} target={8.0} />
        <div style={{ flex: 1, minWidth: 300, background: '#1e293b',
                      borderRadius: 12, padding: 20, border: '1px solid #334155' }}>
          <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 12 }}>
            ε Timeline
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={privacy.timeline}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155',
                                       borderRadius: 8, color: '#e2e8f0' }} />
              <Line type="monotone" dataKey="epsilon" stroke="#f59e0b"
                    strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div style={{ background: '#1e293b', borderRadius: 12, padding: 20,
                    border: '1px solid #334155', color: '#94a3b8', fontSize: 13 }}>
        <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 12 }}>
          Privacy Accounting Method
        </div>
        <p>FALCON uses <strong style={{ color: '#6366f1' }}>Rényi Differential Privacy (RDP)</strong> accounting
        (Mironov 2017), which provides tighter privacy bounds than basic composition.
        The accumulated ε grows as O(σ²·T) rather than O(σ·√T), enabling more training
        rounds for the same privacy budget.</p>
        <p style={{ marginTop: 8 }}>
          Each FL round contributes ε according to the subsampled Gaussian mechanism.
          The privacy guarantee holds for all queries to the global model.
        </p>
      </div>
    </div>
  );
}
