// dashboard/frontend/src/pages/Explain.jsx
import React, { useEffect, useState } from 'react';
import { fetchAllGradCAM, fetchClients } from '../api/client';
import { useWebSocket } from '../hooks/useWebSocket';
import { GradCAMCard, StatusCard } from '../components';

export default function Explain() {
  const [gradcam,  setGradcam]  = useState({});
  const [clients,  setClients]  = useState([]);
  const [selected, setSelected] = useState(null);
  const { gradcam_update } = useWebSocket(['gradcam_update']);

  useEffect(() => {
    fetchAllGradCAM().then(setGradcam).catch(console.error);
    fetchClients().then(setClients).catch(console.error);
  }, []);

  useEffect(() => {
    if (gradcam_update) {
      const { client_id, ...data } = gradcam_update;
      setGradcam(prev => ({ ...prev, [client_id]: data }));
    }
  }, [gradcam_update]);

  const clientIds = Object.keys(gradcam);
  const activeId  = selected || clientIds[0];

  return (
    <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div style={{ color: '#e2e8f0', fontSize: 20, fontWeight: 700 }}>
        Grad-CAM Explainability
      </div>
      <div style={{ color: '#94a3b8', fontSize: 13 }}>
        Grad-CAM highlights image regions driving high reconstruction error — the
        anomalous regions detected by each hospital client's model.
      </div>

      {/* Client selector */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {clientIds.map(cid => (
          <button key={cid} onClick={() => setSelected(cid)}
            style={{
              background: cid === activeId ? '#6366f1' : '#1e293b',
              color: cid === activeId ? '#fff' : '#94a3b8',
              border: '1px solid #334155', borderRadius: 8,
              padding: '6px 16px', cursor: 'pointer', fontSize: 13,
            }}>
            {cid}
          </button>
        ))}
        {clientIds.length === 0 && (
          <div style={{ color: '#475569' }}>No Grad-CAM data available yet</div>
        )}
      </div>

      {/* Main GradCAM display */}
      {activeId && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px,1fr))', gap: 16 }}>
          <GradCAMCard clientId={activeId} data={gradcam[activeId]} />

          {/* Anomaly score breakdown */}
          <div style={{ background: '#1e293b', borderRadius: 10, padding: 20,
                        border: '1px solid #334155' }}>
            <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 12 }}>
              Score Breakdown
            </div>
            {Object.entries(gradcam).map(([cid, d]) => {
              const score = d?.anomaly_score ?? 0;
              const pct   = Math.min(score / 0.2 * 100, 100);
              const col   = score > 0.05 ? '#ef4444' : '#10b981';
              return (
                <div key={cid} style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between',
                                fontSize: 12, marginBottom: 3 }}>
                    <span style={{ color: '#94a3b8' }}>{cid}</span>
                    <span style={{ color: col, fontWeight: 600 }}>
                      {score.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ background: '#334155', borderRadius: 4, height: 6 }}>
                    <div style={{ width: `${pct}%`, background: col,
                                  borderRadius: 4, height: 6 }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
