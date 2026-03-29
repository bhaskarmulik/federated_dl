// dashboard/frontend/src/pages/Overview.jsx
import React, { useEffect, useState } from 'react';
import { fetchSnapshot } from '../api/client';
import { useRounds, useClients, useWebSocket } from '../hooks/useWebSocket';
import {
  StatusCard, TrainingChart, ClientCard, LogViewer, WeightHeatmap
} from '../components';

export default function Overview() {
  const [snap,    setSnap]    = useState(null);
  const liveRounds  = useRounds(snap?.rounds || []);
  const liveClients = useClients(snap?.clients || []);
  const { server_status, weight_update } = useWebSocket(['server_status','weight_update']);

  useEffect(() => {
    fetchSnapshot().then(setSnap).catch(console.error);
  }, []);

  const server  = server_status || snap?.server || {};
  const rounds  = liveRounds;
  const clients = liveClients;
  const log     = snap?.log || [];

  return (
    <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Status row */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <StatusCard title="Status"    value={server.status    || '—'} color="#10b981" />
        <StatusCard title="Round"     value={server.round     ?? '—'} color="#6366f1"
                    sub={server.total_rounds ? `of ${server.total_rounds}` : ''} />
        <StatusCard title="Clients"   value={clients.filter(c=>c.online).length}
                    sub={`${clients.length} registered`} color="#f59e0b" />
        <StatusCard title="Strategy"  value={server.strategy  || '—'} color="#06b6d4" />
        <StatusCard title="SecAgg"    value={server.secure_agg ? 'ON' : 'OFF'}
                    color={server.secure_agg ? '#10b981' : '#64748b'} />
        <StatusCard title="DP"        value={server.dp_enabled ? 'ON' : 'OFF'}
                    color={server.dp_enabled ? '#10b981' : '#64748b'} />
      </div>

      {/* Training chart */}
      <TrainingChart rounds={rounds} />

      {/* Clients grid */}
      <div>
        <div style={{ color: '#94a3b8', fontWeight: 600, marginBottom: 12 }}>
          Connected Clients
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {clients.length === 0
            ? <div style={{ color: '#475569' }}>No clients connected</div>
            : clients.map(c => <ClientCard key={c.client_id} client={c} />)
          }
        </div>
      </div>

      {/* Weight heatmap + log */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <WeightHeatmap heatmapData={snap?.weight_heatmap || {}} />
        <LogViewer log={log} />
      </div>
    </div>
  );
}
