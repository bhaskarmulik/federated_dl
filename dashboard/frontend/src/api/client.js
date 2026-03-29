// dashboard/frontend/src/api/client.js
// =======================================
// Axios wrapper for the FALCON dashboard REST API.

import axios from 'axios';

const BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: `${BASE}/api`,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
});

// ── REST helpers ──────────────────────────────────────────────────────────

export const fetchStatus     = ()      => api.get('/status').then(r => r.data);
export const fetchSnapshot   = ()      => api.get('/snapshot').then(r => r.data);
export const fetchClients    = ()      => api.get('/clients').then(r => r.data.clients);
export const fetchRounds     = (n=50)  => api.get(`/rounds?n=${n}`).then(r => r.data.rounds);
export const fetchLatestRound= ()      => api.get('/rounds/latest').then(r => r.data);
export const fetchWeights    = ()      => api.get('/weights').then(r => r.data.heatmap);
export const fetchPrivacy    = ()      => api.get('/privacy').then(r => r.data);
export const fetchGradCAM    = (id)    => api.get(`/gradcam/${id}`).then(r => r.data);
export const fetchAllGradCAM = ()      => api.get('/gradcam').then(r => r.data.gradcam);
export const fetchContainers = ()      => api.get('/docker/containers').then(r => r.data.containers);
export const fetchConfig     = ()      => api.get('/config').then(r => r.data);
export const fetchLog        = (n=50)  => api.get(`/log?n=${n}`).then(r => r.data.log);

export const launchContainer = (cfg)   => api.post('/docker/launch', cfg).then(r => r.data);
export const stopContainer   = (id)    => api.post(`/docker/stop/${id}`).then(r => r.data);
export const updateConfig    = (cfg)   => api.post('/config', cfg).then(r => r.data);

export default api;
