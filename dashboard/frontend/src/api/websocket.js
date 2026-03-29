// dashboard/frontend/src/api/websocket.js
// ==========================================
// Socket.IO client setup for live FL metrics streaming.

import { io } from 'socket.io-client';

const SOCKET_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

let socket = null;

export function getSocket() {
  if (!socket) {
    socket = io(SOCKET_URL, {
      transports:     ['websocket', 'polling'],
      reconnection:   true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 10,
    });

    socket.on('connect',    () => console.log('[WS] connected'));
    socket.on('disconnect', () => console.log('[WS] disconnected'));
    socket.on('error',  (e) => console.error('[WS] error:', e));
  }
  return socket;
}

export function disconnectSocket() {
  if (socket) { socket.disconnect(); socket = null; }
}
