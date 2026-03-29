// dashboard/frontend/src/hooks/useWebSocket.js
import { useEffect, useRef, useState, useCallback } from 'react';
import { getSocket } from '../api/websocket';

/**
 * Subscribe to one or more WebSocket events.
 * Returns the latest value received for each event.
 *
 * Usage:
 *   const { round_complete, client_update } = useWebSocket(
 *     ['round_complete', 'client_update']
 *   );
 */
export function useWebSocket(events = []) {
  const [data, setData] = useState({});
  const socket = getSocket();

  useEffect(() => {
    const handlers = {};
    events.forEach(evt => {
      handlers[evt] = (payload) => setData(prev => ({ ...prev, [evt]: payload }));
      socket.on(evt, handlers[evt]);
    });
    return () => {
      events.forEach(evt => socket.off(evt, handlers[evt]));
    };
  }, [socket, events.join(',')]);   // eslint-disable-line

  return data;
}

/**
 * useRounds — live training loss/accuracy chart data.
 * Merges WebSocket updates with initial REST data.
 */
export function useRounds(initialRounds = []) {
  const [rounds, setRounds] = useState(initialRounds);
  const socket = getSocket();

  useEffect(() => {
    const handler = (payload) => {
      setRounds(prev => {
        const exists = prev.some(r => r.round === payload.round);
        if (exists) return prev.map(r => r.round === payload.round ? payload : r);
        return [...prev, payload];
      });
    };
    socket.on('round_complete', handler);
    return () => socket.off('round_complete', handler);
  }, [socket]);

  return rounds;
}

/**
 * useClients — live client list.
 */
export function useClients(initialClients = []) {
  const [clients, setClients] = useState(initialClients);
  const socket = getSocket();

  useEffect(() => {
    const handler = (payload) => {
      if (Array.isArray(payload)) setClients(payload);
    };
    socket.on('client_update', handler);
    return () => socket.off('client_update', handler);
  }, [socket]);

  return clients;
}
