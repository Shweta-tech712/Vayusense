import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchWindVectors, fetchTrajectory, fetchTransportStats } from '../services/transportApi';

/**
 * useTransport — fetches ERA5 wind field, HYSPLIT trajectory, and advection stats.
 * Re-fetches on date or selectedState change.
 */
export function useTransport(analysisDate, selectedState = 'All India') {
  const [windVectors, setWindVectors] = useState([]);
  const [trajectory,  setTrajectory]  = useState([]);
  const [stats,       setStats]       = useState(null);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);

  const requestId = useRef(0);

  const load = useCallback(async () => {
    const myId = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const [windData, trajData, statsData] = await Promise.all([
        fetchWindVectors(analysisDate),
        fetchTrajectory(analysisDate),
        fetchTransportStats(analysisDate),
      ]);
      if (myId !== requestId.current) return;
      setWindVectors(windData  ?? []);
      setTrajectory(trajData   ?? []);
      setStats(statsData       ?? null);
    } catch (err) {
      if (myId !== requestId.current) return;
      setError(err?.response?.data?.message || err.message || 'Transport data fetch failed');
    } finally {
      if (myId === requestId.current) setLoading(false);
    }
  }, [analysisDate, selectedState]);

  useEffect(() => { load(); }, [load]);

  return { windVectors, trajectory, stats, loading, error, refetch: load };
}
