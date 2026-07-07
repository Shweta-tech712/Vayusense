import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchHCHOHotspots, fetchHCHOGrid, fetchHCHOSeasonal } from '../services/hchoApi';

/**
 * useHCHO — fetches DBSCAN hotspots, TROPOMI grid, and seasonal statistics.
 * Re-fetches on any filter change (date, state, threshold).
 */
export function useHCHO(analysisDate, threshold = 2.0, selectedState = 'All India') {
  const [hotspots, setHotspots] = useState([]);
  const [grid,     setGrid]     = useState([]);
  const [seasonal, setSeasonal] = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);

  const requestId = useRef(0);
  const year = analysisDate?.slice(0, 4) || new Date().getFullYear().toString();

  const load = useCallback(async () => {
    const myId = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const [hsData, gridData, seaData] = await Promise.all([
        fetchHCHOHotspots(analysisDate, threshold),
        fetchHCHOGrid(analysisDate),
        fetchHCHOSeasonal(year),
      ]);
      if (myId !== requestId.current) return;
      setHotspots(hsData  ?? []);
      setGrid(gridData    ?? []);
      setSeasonal(seaData ?? []);
    } catch (err) {
      if (myId !== requestId.current) return;
      setError(err?.response?.data?.message || err.message || 'HCHO data fetch failed');
    } finally {
      if (myId === requestId.current) setLoading(false);
    }
  }, [analysisDate, threshold, selectedState, year]);

  useEffect(() => { load(); }, [load]);

  return { hotspots, grid, seasonal, loading, error, refetch: load };
}
