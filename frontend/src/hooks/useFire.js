import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchFires, fetchMonthlyFireStats, fetchFireSummary } from '../services/fireApi';

/**
 * useFire — fetches FIRMS hotspot points, monthly stats, and fire summary.
 * Re-fetches whenever date, minFRP, or selectedState changes.
 * Client-side FRP filter exposed as `fires` (no extra re-fetch needed).
 */
export function useFire(analysisDate, minFRP = 0, selectedState = 'All India') {
  const [allFires, setAllFires] = useState([]);
  const [monthly,  setMonthly]  = useState([]);
  const [summary,  setSummary]  = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);

  const requestId = useRef(0);
  const year = analysisDate?.slice(0, 4) || new Date().getFullYear().toString();

  const load = useCallback(async () => {
    const myId = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const [fireData, monthlyData, summaryData] = await Promise.all([
        fetchFires(analysisDate, 0),          // Fetch ALL, filter client-side for instant slider
        fetchMonthlyFireStats(year),
        fetchFireSummary(analysisDate),
      ]);
      if (myId !== requestId.current) return;
      setAllFires(fireData    ?? []);
      setMonthly(monthlyData  ?? []);
      setSummary(summaryData  ?? null);
    } catch (err) {
      if (myId !== requestId.current) return;
      setError(err?.response?.data?.message || err.message || 'Fire data fetch failed');
    } finally {
      if (myId === requestId.current) setLoading(false);
    }
  }, [analysisDate, selectedState, year]);

  useEffect(() => { load(); }, [load]);

  // Instant client-side FRP filter — no re-fetch
  const fires = allFires.filter(f => f.frp >= minFRP);

  return { fires, allFires, monthly, summary, loading, error, refetch: load };
}
