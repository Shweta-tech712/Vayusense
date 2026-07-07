import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchStations, fetchAQIPredictions, fetchAQITrend } from '../services/aqiApi';

/**
 * useAQI — fetches station data, predictions, and time-series trend.
 * Re-fetches whenever analysisDate, selectedState, or selectedPollutant changes.
 * Stale-request guard prevents race conditions when filters change rapidly.
 */
export function useAQI(analysisDate, selectedState = 'All India', selectedPollutant = 'PM2.5') {
  const [stations,    setStations]    = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [trend,       setTrend]       = useState([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);

  // Monotonic request token — cancels stale responses
  const requestId = useRef(0);

  const load = useCallback(async () => {
    const myId = ++requestId.current;
    setLoading(true);
    setError(null);

    try {
      const [stData, predData, trendData] = await Promise.all([
        fetchStations(analysisDate),
        fetchAQIPredictions(analysisDate),
        fetchAQITrend(analysisDate, selectedState !== 'All India' ? selectedState : 'all'),
      ]);

      if (myId !== requestId.current) return; // Discard stale response

      // Client-side state filter applied here so both map & table react instantly
      const applyStateFilter = (arr) =>
        !arr || selectedState === 'All India'
          ? (arr ?? [])
          : arr.filter(s =>
              (s.station ?? s.name ?? '').toLowerCase().includes(selectedState.toLowerCase())
            );

      setStations(applyStateFilter(stData)       ?? []);
      setPredictions(applyStateFilter(predData)  ?? []);
      setTrend(trendData                         ?? []);
    } catch (err) {
      if (myId !== requestId.current) return;
      setError(err?.response?.data?.message || err.message || 'AQI data fetch failed');
    } finally {
      if (myId === requestId.current) setLoading(false);
    }
  }, [analysisDate, selectedState, selectedPollutant]);

  useEffect(() => { load(); }, [load]);

  return { stations, predictions, trend, loading, error, refetch: load };
}
