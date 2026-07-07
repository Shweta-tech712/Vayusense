import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchScatterData, fetchCorrelation, fetchAQIDistribution } from '../services/analyticsApi';

/**
 * useAnalytics — fetches scatter data, feature correlation, and AQI distribution.
 * Re-fetches on date, state, or pollutant filter change.
 */
export function useAnalytics(analysisDate, selectedState = 'All India', selectedPollutant = 'PM2.5') {
  const [scatter,      setScatter]      = useState([]);
  const [correlation,  setCorrelation]  = useState([]);
  const [distribution, setDistribution] = useState([]);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);

  const requestId = useRef(0);

  const load = useCallback(async () => {
    const myId = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const [scData, corrData, distData] = await Promise.all([
        fetchScatterData(analysisDate),
        fetchCorrelation(analysisDate),
        fetchAQIDistribution(analysisDate),
      ]);
      if (myId !== requestId.current) return;
      setScatter(scData        ?? []);
      setCorrelation(corrData  ?? []);
      setDistribution(distData ?? []);
    } catch (err) {
      if (myId !== requestId.current) return;
      setError(err?.response?.data?.message || err.message || 'Analytics data fetch failed');
    } finally {
      if (myId === requestId.current) setLoading(false);
    }
  }, [analysisDate, selectedState, selectedPollutant]);

  useEffect(() => { load(); }, [load]);

  return { scatter, correlation, distribution, loading, error, refetch: load };
}
