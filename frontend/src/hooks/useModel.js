import { useState, useEffect, useCallback } from 'react';
import { fetchModelMetrics, fetchLossCurve, fetchResiduals } from '../services/modelApi';

/**
 * useModel — fetches ML model validation data: scalar metrics, loss curves, and
 * observed-vs-predicted residuals. Results are cached at the API layer.
 */
export function useModel() {
  const [metrics,   setMetrics]   = useState(null);
  const [lossCurve, setLossCurve] = useState(null);
  const [residuals, setResiduals] = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [metricsData, curveData, residData] = await Promise.all([
        fetchModelMetrics(),
        fetchLossCurve(),
        fetchResiduals(250),
      ]);
      setMetrics(metricsData   ?? null);
      setLossCurve(curveData   ?? null);
      setResiduals(residData   ?? []);
    } catch (err) {
      setError(err?.response?.data?.message || err.message || 'Model data fetch failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  return { metrics, lossCurve, residuals, loading, error, refetch: load };
}
