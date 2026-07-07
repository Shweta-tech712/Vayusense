import axiosInstance from './axiosInstance';

// ─── Local Mock Fallbacks ────────────────────────────────────────────────────
const MOCK_HOTSPOTS = [
  {
    cluster_id: 0,
    label: 'Punjab Crop Residue Belt',
    point_count: 18,
    fire_count: 24,
    cumulative_frp: 850.5,
    mean_hcho: 8.42,
    coordinates: [[30.2, 74.5], [30.9, 74.8], [31.2, 75.5], [30.8, 76.2], [30.1, 75.8], [30.2, 74.5]],
  },
  {
    cluster_id: 1,
    label: 'Central India Forests',
    point_count: 8,
    fire_count: 12,
    cumulative_frp: 340.2,
    mean_hcho: 4.15,
    coordinates: [[21.5, 80.2], [22.1, 80.5], [22.4, 81.2], [21.8, 81.5], [21.3, 80.8], [21.5, 80.2]],
  },
  {
    cluster_id: 2,
    label: 'Ganga Plain Industrial',
    point_count: 12,
    fire_count: 6,
    cumulative_frp: 210.8,
    mean_hcho: 6.78,
    coordinates: [[25.8, 82.1], [26.4, 82.6], [26.7, 83.4], [26.2, 83.8], [25.6, 83.0], [25.8, 82.1]],
  },
];

const MOCK_HCHO_GRID = Array.from({ length: 120 }, (_, i) => ({
  latitude:  10 + Math.random() * 27,
  longitude: 68 + Math.random() * 28,
  hcho_vcd:  Math.random() * 12,          // vertical column density ×10¹⁵ mol/cm²
  quality_flag: Math.random() > 0.15 ? 1 : 0,
})).filter((p) => p.quality_flag === 1);

// ─── API Functions ────────────────────────────────────────────────────────────

/**
 * Fetch DBSCAN-derived HCHO hotspot clusters for a given date.
 * GET /api/hotspots?date=YYYY-MM-DD&threshold=<float>
 */
export async function fetchHCHOHotspots(date, threshold = 2.0) {
  try {
    return await axiosInstance.get('/hotspots', { params: { date, threshold } });
  } catch {
    console.warn('[hchoApi] Backend offline – serving mock hotspot clusters.');
    return MOCK_HOTSPOTS.filter((h) => h.mean_hcho >= threshold);
  }
}

/**
 * Fetch raw TROPOMI HCHO grid points for a given date.
 * GET /api/hcho/grid?date=YYYY-MM-DD
 */
export async function fetchHCHOGrid(date) {
  try {
    return await axiosInstance.get('/hcho/grid', { params: { date } });
  } catch {
    console.warn('[hchoApi] Backend offline – serving mock HCHO grid.');
    return MOCK_HCHO_GRID;
  }
}

/**
 * Fetch HCHO monthly seasonal statistics.
 * GET /api/hcho/seasonal?year=YYYY
 */
export async function fetchHCHOSeasonal(year) {
  try {
    return await axiosInstance.get('/hcho/seasonal', { params: { year } });
  } catch {
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return months.map((m, i) => ({
      month: m,
      mean_vcd: +(3 + Math.sin(i * 0.6) * 2.5 + Math.random()).toFixed(2),
      anomaly:  +(Math.random() * 2 - 1).toFixed(2),
    }));
  }
}
