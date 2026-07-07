import axiosInstance from './axiosInstance';

// ─── Local Mock Fallbacks ────────────────────────────────────────────────────
const generateMockFires = () => {
  const fires = [];
  // Punjab / Haryana crop residue burning belt
  for (let i = 0; i < 38; i++) {
    fires.push({
      latitude:   29.4 + Math.random() * 2.2,
      longitude:  73.8 + Math.random() * 3.4,
      frp:        15 + Math.random() * 195,
      confidence: Math.floor(70 + Math.random() * 30),
      acq_date:   new Date().toISOString().slice(0, 10),
      satellite:  'VIIRS',
    });
  }
  // Central India / Chhattisgarh forest belt
  for (let i = 0; i < 18; i++) {
    fires.push({
      latitude:   19.2 + Math.random() * 3.8,
      longitude:  77.8 + Math.random() * 5.5,
      frp:        10 + Math.random() * 80,
      confidence: Math.floor(60 + Math.random() * 35),
      acq_date:   new Date().toISOString().slice(0, 10),
      satellite:  'MODIS',
    });
  }
  // Northeast India
  for (let i = 0; i < 12; i++) {
    fires.push({
      latitude:   24.5 + Math.random() * 3.5,
      longitude:  90.0 + Math.random() * 6.0,
      frp:        8 + Math.random() * 55,
      confidence: Math.floor(55 + Math.random() * 40),
      acq_date:   new Date().toISOString().slice(0, 10),
      satellite:  'MODIS',
    });
  }
  return fires;
};

const MOCK_MONTHLY = [
  { month: 'Jul', baseline: 120,  current: 85   },
  { month: 'Aug', baseline: 180,  current: 140  },
  { month: 'Sep', baseline: 420,  current: 310  },
  { month: 'Oct', baseline: 2500, current: 1980 },
  { month: 'Nov', baseline: 4800, current: 3950 },
  { month: 'Dec', baseline: 920,  current: 640  },
];

// ─── API Functions ────────────────────────────────────────────────────────────

/**
 * Fetch active fire hotspot points (NASA FIRMS) for a given date.
 * GET /api/fires?date=YYYY-MM-DD&min_frp=<float>
 */
export async function fetchFires(date, minFRP = 0) {
  try {
    return await axiosInstance.get('/fires', { params: { date, min_frp: minFRP } });
  } catch {
    console.warn('[fireApi] Backend offline – serving mock FIRMS data.');
    return generateMockFires().filter((f) => f.frp >= minFRP);
  }
}

/**
 * Fetch monthly fire count statistics for seasonal trend charts.
 * GET /api/fires/monthly?year=YYYY
 */
export async function fetchMonthlyFireStats(year) {
  try {
    return await axiosInstance.get('/fires/monthly', { params: { year } });
  } catch {
    console.warn('[fireApi] Backend offline – serving mock monthly stats.');
    return MOCK_MONTHLY;
  }
}

/**
 * Fetch fire summary aggregates (total, max FRP, by satellite).
 * GET /api/fires/summary?date=YYYY-MM-DD
 */
export async function fetchFireSummary(date) {
  try {
    return await axiosInstance.get('/fires/summary', { params: { date } });
  } catch {
    const fires = generateMockFires();
    return {
      total: fires.length,
      max_frp: Math.max(...fires.map((f) => f.frp)).toFixed(1),
      modis_count: fires.filter((f) => f.satellite === 'MODIS').length,
      viirs_count: fires.filter((f) => f.satellite === 'VIIRS').length,
    };
  }
}
