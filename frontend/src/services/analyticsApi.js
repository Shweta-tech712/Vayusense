import axiosInstance from './axiosInstance';

// ─── Local Mock Fallbacks ────────────────────────────────────────────────────
const generateScatterPoints = (n = 200) =>
  Array.from({ length: n }, () => {
    const pm25 = 20 + Math.random() * 280;
    return {
      pm25,
      satellite_aqi: Math.round(pm25 * 1.6 + (Math.random() * 40 - 20)),
      observed_aqi:  Math.round(pm25 * 1.55 + (Math.random() * 30 - 15)),
      no2:           +(5 + Math.random() * 60).toFixed(1),
      o3:            +(20 + Math.random() * 80).toFixed(1),
      month:         Math.ceil(Math.random() * 12),
    };
  });

const MOCK_CORRELATION = [
  { variable: 'PM2.5',   r: 0.91 },
  { variable: 'PM10',    r: 0.87 },
  { variable: 'NO2',     r: 0.74 },
  { variable: 'AOD',     r: 0.82 },
  { variable: 'HCHO',    r: 0.68 },
  { variable: 'Fire FRP',r: 0.72 },
  { variable: 'Humidity',r: -0.45 },
  { variable: 'Wind Speed', r: -0.61 },
];

const MOCK_AQI_DISTRIBUTION = [
  { range: '0–50',    count: 38, label: 'Good'        },
  { range: '51–100',  count: 72, label: 'Satisfactory'},
  { range: '101–200', count: 118, label: 'Moderate'   },
  { range: '201–300', count: 64,  label: 'Poor'       },
  { range: '301–400', count: 31,  label: 'Very Poor'  },
  { range: '401+',    count: 12,  label: 'Severe'     },
];

// ─── API Functions ────────────────────────────────────────────────────────────

/**
 * Fetch PM2.5 vs AQI scatter plot data for analytics view.
 * GET /api/analytics/scatter?date=YYYY-MM-DD&n=<int>
 */
export async function fetchScatterData(date, n = 200) {
  try {
    return await axiosInstance.get('/analytics/scatter', { params: { date, n } });
  } catch {
    console.warn('[analyticsApi] Backend offline – serving mock scatter data.');
    return generateScatterPoints(n);
  }
}

/**
 * Fetch feature-AQI Pearson correlation values for bar chart.
 * GET /api/analytics/correlation?date=YYYY-MM-DD
 */
export async function fetchCorrelation(date) {
  try {
    return await axiosInstance.get('/analytics/correlation', { params: { date } });
  } catch {
    console.warn('[analyticsApi] Backend offline – serving mock correlation.');
    return MOCK_CORRELATION;
  }
}

/**
 * Fetch AQI station count distribution across AQI ranges.
 * GET /api/analytics/distribution?date=YYYY-MM-DD
 */
export async function fetchAQIDistribution(date) {
  try {
    return await axiosInstance.get('/analytics/distribution', { params: { date } });
  } catch {
    return MOCK_AQI_DISTRIBUTION;
  }
}
