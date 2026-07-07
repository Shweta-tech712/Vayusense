import axiosInstance from './axiosInstance';

// ─── Local Mock Fallbacks ────────────────────────────────────────────────────
const generateMockWindVectors = () => {
  const vectors = [];
  for (let lat = 8; lat <= 38; lat += 3) {
    for (let lon = 68; lon <= 97; lon += 3) {
      const angle = (lat * 0.3 + lon * 0.1) % (2 * Math.PI);
      const speed = 3.5 + Math.random() * 5.5;
      vectors.push({
        latitude:  lat,
        longitude: lon,
        u:         +(speed * Math.cos(angle)).toFixed(3),
        v:         +(speed * Math.sin(angle)).toFixed(3),
        speed:     +speed.toFixed(2),
        direction: +((Math.atan2(speed * Math.sin(angle), speed * Math.cos(angle)) * 180) / Math.PI).toFixed(1),
        pressure_level: 850,  // hPa boundary layer
      });
    }
  }
  return vectors;
};

const MOCK_TRAJECTORY = [
  [31.20, 74.80],   // Origin – Punjab burning zone
  [30.85, 75.35],
  [30.40, 75.90],
  [29.90, 76.45],
  [29.35, 77.00],
  [28.85, 77.25],   // Arrives over Delhi
];

// ─── API Functions ────────────────────────────────────────────────────────────

/**
 * Fetch ERA5 boundary-layer wind vector field for a given date.
 * GET /api/winds?date=YYYY-MM-DD&pressure=850
 */
export async function fetchWindVectors(date, pressure = 850) {
  try {
    return await axiosInstance.get('/winds', { params: { date, pressure } });
  } catch {
    console.warn('[transportApi] Backend offline – serving mock ERA5 wind field.');
    return generateMockWindVectors();
  }
}

/**
 * Fetch HYSPLIT-style backward air-mass trajectory path.
 * GET /api/trajectory?date=YYYY-MM-DD&lat=<float>&lon=<float>&hours=72
 */
export async function fetchTrajectory(date, lat = 28.65, lon = 77.22, hours = 72) {
  try {
    return await axiosInstance.get('/trajectory', { params: { date, lat, lon, hours } });
  } catch {
    console.warn('[transportApi] Backend offline – serving mock trajectory path.');
    return MOCK_TRAJECTORY;
  }
}

/**
 * Fetch advection statistics (mean transport direction, plume residence).
 * GET /api/transport/stats?date=YYYY-MM-DD
 */
export async function fetchTransportStats(date) {
  try {
    return await axiosInstance.get('/transport/stats', { params: { date } });
  } catch {
    return {
      mean_wind_speed:  +(4.2 + Math.random() * 2.5).toFixed(1),
      dominant_direction: 'WNW',
      mixing_height_m:  1200 + Math.round(Math.random() * 400),
      transport_distance_km: 350 + Math.round(Math.random() * 150),
    };
  }
}
