import axiosInstance from './axiosInstance';

// ─── Local Mock Fallbacks ────────────────────────────────────────────────────
const MOCK_STATIONS = [
  { station: 'Anand Vihar, Delhi',       latitude: 28.647, longitude: 77.315, pm25: 165, pm10: 280, no2: 45, o3: 62,  cpcb_aqi: 335, aqi_category: 'Very Poor'    },
  { station: 'Bandra, Mumbai',           latitude: 19.055, longitude: 72.842, pm25: 58,  pm10: 92,  no2: 28, o3: 38,  cpcb_aqi: 98,  aqi_category: 'Satisfactory' },
  { station: 'Hebbal, Bengaluru',        latitude: 13.035, longitude: 77.598, pm25: 35,  pm10: 62,  no2: 18, o3: 44,  cpcb_aqi: 62,  aqi_category: 'Satisfactory' },
  { station: 'Victoria, Kolkata',        latitude: 22.544, longitude: 88.342, pm25: 110, pm10: 185, no2: 38, o3: 51,  cpcb_aqi: 220, aqi_category: 'Poor'          },
  { station: 'Manali, Chennai',          latitude: 13.165, longitude: 80.263, pm25: 42,  pm10: 74,  no2: 15, o3: 35,  cpcb_aqi: 74,  aqi_category: 'Satisfactory' },
  { station: 'Sanathnagar, Hyderabad',   latitude: 17.456, longitude: 78.441, pm25: 72,  pm10: 115, no2: 24, o3: 48,  cpcb_aqi: 125, aqi_category: 'Moderate'     },
  { station: 'IGIMS, Patna',             latitude: 25.611, longitude: 85.093, pm25: 192, pm10: 320, no2: 52, o3: 75,  cpcb_aqi: 380, aqi_category: 'Very Poor'    },
  { station: 'Shivajinagar, Pune',       latitude: 18.531, longitude: 73.849, pm25: 65,  pm10: 105, no2: 22, o3: 40,  cpcb_aqi: 108, aqi_category: 'Moderate'     },
  { station: 'Sector 22, Chandigarh',    latitude: 30.733, longitude: 76.789, pm25: 135, pm10: 220, no2: 40, o3: 58,  cpcb_aqi: 280, aqi_category: 'Poor'          },
  { station: 'Civil Lines, Lucknow',     latitude: 26.847, longitude: 80.947, pm25: 178, pm10: 295, no2: 48, o3: 68,  cpcb_aqi: 355, aqi_category: 'Very Poor'    },
  { station: 'Palasia, Indore',          latitude: 22.719, longitude: 75.857, pm25: 88,  pm10: 145, no2: 30, o3: 42,  cpcb_aqi: 155, aqi_category: 'Moderate'     },
  { station: 'New Rajendra Nagar, Bhopal', latitude: 23.259, longitude: 77.413, pm25: 95, pm10: 158, no2: 33, o3: 50, cpcb_aqi: 172, aqi_category: 'Moderate'   },
];

const generateMockTrend = (date, avgAQI) =>
  Array.from({ length: 14 }, (_, i) => {
    const d = new Date(new Date(date).getTime() - (13 - i) * 864e5);
    return {
      date: d.toISOString().slice(5, 10),
      observed_aqi: Math.round(avgAQI + Math.sin(i) * 22 + (Math.random() * 14 - 7)),
      predicted_aqi: Math.round(avgAQI + Math.sin(i) * 20 + (Math.random() * 18 - 9)),
    };
  });

// ─── API Functions ────────────────────────────────────────────────────────────

/**
 * Fetch CPCB monitoring station observations for a given date.
 * GET /api/stations?date=YYYY-MM-DD
 */
export async function fetchStations(date) {
  try {
    return await axiosInstance.get('/stations', { params: { date } });
  } catch {
    console.warn('[aqiApi] Backend offline – serving mock station data.');
    return MOCK_STATIONS;
  }
}

/**
 * Fetch satellite-derived AQI predictions grid for a given date.
 * GET /api/aqi/predict?date=YYYY-MM-DD
 */
export async function fetchAQIPredictions(date) {
  try {
    return await axiosInstance.get('/aqi/predict', { params: { date } });
  } catch {
    console.warn('[aqiApi] Backend offline – serving mock predictions.');
    return MOCK_STATIONS.map((s) => ({
      ...s,
      satellite_aqi: Math.round(s.cpcb_aqi * (0.9 + Math.random() * 0.2)),
    }));
  }
}

/**
 * Fetch 14-day AQI time-series trend for a station or all-India.
 * GET /api/aqi/trend?date=YYYY-MM-DD&station=<name>
 */
export async function fetchAQITrend(date, station = 'all') {
  try {
    return await axiosInstance.get('/aqi/trend', { params: { date, station } });
  } catch {
    const avg = 180;
    return generateMockTrend(date, avg);
  }
}
