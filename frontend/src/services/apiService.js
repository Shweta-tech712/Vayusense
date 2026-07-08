import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
const IS_PROD = import.meta.env.PROD === true;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Response interceptor to log errors and handle fallbacks
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.warn(`API call failed: ${error.message}. Serving local synthetic data fallback.`);
    return Promise.reject(error);
  }
);

// ----------------------------------------------------------------------
// LOCAL SYNTHETIC GENERATORS (Used when backend is offline)
// ----------------------------------------------------------------------
const generateMockStations = (date) => [
  { station: "Anand Vihar, Delhi", latitude: 28.647, longitude: 77.315, pm25: 165, pm10: 280, no2: 45, o3: 62, cpcb_aqi: 335, aqi_category: "Very Poor" },
  { station: "Bandra, Mumbai", latitude: 19.055, longitude: 72.842, pm25: 58, pm10: 92, no2: 28, o3: 38, cpcb_aqi: 98, aqi_category: "Satisfactory" },
  { station: "Hebbal, Bengaluru", latitude: 13.035, longitude: 77.598, pm25: 35, pm10: 62, no2: 18, o3: 44, cpcb_aqi: 62, aqi_category: "Satisfactory" },
  { station: "Victoria, Kolkata", latitude: 22.544, longitude: 88.342, pm25: 110, pm10: 185, no2: 38, o3: 51, cpcb_aqi: 220, aqi_category: "Poor" },
  { station: "Manali, Chennai", latitude: 13.165, longitude: 80.263, pm25: 42, pm10: 74, no2: 15, o3: 35, cpcb_aqi: 74, aqi_category: "Satisfactory" },
  { station: "Sanathnagar, Hyderabad", latitude: 17.456, longitude: 78.441, pm25: 72, pm10: 115, no2: 24, o3: 48, cpcb_aqi: 125, aqi_category: "Moderate" },
  { station: "IGIMS, Patna", latitude: 25.611, longitude: 85.093, pm25: 192, pm10: 320, no2: 52, o3: 75, cpcb_aqi: 380, aqi_category: "Very Poor" },
  { station: "Shivajinagar, Pune", latitude: 18.531, longitude: 73.849, pm25: 65, pm10: 105, no2: 22, o3: 40, cpcb_aqi: 108, aqi_category: "Moderate" }
];

const generateMockFires = () => {
  const fires = [];
  // Crop burning region (Punjab/Haryana)
  for (let i = 0; i < 35; i++) {
    fires.push({
      latitude: 29.5 + Math.random() * 2.0,
      longitude: 74.0 + Math.random() * 3.0,
      frp: 15.0 + Math.random() * 180.0,
      confidence: Math.floor(70 + Math.random() * 30)
    });
  }
  // Central Forest regions
  for (let i = 0; i < 15; i++) {
    fires.push({
      latitude: 19.5 + Math.random() * 3.5,
      longitude: 78.0 + Math.random() * 5.0,
      frp: 10.0 + Math.random() * 75.0,
      confidence: Math.floor(60 + Math.random() * 35)
    });
  }
  return fires;
};

const generateMockHotspots = () => {
  // DBSCAN cluster polygons coordinates bounding India crop burning regions
  return [
    {
      cluster_id: 0,
      point_count: 18,
      fire_count: 24,
      cumulative_frp: 850.5,
      coordinates: [
        [30.2, 74.5], [30.9, 74.8], [31.2, 75.5], [30.8, 76.2], [30.1, 75.8], [30.2, 74.5]
      ]
    },
    {
      cluster_id: 1,
      point_count: 8,
      fire_count: 12,
      cumulative_frp: 340.2,
      coordinates: [
        [21.5, 80.2], [22.1, 80.5], [22.4, 81.2], [21.8, 81.5], [21.3, 80.8], [21.5, 80.2]
      ]
    }
  ];
};

// ----------------------------------------------------------------------
// API SERVICE INTERFACES
// ----------------------------------------------------------------------
export const apiService = {
  
  async getStations(date) {
    try {
      return await apiClient.get(`/stations?date=${date}`);
    } catch (err) {
      if (IS_PROD) throw err;
      return generateMockStations(date);
    }
  },

  async getFires(date) {
    try {
      return await apiClient.get(`/fires?date=${date}`);
    } catch (err) {
      if (IS_PROD) throw err;
      return generateMockFires();
    }
  },

  async getHCHOHotspots(date) {
    try {
      return await apiClient.get(`/hotspots?date=${date}`);
    } catch (err) {
      if (IS_PROD) throw err;
      return generateMockHotspots();
    }
  },

  async getWindVectors(date) {
    try {
      return await apiClient.get(`/winds?date=${date}`);
    } catch (err) {
      if (IS_PROD) throw err;
      // Return sparse mesh of wind vectors (lat, lon, u, v, speed)
      const vectors = [];
      for (let lat = 10; lat <= 35; lat += 4) {
        for (let lon = 70; lon <= 95; lon += 4) {
          vectors.push({
            latitude: lat,
            longitude: lon,
            u: 3.5 + Math.random() * 2,
            v: -1.5 - Math.random() * 2,
            speed: 4.0 + Math.random() * 3,
            direction: 110 + Math.random() * 30
          });
        }
      }
      return vectors;
    }
  },

  async getAdvectionTrajectory(date) {
    try {
      return await apiClient.get(`/trajectory?date=${date}`);
    } catch (err) {
      if (IS_PROD) throw err;
      // Mock trajectory coords starting in Punjab going south-east towards Delhi
      return [
        [30.20, 74.80],
        [29.85, 75.35],
        [29.50, 75.90],
        [29.10, 76.40],
        [28.65, 77.30] // Arrives near Delhi
      ];
    }
  },

  async getValidationMetrics() {
    try {
      return await apiClient.get('/model/performance');
    } catch (err) {
      if (IS_PROD) throw err;
      return {
        r2: 0.842,
        mae: 18.54,
        rmse: 26.42,
        pearson: 0.895,
        epochs: Array.from({ length: 40 }, (_, i) => i + 1),
        train_loss: Array.from({ length: 40 }, (_, i) => 2500 * Math.exp(-(i + 1) / 8) + Math.random() * 15 + 120),
        val_loss: Array.from({ length: 40 }, (_, i) => 2650 * Math.exp(-(i + 1) / 9) + Math.random() * 18 + 145),
        residuals: Array.from({ length: 200 }, () => Math.random() * 50 - 25)
      };
    }
  }
};
