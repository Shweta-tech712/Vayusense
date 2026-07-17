import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Response interceptor to log errors
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error(`API call failed: ${error.message}`);
    return Promise.reject(error);
  }
);

// ----------------------------------------------------------------------
// API SERVICE INTERFACES
// ----------------------------------------------------------------------
export const apiService = {
  
  async getStations(date) {
    return await apiClient.get(`/stations?date=${date}`);
  },

  async getFires(date) {
    return await apiClient.get(`/fires?date=${date}`);
  },

  async getHCHOHotspots(date) {
    return await apiClient.get(`/hotspots?date=${date}`);
  },

  async getWindVectors(date) {
    return await apiClient.get(`/winds?date=${date}`);
  },

  async getAdvectionTrajectory(date) {
    return await apiClient.get(`/trajectory?date=${date}`);
  },

  async getValidationMetrics() {
    return await apiClient.get('/model-performance');
  }
};
