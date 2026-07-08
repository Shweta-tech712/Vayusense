import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

// Simple in-memory cache for GET requests
const requestCache = new Map();
const CACHE_TTL = 30000; // 30 seconds cache TTL

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// ----------------------------------------------------------------------
// 1. CACHE INTERCEPTOR
// ----------------------------------------------------------------------
axiosInstance.interceptors.request.use((config) => {
  if (config.method?.toLowerCase() === 'get' && config.cache !== false) {
    const cacheKey = `${config.url}?${JSON.stringify(config.params || '')}`;
    const cachedEntry = requestCache.get(cacheKey);

    if (cachedEntry) {
      const isExpired = (Date.now() - cachedEntry.timestamp) > CACHE_TTL;
      if (!isExpired) {
        // Return resolved cached promise
        config.adapter = () => Promise.resolve({
          data: cachedEntry.data,
          status: 200,
          statusText: 'OK',
          headers: {},
          config
        });
      } else {
        requestCache.delete(cacheKey);
      }
    }
  }
  return config;
}, (error) => Promise.reject(error));

// Save responses in cache
axiosInstance.interceptors.response.use((response) => {
  const config = response.config;
  if (config.method?.toLowerCase() === 'get' && config.cache !== false && response.status === 200) {
    const cacheKey = `${config.url}?${JSON.stringify(config.params || '')}`;
    requestCache.set(cacheKey, {
      data: response.data,
      timestamp: Date.now()
    });
  }
  return response.data;
});

// ----------------------------------------------------------------------
// 2. RETRY INTERCEPTOR & ERROR TOASTS EMITTER
// ----------------------------------------------------------------------
axiosInstance.interceptors.response.use(
  (data) => data,
  async (error) => {
    const config = error.config;
    
    // Check if configuration exists and has retry enabled
    if (!config || !config.retry) {
      config.retry = 3;
      config.retryDelay = 1500;
      config.retryCount = config.retryCount || 0;
    }

    // Trigger retry if connection failed or server returned a 5xx error
    const isNetworkError = !error.response;
    const isServerError = error.response && error.response.status >= 500;

    if ((isNetworkError || isServerError) && config.retryCount < config.retry) {
      config.retryCount += 1;
      console.warn(`API Error: ${error.message}. Retrying request (${config.retryCount}/${config.retry})...`);
      
      // Delay before next attempt
      const backoffDelay = config.retryDelay * Math.pow(1.5, config.retryCount);
      await new Promise(resolve => setTimeout(resolve, backoffDelay));
      
      // Re-trigger request
      return axiosInstance(config);
    }

    // Global Error Warning Emitter (Dispatches custom event caught by Layout Toast hud)
    const toastMessage = error.response?.data?.message || error.message || 'System Connection Error';
    const toastEvent = new CustomEvent('show-toast', { detail: { message: toastMessage, type: 'error' } });
    window.dispatchEvent(toastEvent);

    return Promise.reject(error);
  }
);

export default axiosInstance;
