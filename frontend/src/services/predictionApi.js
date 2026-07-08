/**
 * src/services/predictionApi.js
 *
 * Thin Axios wrapper for the CNN-LSTM prediction endpoint.
 * Classifies error types so the UI can show specific messages.
 */
import axiosInstance from './axiosInstance';

// Error type constants for clear UI messaging
export const PREDICTION_ERRORS = {
  BACKEND_OFFLINE:    'BACKEND_OFFLINE',
  MODEL_UNAVAILABLE:  'MODEL_UNAVAILABLE',
  LOCATION_NOT_FOUND: 'LOCATION_NOT_FOUND',
  SATELLITE_MISSING:  'SATELLITE_MISSING',
  UNKNOWN:            'UNKNOWN',
};

/**
 * Call POST /api/predict/location with a city/state/lat-lon string.
 *
 * @param {string} location  City or state name, or "lat,lon" string
 * @param {number|null} latitude   Optional explicit latitude
 * @param {number|null} longitude  Optional explicit longitude
 * @returns {Promise<Object>}  Raw CNN-LSTM response
 * @throws {Object}  { type: PREDICTION_ERRORS.*, message: string }
 */
export async function getAQIPrediction(location, latitude = null, longitude = null, signal = null) {
  try {
    const payload = { location };
    if (latitude  != null) payload.latitude  = latitude;
    if (longitude != null) payload.longitude = longitude;

    // axiosInstance already has timeout, retry, and error interceptors
    const data = await axiosInstance.post('/predict/location', payload, { signal });
    return data;

  } catch (err) {
    if (err.name === 'CanceledError' || err.name === 'AbortError' || axiosInstance.isCancel?.(err)) {
      // Re-throw cancel/abort errors directly to let callers ignore or handle them
      throw err;
    }
    const status = err?.response?.status;
    const detail = err?.response?.data?.detail ?? err?.message ?? 'Unknown error';

    // Classify the error for the UI
    if (!err.response) {
      throw { type: PREDICTION_ERRORS.BACKEND_OFFLINE,
              message: 'Backend server is offline. Please start the FastAPI server on port 8000.' };
    }
    if (status === 503) {
      throw { type: PREDICTION_ERRORS.MODEL_UNAVAILABLE,
              message: 'CNN-LSTM model is not loaded. Run the training script and restart the server.' };
    }
    if (status === 422) {
      throw { type: PREDICTION_ERRORS.LOCATION_NOT_FOUND,
              message: `Location not found: "${location}". Try a known Indian city or state name.` };
    }
    if (status === 404) {
      throw { type: PREDICTION_ERRORS.SATELLITE_MISSING,
              message: 'Satellite data is unavailable for this region or date.' };
    }

    throw { type: PREDICTION_ERRORS.UNKNOWN,
            message: `Prediction failed: ${detail}` };
  }
}
