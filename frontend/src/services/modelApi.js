import axiosInstance from './axiosInstance';

// ─── Local Mock Fallbacks ────────────────────────────────────────────────────
const N_EPOCHS = 50;

const generateLossCurve = (initVal, decay, noise) =>
  Array.from({ length: N_EPOCHS }, (_, i) =>
    +(initVal * Math.exp(-(i + 1) / decay) + noise * Math.random() + 80).toFixed(3)
  );

const MOCK_METRICS = {
  r2:      0.842,
  mae:     18.54,
  rmse:    26.42,
  pearson: 0.895,
  mape:    14.7,
  bias:    -2.3,
};

const MOCK_CURVE = {
  epochs:     Array.from({ length: N_EPOCHS }, (_, i) => i + 1),
  train_loss: generateLossCurve(2800, 10, 18),
  val_loss:   generateLossCurve(3100, 11, 22),
};

const generateResiduals = (n = 250) =>
  Array.from({ length: n }, () => ({
    observed:  20 + Math.random() * 360,
    predicted: 0,                          // filled below
    residual:  +(Math.random() * 60 - 30).toFixed(2),
  })).map((r) => ({ ...r, predicted: +(r.observed + r.residual).toFixed(2) }));

// ─── API Functions ────────────────────────────────────────────────────────────

/**
 * Fetch model performance scalar metrics (R², MAE, RMSE, Pearson).
 * GET /api/model/performance
 */
export async function fetchModelMetrics() {
  try {
    return await axiosInstance.get('/model/performance', { cache: true });
  } catch {
    console.warn('[modelApi] Backend offline – serving mock model metrics.');
    return MOCK_METRICS;
  }
}

/**
 * Fetch training / validation loss curves over epochs.
 * GET /api/model/loss-curve
 */
export async function fetchLossCurve() {
  try {
    return await axiosInstance.get('/model/loss-curve', { cache: true });
  } catch {
    console.warn('[modelApi] Backend offline – serving mock loss curve.');
    return MOCK_CURVE;
  }
}

/**
 * Fetch observed vs predicted residual scatter data for validation plots.
 * GET /api/model/residuals?n=<int>
 */
export async function fetchResiduals(n = 250) {
  try {
    return await axiosInstance.get('/model/residuals', { params: { n }, cache: true });
  } catch {
    console.warn('[modelApi] Backend offline – serving mock residuals.');
    return generateResiduals(n);
  }
}
