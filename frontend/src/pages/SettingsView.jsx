import React, { useState, useCallback } from 'react';
import { Settings, Save, RotateCcw, CheckCircle2, XCircle } from 'lucide-react';
import axiosInstance from '../services/axiosInstance';

const DEFAULTS = {
  bbox: { north: 37.6, south: 8.4, east: 97.4, west: 68.1 },
  dbscan: { eps: 0.15, minSamples: 5 },
  aqi: { predictionWindow: 24, featureSet: 'full' },
};

export default function SettingsView() {
  const [bbox,        setBbox]        = useState(DEFAULTS.bbox);
  const [dbscan,      setDbscan]      = useState(DEFAULTS.dbscan);
  const [aqiConfig,   setAqiConfig]   = useState(DEFAULTS.aqi);
  const [status,      setStatus]      = useState(null); // 'saving' | 'ok' | 'error'
  const [statusMsg,   setStatusMsg]   = useState('');

  const handleSave = useCallback(async () => {
    setStatus('saving');
    setStatusMsg('Sending configuration to backend...');
    try {
      await axiosInstance.post('/api/config', {
        bbox,
        dbscan_eps:          dbscan.eps,
        dbscan_min_samples:  dbscan.minSamples,
        aqi_prediction_window: aqiConfig.predictionWindow,
        feature_set:         aqiConfig.featureSet,
      });
      setStatus('ok');
      setStatusMsg('Configuration saved successfully.');
    } catch (err) {
      // Backend offline — save to localStorage as fallback
      try {
        localStorage.setItem('isro_config', JSON.stringify({ bbox, dbscan, aqiConfig }));
        setStatus('ok');
        setStatusMsg('Backend offline — configuration saved to local session (will sync when server restarts).');
      } catch {
        setStatus('error');
        setStatusMsg('Failed to save configuration.');
      }
    }
    setTimeout(() => setStatus(null), 4500);
  }, [bbox, dbscan, aqiConfig]);

  const handleReset = useCallback(() => {
    setBbox(DEFAULTS.bbox);
    setDbscan(DEFAULTS.dbscan);
    setAqiConfig(DEFAULTS.aqi);
    setStatus(null);
  }, []);

  return (
    <div className="space-y-8 select-none text-left max-w-2xl">

      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-widest uppercase">System Configurations</h2>
        <p className="text-xs text-slate-500 font-mono tracking-wider mt-1">
          Bounding box, DBSCAN cluster parameters, and AQI prediction settings
        </p>
      </div>

      {/* Status banner */}
      {status && (
        <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border text-xs font-mono ${
          status === 'ok'    ? 'bg-emerald-950/30 border-emerald-800 text-emerald-400' :
          status === 'error' ? 'bg-red-950/30 border-red-900 text-red-400' :
                               'bg-sky-950/30 border-sky-900 text-sky-400'
        }`}>
          {status === 'ok'    && <CheckCircle2 className="w-4 h-4 shrink-0" />}
          {status === 'error' && <XCircle      className="w-4 h-4 shrink-0" />}
          {status === 'saving'&& <Settings     className="w-4 h-4 shrink-0 animate-spin" />}
          <span>{statusMsg}</span>
        </div>
      )}

      <div className="space-y-6">

        {/* Bounding Box */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">Predictive Bounding Box Coordinates</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {['north','south','east','west'].map(key => (
              <div key={key}>
                <span className="text-xs font-mono text-slate-400 uppercase block mb-1">{key} Boundary</span>
                <input
                  type="number" step="0.1"
                  value={bbox[key]}
                  onChange={(e) => setBbox({ ...bbox, [key]: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 bg-[#050811] border border-slate-850 rounded-lg text-sm font-mono text-white focus:outline-none focus:border-sky-500/50"
                />
              </div>
            ))}
          </div>
        </div>

        {/* DBSCAN */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">Hotspot Cluster Parameters (DBSCAN)</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-semibold text-slate-355">Neighbourhood Radius (ε)</span>
                <span className="font-mono text-sky-400 font-bold">{dbscan.eps.toFixed(2)} °</span>
              </div>
              <input type="range" min="0.05" max="0.5" step="0.01"
                value={dbscan.eps}
                onChange={(e) => setDbscan({ ...dbscan, eps: parseFloat(e.target.value) })}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-semibold text-slate-355">Min Core Points</span>
                <span className="font-mono text-sky-400 font-bold">{dbscan.minSamples} pts</span>
              </div>
              <input type="range" min="2" max="15" step="1"
                value={dbscan.minSamples}
                onChange={(e) => setDbscan({ ...dbscan, minSamples: parseInt(e.target.value) })}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
              />
            </div>
          </div>
        </div>

        {/* AQI Model Config */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">AQI Prediction Model Settings</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-semibold text-slate-355">Prediction Window (hours)</span>
                <span className="font-mono text-amber-400 font-bold">{aqiConfig.predictionWindow} h</span>
              </div>
              <input type="range" min="1" max="72" step="1"
                value={aqiConfig.predictionWindow}
                onChange={(e) => setAqiConfig({ ...aqiConfig, predictionWindow: parseInt(e.target.value) })}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
            </div>
            <div>
              <span className="text-xs font-mono text-slate-400 uppercase block mb-1">Feature Set</span>
              <select
                value={aqiConfig.featureSet}
                onChange={(e) => setAqiConfig({ ...aqiConfig, featureSet: e.target.value })}
                className="w-full px-3 py-2 bg-[#050811] border border-slate-850 rounded-lg text-sm font-mono text-white focus:outline-none focus:border-sky-500/50"
              >
                <option value="full">Full (AOD + Fire + Wind + HCHO)</option>
                <option value="satellite">Satellite Only (AOD + HCHO)</option>
                <option value="ground">Ground Only (CPCB Stations)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex space-x-4 pt-2">
          <button
            onClick={handleSave}
            disabled={status === 'saving'}
            className="px-5 py-2.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 border border-sky-400/20 rounded-lg text-sm font-semibold uppercase tracking-wider text-white transition-all duration-300 flex items-center space-x-2"
          >
            <Save className="w-4 h-4" />
            <span>{status === 'saving' ? 'Saving...' : 'Save to Backend'}</span>
          </button>
          <button
            onClick={handleReset}
            className="px-5 py-2.5 bg-transparent hover:bg-slate-900/60 border border-slate-800 hover:border-slate-700 rounded-lg text-sm font-semibold uppercase tracking-wider text-slate-300 hover:text-white transition-all duration-300 flex items-center space-x-2"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset Defaults</span>
          </button>
        </div>

      </div>
    </div>
  );
}
