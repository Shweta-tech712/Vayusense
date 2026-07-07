import React, { useMemo, useState, useCallback } from 'react';
import { useFilters } from '../context/FilterContext';
import { useAQI } from '../hooks/useAQI';
import { useAnalytics } from '../hooks/useAnalytics';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { RefreshCw, AlertTriangle, BarChart2 } from 'lucide-react';
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar, Legend
} from 'recharts';

// ─── Axis config for each preset ─────────────────────────────────────────────
const PRESETS = {
  pm25_aqi: {
    title: 'Ground PM2.5 vs Satellite AQI',
    xLabel: 'Observed PM2.5 (µg/m³)',    xKey: 'pm25',
    yLabel: 'Satellite-Derived AQI',      yKey: 'satellite_aqi',
  },
  aod_pm25: {
    title: 'MODIS AOD vs CPCB PM2.5',
    xLabel: 'Aerosol Optical Depth (AOD)', xKey: 'aod',
    yLabel: 'PM2.5 Concentration (µg/m³)', yKey: 'pm25',
  },
  fire_hcho: {
    title: 'FIRMS Fire Count vs TROPOMI HCHO',
    xLabel: 'Daily Fire Counts',           xKey: 'fire_count',
    yLabel: 'Tropospheric HCHO (×10⁻⁴ mol/m²)', yKey: 'hcho_vcd',
  },
};

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function CustomDot({ cx, cy, payload, xKey, yKey }) {
  return <circle cx={cx} cy={cy} r={4} fill="#f59e0b" fillOpacity={0.72} />;
}

// ─── Main Component ────────────────────────────────────────────────────────────
export default function AnalyticsView() {
  const { analysisDate, selectedState, selectedPollutant } = useFilters();
  const { scatter, correlation, loading, error, refetch } = useAnalytics(analysisDate, selectedState, selectedPollutant);
  const [activePreset, setActivePreset] = useState('pm25_aqi');

  const preset = PRESETS[activePreset];

  // Map raw scatter rows to the chosen axis keys
  const chartData = useMemo(() => {
    if (!scatter || scatter.length === 0) return [];
    return scatter.map(pt => ({
      x: pt[preset.xKey] ?? 0,
      y: pt[preset.yKey] ?? 0,
    })).filter(pt => pt.x > 0 && pt.y > 0);
  }, [scatter, preset]);

  // Find the Pearson r for the current preset's x-variable
  const pearson = useMemo(() => {
    if (!correlation || correlation.length === 0) return null;
    const varName = preset.xKey === 'pm25'       ? 'PM2.5'
                  : preset.xKey === 'aod'        ? 'AOD'
                  : preset.xKey === 'fire_count' ? 'Fire FRP'
                  : null;
    const row = correlation.find(c => c.variable === varName);
    return row?.r ?? null;
  }, [correlation, preset.xKey]);

  const handlePresetChange = useCallback((e) => setActivePreset(e.target.value), []);

  if (loading) return <Spinner message="Computing Cross-Variable Correlation Analysis..." />;

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <AlertTriangle className="w-8 h-8 text-amber-500" />
      <p className="text-slate-400 text-sm">{error}</p>
      <button onClick={refetch} className="flex items-center gap-2 px-4 py-2 bg-sky-600/20 border border-sky-500/30 text-sky-400 rounded-lg text-xs hover:bg-sky-600/30 transition-colors">
        <RefreshCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );

  return (
    <div className="space-y-8 select-none text-left">

      {/* 1. HEADER */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-widest uppercase">Cross-Variable Analysis</h2>
          <p className="text-xs text-slate-500 font-mono tracking-wider mt-1">
            Statistical regression between remote sensing columns and ground-based observations · {selectedState}
          </p>
        </div>
        <select
          value={activePreset}
          onChange={handlePresetChange}
          className="bg-[#090d16] border border-slate-850 hover:border-slate-750 px-3 py-2 rounded-lg text-xs font-semibold text-slate-300 focus:outline-none cursor-pointer"
        >
          {Object.entries(PRESETS).map(([key, p]) => (
            <option key={key} value={key}>{p.title}</option>
          ))}
        </select>
      </div>

      {/* 2. STATS + SCATTER */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Stats column */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 flex flex-col justify-between h-fit lg:min-h-[480px]">
          <div className="space-y-6">
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">Correlation Coefficients</h3>
            <div className="space-y-4">
              <div>
                <span className="text-xs font-mono text-slate-400 uppercase block">Pearson Correlation (R)</span>
                <span className="text-3xl font-bold text-sky-400">
                  {pearson != null ? pearson.toFixed(3) : '—'}
                </span>
                <span className="text-xs font-mono text-slate-400 block mt-1">
                  {pearson != null && pearson > 0.7 ? 'Strong positive relationship'
                   : pearson != null ? 'Moderate linear relationship' : 'Awaiting backend data'}
                </span>
              </div>
              <div className="border-t border-slate-900/60 pt-4">
                <span className="text-xs font-mono text-slate-400 uppercase block">Observations Plotted</span>
                <span className="text-3xl font-bold text-amber-500">{chartData.length}</span>
                <span className="text-xs font-mono text-slate-400 block mt-1">Station-pair daily records</span>
              </div>
              {/* All features list */}
              {correlation.length > 0 && (
                <div className="border-t border-slate-900/60 pt-4">
                  <span className="text-xs font-mono text-slate-400 uppercase block mb-2">All Features vs AQI</span>
                  <div className="space-y-1.5">
                    {correlation.slice(0, 6).map(c => (
                      <div key={c.variable} className="flex justify-between text-sm">
                        <span className="text-slate-400">{c.variable}</span>
                        <span className={`font-mono font-bold ${Math.abs(c.r) > 0.7 ? 'text-sky-400' : 'text-slate-300'}`}>
                          {c.r?.toFixed(3) ?? '—'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="border-t border-slate-900/60 pt-4 text-xs text-slate-400 font-mono mt-4 leading-relaxed">
            Daily spatial aggregations across stations during the crop residue burning season ({analysisDate.slice(0, 7)}).
          </div>
        </div>

        {/* Scatter Chart */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 lg:col-span-2 h-fit lg:min-h-[480px] flex flex-col justify-between">
          <h3 className="text-sm font-bold uppercase tracking-wider text-white mb-4">{preset.title}</h3>
          {chartData.length === 0 ? (
            <EmptyState
              icon={BarChart2}
              title="No Scatter Data"
              message="Backend returned no observations for this date. Try a different analysis date."
              action={{ label: 'Retry', onClick: refetch }}
            />
          ) : (
            <div className="w-full h-[350px] min-h-[350px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                  <XAxis
                    type="number" dataKey="x" name={preset.xLabel}
                    stroke="#64748b" fontSize={11}
                    label={{ value: preset.xLabel, position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }}
                  />
                  <YAxis
                    type="number" dataKey="y" name={preset.yLabel}
                    stroke="#64748b" fontSize={11}
                    label={{ value: preset.yLabel, angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11, fontFamily: 'monospace' }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.1)' }}
                    contentStyle={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff', fontSize: '11px' }}
                  />
                  <Scatter name="Station Observations" data={chartData} fill="#f59e0b" fillOpacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* 3. FEATURE CORRELATION BAR CHART */}
      {correlation.length > 0 && (
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6">
          <h3 className="text-sm font-bold uppercase tracking-wider text-white mb-1">Feature-AQI Pearson Correlation</h3>
          <p className="text-xs font-mono text-slate-400 tracking-wider mb-4">
            All satellite and ground features ranked by linear correlation strength (backend-sourced)
          </p>
          <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={correlation} layout="vertical" margin={{ top: 0, right: 20, left: 80, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" horizontal={false} />
                <XAxis type="number" domain={[-1, 1]} stroke="#64748b" fontSize={11} />
                <YAxis type="category" dataKey="variable" stroke="#64748b" fontSize={11} width={80} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff', fontSize: '11px' }}
                />
                <Bar dataKey="r" name="Pearson r" radius={[0, 3, 3, 0]}>
                  {correlation.map((entry, idx) => (
                    <rect
                      key={idx}
                      fill={entry.r > 0 ? '#0ea5e9' : '#ef4444'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

    </div>
  );
}
