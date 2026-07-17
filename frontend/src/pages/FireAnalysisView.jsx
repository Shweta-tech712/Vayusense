import React, { useMemo, memo } from 'react';
import { useFilters } from '../context/FilterContext';
import { useFire } from '../hooks/useFire';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { useMap, MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import { RefreshCw, AlertTriangle, Flame } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const STATE_COORDS = {
  'All India': { center: [22.0, 79.0], zoom: 5 },
  'Delhi': { center: [28.6139, 77.2090], zoom: 9 },
  'Maharashtra': { center: [19.7515, 75.7139], zoom: 6 },
  'Karnataka': { center: [15.3173, 75.7139], zoom: 6 },
  'West Bengal': { center: [22.9868, 87.8550], zoom: 7 },
  'Tamil Nadu': { center: [11.1271, 78.6569], zoom: 6 },
  'Telangana': { center: [18.1124, 79.0193], zoom: 6 },
  'Bihar': { center: [25.0961, 85.3131], zoom: 7 }
};

// Component to dynamically update map center
function MapUpdater({ center, zoom }) {
  const map = useMap();
  React.useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

// ─── Colour by FRP intensity ──────────────────────────────────────────────────
function getFireColor(frp) {
  if (frp <= 20)  return '#f59e0b';
  if (frp <= 80)  return '#f97316';
  return '#ef4444';
}

// ─── Memoised fire map marker ─────────────────────────────────────────────────
const FireMarker = memo(function FireMarker({ fire, idx }) {
  const color = getFireColor(fire.frp);
  return (
    <CircleMarker
      center={[fire.latitude, fire.longitude]}
      radius={Math.min(6 + fire.frp / 60, 18)}
      fillColor={color}
      color="#050811"
      weight={1}
      fillOpacity={0.82}
    >
      <Popup className="leaflet-popup-dark">
        <div className="text-xs font-sans">
          <strong className="text-red-500">Active Fire Spot</strong>
          <div className="mt-1.5 space-y-1 text-[11px]">
            <div>FRP: <strong>{fire.frp.toFixed(1)} MW</strong></div>
            <div>Confidence: <strong>{fire.confidence}%</strong></div>
            <div>Satellite: <strong>{fire.satellite ?? 'MODIS/VIIRS'}</strong></div>
            {fire.acq_date && <div>Acquired: <strong>{fire.acq_date}</strong></div>}
            <div>Coords: {fire.latitude.toFixed(4)}°N, {fire.longitude.toFixed(4)}°E</div>
          </div>
        </div>
      </Popup>
    </CircleMarker>
  );
});

// ─── Main Component ────────────────────────────────────────────────────────────
export default function FireAnalysisView() {
  const { analysisDate, minFRP, setMinFRP, selectedState, searchQuery } = useFilters();
  const { fires, allFires, monthly, summary, loading, error, refetch } = useFire(analysisDate, minFRP, selectedState);

  // Client-side search query filtering for active fires
  const filteredFires = useMemo(() => {
    if (!searchQuery) return fires;
    const query = searchQuery.toLowerCase();
    return fires.filter(f => 
      (f.satellite || '').toLowerCase().includes(query) ||
      String(f.confidence).includes(query) ||
      String(f.frp).includes(query)
    );
  }, [fires, searchQuery]);

  // Memoised monthly data to prevent bar chart re-render on slider move
  const monthlyChartData = useMemo(() => monthly, [monthly]);

  if (loading) return <Spinner message="Interrogating NASA FIRMS Active Fire Feeds..." />;

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <AlertTriangle className="w-8 h-8 text-amber-500" />
      <p className="text-slate-400 text-sm">{error}</p>
      <button onClick={refetch} className="flex items-center gap-2 px-4 py-2 bg-amber-600/20 border border-amber-500/30 text-amber-400 rounded-lg text-xs hover:bg-amber-600/30 transition-colors">
        <RefreshCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-12rem)] select-none text-left">

      {/* 1. CONTROLS & CHART PANEL */}
      <div className="w-full lg:w-96 bg-[#090d16] border border-slate-900 rounded-xl p-5 flex flex-col justify-between shrink-0 overflow-y-auto">
        <div className="space-y-6">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">Biomass Burning Analysis</h3>
            <p className="text-xs text-slate-400 font-mono tracking-wide mt-1">
              Active open combustion monitoring from MODIS &amp; VIIRS telemetry · {selectedState}
            </p>
          </div>

          {/* FRP Filter Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-400">Min Fire Radiative Power</span>
              <span className="font-mono font-bold text-amber-500">{minFRP} MW</span>
            </div>
            <input
              type="range" min="5" max="200" step="5"
              value={minFRP}
              onChange={(e) => setMinFRP(parseInt(e.target.value))}
              className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
            <span className="text-xs font-mono text-slate-400 block">Instant client-side filter — no re-fetch</span>
          </div>

          {/* Live stats from backend summary */}
          <div className="p-4 bg-slate-950/40 border border-slate-900 rounded-lg grid grid-cols-2 gap-4">
            <div>
              <span className="text-xs font-mono text-slate-400 uppercase block">Total FIRMS Spots</span>
              <span className="text-xl font-bold text-white">{allFires.length}</span>
            </div>
            <div>
              <span className="text-xs font-mono text-slate-400 uppercase block">After FRP Filter</span>
              <span className="text-xl font-bold text-amber-500">{filteredFires.length}</span>
            </div>
            {summary?.max_frp != null && (
              <div>
                <span className="text-xs font-mono text-slate-400 uppercase block">Peak FRP</span>
                <span className="text-base font-bold text-red-400">{summary.max_frp} MW</span>
              </div>
            )}
            {summary?.modis_count != null && (
              <div>
                <span className="text-xs font-mono text-slate-400 uppercase block">MODIS / VIIRS</span>
                <span className="text-base font-bold text-slate-300">{summary.modis_count} / {summary.viirs_count}</span>
              </div>
            )}
          </div>

          {/* Monthly Seasonal Trend — from backend */}
          <div className="space-y-3 pt-2">
            <h4 className="text-xs font-mono tracking-widest text-slate-400 uppercase">Residue Burning Season Trends</h4>
            {monthlyChartData.length === 0 ? (
              <EmptyState icon={Flame} title="No Monthly Data" message="Backend returned no seasonal fire statistics." />
            ) : (
              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={monthlyChartData} margin={{ top: 5, right: 0, left: -25, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                    <XAxis dataKey="month" stroke="#64748b" fontSize={9} />
                    <YAxis stroke="#64748b" fontSize={9} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px' }}
                      labelStyle={{ color: '#94a3b8', fontSize: '9px', fontFamily: 'monospace' }}
                      itemStyle={{ color: '#fff', fontSize: '10px' }}
                    />
                    <Bar dataKey="baseline" name="3-Yr Average" fill="#334155" radius={[2, 2, 0, 0]} />
                    <Bar dataKey="current"  name="Current Year" fill="#f59e0b" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-slate-900/60 pt-4 text-xs text-slate-400 leading-relaxed font-mono mt-4">
          FRP (Megawatts) represents radiative heat energy released per unit time from open biomass combustion sources.
        </div>
      </div>

      {/* 2. MAP VIEW */}
      <div className="flex-1 bg-[#090d16] border border-slate-900 rounded-xl overflow-hidden relative z-0 min-h-[350px]">
        {fires.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center">
            <EmptyState
              icon={Flame}
              title="No Fire Events"
              message={`No fire detections above ${minFRP} MW FRP. Try lowering the FRP threshold.`}
              action={{ label: 'Reset FRP Filter', onClick: () => setMinFRP(5) }}
            />
          </div>
        ) : (
          <MapContainer center={STATE_COORDS[selectedState]?.center || [30.0, 75.5]} zoom={STATE_COORDS[selectedState]?.zoom || 7} className="w-full h-full" style={{ background: '#050811' }}>
            <MapUpdater center={STATE_COORDS[selectedState]?.center || [30.0, 75.5]} zoom={STATE_COORDS[selectedState]?.zoom || 7} />
            <TileLayer
              attribution="&copy; NASA LANCE FIRMS"
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />
            {filteredFires.map((fire, idx) => (
              <FireMarker key={`${fire.latitude}-${fire.longitude}-${idx}`} fire={fire} idx={idx} />
            ))}
          </MapContainer>
        )}
      </div>

    </div>
  );
}
