import React, { memo, useMemo } from 'react';
import { useFilters } from '../context/FilterContext';
import { useTransport } from '../hooks/useTransport';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup } from 'react-leaflet';
import { RefreshCw, AlertTriangle, Wind } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// ─── Memoised wind node marker ────────────────────────────────────────────────
const WindNode = memo(function WindNode({ w, idx }) {
  return (
    <CircleMarker
      key={idx}
      center={[w.latitude, w.longitude]}
      radius={5}
      fillColor="#0ea5e9"
      color="#0284c7"
      weight={1}
      fillOpacity={0.6}
    >
      <Popup className="leaflet-popup-dark">
        <div className="text-xs font-sans text-left">
          <strong>Wind Grid Coordinate</strong>
          <div className="mt-1 space-y-1 text-[11px] font-mono">
            <div>Speed: <span>{w.speed?.toFixed(1) ?? '–'} m/s</span></div>
            <div>Direction: <span>{w.direction?.toFixed(0) ?? '–'}°</span></div>
            <div>U-Component: <span>{w.u?.toFixed(2) ?? '–'} m/s</span></div>
            <div>V-Component: <span>{w.v?.toFixed(2) ?? '–'} m/s</span></div>
            {w.temperature  != null && <div>Temperature: <span>{w.temperature} K</span></div>}
            {w.humidity     != null && <div>Rel. Humidity: <span>{w.humidity}%</span></div>}
            {w.pressure_hpa != null && <div>Pressure: <span>{w.pressure_hpa} hPa</span></div>}
          </div>
        </div>
      </Popup>
    </CircleMarker>
  );
});

// ─── Main Component ────────────────────────────────────────────────────────────
export default function TransportView() {
  const { analysisDate, selectedState } = useFilters();
  const { windVectors, trajectory, stats, loading, error, refetch } = useTransport(analysisDate, selectedState);

  // Memoised trajectory for Polyline (prevents re-render on every state change)
  const trajPositions = useMemo(
    () => trajectory.map(pt => Array.isArray(pt) ? pt : [pt.lat ?? pt.latitude, pt.lon ?? pt.longitude]),
    [trajectory]
  );

  if (loading) return <Spinner message="Computing ERA5 Wind Field Advection Coordinates..." />;

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <AlertTriangle className="w-8 h-8 text-amber-500" />
      <p className="text-slate-400 text-sm">{error}</p>
      <button onClick={refetch} className="flex items-center gap-2 px-4 py-2 bg-sky-600/20 border border-sky-500/30 text-sky-400 rounded-lg text-xs hover:bg-sky-600/30 transition-colors">
        <RefreshCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );

  if (windVectors.length === 0) return (
    <EmptyState
      icon={Wind}
      title="No Wind Data"
      message={`ERA5 wind vectors unavailable for ${analysisDate}. Try a different date.`}
      action={{ label: 'Retry', onClick: refetch }}
    />
  );

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-12rem)] select-none text-left">

      {/* 1. TRANSPORT DISPERSAL METRICS HUD */}
      <div className="w-full lg:w-88 bg-[#090d16] border border-slate-900 rounded-xl p-5 flex flex-col justify-between shrink-0 overflow-y-auto">
        <div className="space-y-6">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">Pollutant Transport Analysis</h3>
            <p className="text-xs text-slate-400 font-mono tracking-wide mt-1">
              ERA5 boundary layer winds and HYSPLIT-simulated trajectory advection
            </p>
          </div>

          {/* Stats from backend */}
          <div className="space-y-3">
            <h4 className="text-xs font-mono tracking-widest text-slate-400 uppercase">Trajectory Telemetry</h4>
            <div className="p-3 bg-slate-950/40 border border-slate-900 rounded-lg space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Advection Vector:</span>
                <span className="font-mono font-bold text-sky-400">{stats?.dominant_direction ?? 'WNW'} ➔ SE</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Mean Wind Speed:</span>
                <span className="font-mono font-bold text-white">{stats?.mean_wind_speed ?? '–'} m/s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Transport Distance:</span>
                <span className="font-mono font-bold text-amber-500">{stats?.transport_distance_km ?? '–'} km</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Mixing Height:</span>
                <span className="font-mono font-bold text-slate-300">{stats?.mixing_height_m ?? '–'} m</span>
              </div>
              {stats?.mean_temp_k != null && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Mean Temp:</span>
                  <span className="font-mono font-bold text-slate-300">{(stats.mean_temp_k - 273.15).toFixed(1)} °C</span>
                </div>
              )}
              {stats?.mean_humidity != null && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Mean Humidity:</span>
                  <span className="font-mono font-bold text-slate-300">{stats.mean_humidity} %</span>
                </div>
              )}
              {stats?.mean_pressure_hpa != null && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Mean Pressure:</span>
                  <span className="font-mono font-bold text-slate-300">{stats.mean_pressure_hpa} hPa</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-slate-400">Wind Nodes:</span>
                <span className="font-mono font-bold text-teal-400">{windVectors.length}</span>
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="border-t border-slate-900/60 pt-4">
            <h4 className="text-xs font-mono tracking-widest text-slate-400 uppercase mb-2">Transport Legend</h4>
            <div className="space-y-2 text-xs text-slate-300">
              <div className="flex items-center space-x-2">
                <span className="w-4 h-0.5 bg-sky-400 block" />
                <span>Calculated Advection Path</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-400/50 block" />
                <span>ERA5 Wind Direction Node</span>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-900/60 pt-4 text-xs text-slate-400 leading-relaxed font-mono">
          Wind advection governs transport of fine aerosols (PM2.5) and formaldehyde plumes across state boundaries.
        </div>
      </div>

      {/* 2. MAP VIEW */}
      <div className="flex-1 bg-[#090d16] border border-slate-900 rounded-xl overflow-hidden relative z-0 min-h-[350px]">
        <MapContainer center={[26.0, 78.0]} zoom={6} className="w-full h-full" style={{ background: '#050811' }}>
          <TileLayer
            attribution="&copy; Copernicus ERA5 contributors"
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />

          {/* Wind vector nodes */}
          {windVectors.map((w, idx) => (
            <WindNode key={idx} w={w} idx={idx} />
          ))}

          {/* Advection trajectory polyline */}
          {trajPositions.length > 1 && (
            <Polyline
              positions={trajPositions}
              pathOptions={{ color: '#38bdf8', weight: 3.5, opacity: 0.9, dashArray: '8,8' }}
            >
              <Popup className="leaflet-popup-dark text-xs">
                <div>
                  <strong>Plume Advection Trajectory</strong>
                  <p className="mt-1 text-slate-500 font-mono text-[10px]">
                    Calculated downstream air-mass movement ({trajPositions.length} nodes).
                  </p>
                </div>
              </Popup>
            </Polyline>
          )}

          {/* Source & Receptor markers */}
          {trajPositions.length > 1 && (
            <>
              <CircleMarker center={trajPositions[0]} radius={8} fillColor="#f59e0b" color="#b45309" weight={1.5}>
                <Popup className="leaflet-popup-dark">
                  <div className="text-xs"><strong>Source Region</strong><br />Punjab Cropland Fires</div>
                </Popup>
              </CircleMarker>
              <CircleMarker center={trajPositions[trajPositions.length - 1]} radius={8} fillColor="#ef4444" color="#b91c1c" weight={1.5}>
                <Popup className="leaflet-popup-dark">
                  <div className="text-xs"><strong>Receptor Region</strong><br />Indo-Gangetic Plains / Delhi NCT</div>
                </Popup>
              </CircleMarker>
            </>
          )}
        </MapContainer>
      </div>

    </div>
  );
}
