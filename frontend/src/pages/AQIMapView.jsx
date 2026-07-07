import React, { useState, useEffect, useCallback, useMemo, memo } from 'react';
import { useFilters } from '../context/FilterContext';
import { useAQI } from '../hooks/useAQI';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { RefreshCw, AlertTriangle, Map } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// ─── AQI colour scale ────────────────────────────────────────────────────────
function getAQITheme(aqi) {
  if (aqi <= 50)  return { fill: '#10b981', text: 'Good'         };
  if (aqi <= 100) return { fill: '#84cc16', text: 'Satisfactory' };
  if (aqi <= 200) return { fill: '#eab308', text: 'Moderate'     };
  if (aqi <= 300) return { fill: '#f97316', text: 'Poor'         };
  if (aqi <= 400) return { fill: '#ef4444', text: 'Very Poor'    };
  return               { fill: '#a855f7', text: 'Severe'         };
}

// ─── Map view controller (no re-render overhead) ─────────────────────────────
function MapViewCoordinator({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.setView(center, zoom, { animate: true, duration: 0.8 });
  }, [center, zoom, map]);
  return null;
}

// ─── Memoised station row (prevents list re-renders on map pan) ──────────────
const StationRow = memo(function StationRow({ station, isActive, onSelect }) {
  const theme = getAQITheme(station.cpcb_aqi);
  return (
    <button
      onClick={() => onSelect(station)}
      className={`w-full text-left p-3 rounded-lg border transition-all duration-200 ${
        isActive ? 'border-sky-500 bg-sky-950/20' : 'border-slate-900 bg-slate-950/40 hover:border-slate-800'
      }`}
    >
      <div className="flex justify-between items-start">
        <span className="text-sm font-bold text-slate-200 truncate pr-2 w-48 block">
          {station.station.split(',')[0]}
        </span>
        <span
          className="text-xs font-mono font-bold px-1.5 py-0.5 rounded"
          style={{ backgroundColor: `${theme.fill}20`, color: theme.fill }}
        >
          {station.cpcb_aqi}
        </span>
      </div>
      <div className="flex justify-between items-center mt-2">
        <span className="text-xs text-slate-400 truncate w-36 block">
          {station.station.split(',')[1] || 'India'}
        </span>
        <span className="text-xs text-slate-300 font-medium">{theme.text}</span>
      </div>
    </button>
  );
});

// ─── Memoised circle marker (prevents map re-renders) ────────────────────────
const StationMarker = memo(function StationMarker({ station }) {
  const theme = getAQITheme(station.cpcb_aqi);
  return (
    <CircleMarker
      center={[station.latitude, station.longitude]}
      radius={8}
      fillColor={theme.fill}
      color="#050811"
      weight={1.2}
      fillOpacity={0.85}
    >
      <Popup className="leaflet-popup-dark">
        <div className="text-xs font-sans text-left">
          <h4 className="font-bold text-slate-800 border-b pb-1 mb-1.5">{station.station}</h4>
          <div className="space-y-1">
            <div className="flex justify-between"><span>AQI:</span>       <strong style={{ color: theme.fill }}>{station.cpcb_aqi}</strong></div>
            <div className="flex justify-between"><span>PM2.5:</span>     <strong className="text-amber-600">{station.pm25} µg/m³</strong></div>
            <div className="flex justify-between"><span>PM10:</span>      <span>{station.pm10} µg/m³</span></div>
            <div className="flex justify-between"><span>NO₂:</span>       <span>{station.no2} µg/m³</span></div>
            <div className="flex justify-between"><span>Ozone:</span>     <span>{station.o3} µg/m³</span></div>
            {station.so2  != null && <div className="flex justify-between"><span>SO₂:</span> <span>{station.so2} µg/m³</span></div>}
            {station.co   != null && <div className="flex justify-between"><span>CO:</span>  <span>{station.co} mg/m³</span></div>}
            {station.temp != null && <div className="flex justify-between"><span>Temp:</span><span>{station.temp} °C</span></div>}
            {station.humidity != null && <div className="flex justify-between"><span>Humidity:</span><span>{station.humidity}%</span></div>}
          </div>
        </div>
      </Popup>
    </CircleMarker>
  );
});

// ─── Main Component ────────────────────────────────────────────────────────────
export default function AQIMapView() {
  const { analysisDate, selectedState, selectedPollutant, searchQuery } = useFilters();
  const { stations, loading, error, refetch } = useAQI(analysisDate, selectedState, selectedPollutant);

  const [mapCenter,     setMapCenter]     = useState([20.5937, 78.9629]);
  const [mapZoom,       setMapZoom]       = useState(5);
  const [activeStation, setActiveStation] = useState(null);

  // Search filter applied locally so UI reacts instantly
  const filteredStations = useMemo(() => {
    if (!searchQuery) return stations;
    const query = searchQuery.toLowerCase();
    return stations.filter(s => 
      s.station.toLowerCase().includes(query) || 
      (s.aqi_category || '').toLowerCase().includes(query)
    );
  }, [stations, searchQuery]);

  // Auto-pan when selectedState changes
  useEffect(() => {
    if (selectedState === 'All India') {
      setMapCenter([20.5937, 78.9629]);
      setMapZoom(5);
      setActiveStation(null);
    } else if (filteredStations.length > 0) {
      setMapCenter([filteredStations[0].latitude, filteredStations[0].longitude]);
      setMapZoom(7);
    }
  }, [selectedState, filteredStations]);

  const selectStation = useCallback((station) => {
    setActiveStation(station);
    setMapCenter([station.latitude, station.longitude]);
    setMapZoom(10);
  }, []);

  // Memoised stats summary shown in the active-station HUD
  const activeTheme = useMemo(
    () => activeStation ? getAQITheme(activeStation.cpcb_aqi) : null,
    [activeStation]
  );

  if (loading) return <Spinner message="Interrogating CPCB Monitoring Stations..." />;

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <AlertTriangle className="w-8 h-8 text-amber-500" />
      <p className="text-slate-400 text-sm">{error}</p>
      <button onClick={refetch} className="flex items-center gap-2 px-4 py-2 bg-sky-600/20 border border-sky-500/30 text-sky-400 rounded-lg text-xs hover:bg-sky-600/30 transition-colors">
        <RefreshCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );

  if (stations.length === 0) return (
    <EmptyState
      icon={Map}
      title="No Stations Found"
      message={`No CPCB stations matched "${selectedState}" for ${analysisDate}. Try a different state or date.`}
      action={{ label: 'Reset to All India', onClick: refetch }}
    />
  );

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-12rem)] select-none text-left">

      {/* 1. STATION LIST PANEL */}
      <div className="w-full lg:w-88 bg-[#090d16] border border-slate-900 rounded-xl p-5 flex flex-col justify-between shrink-0 overflow-y-auto">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-white mb-1">Stations Directory</h3>
          <p className="text-xs text-slate-400 font-mono tracking-wide mb-3">
            {filteredStations.length} stations · {selectedPollutant} · {selectedState}
          </p>
          <div className="space-y-2 overflow-y-auto max-h-[300px] lg:max-h-[380px] pr-1">
            {filteredStations.map((station) => (
              <StationRow
                key={station.station}
                station={station}
                isActive={activeStation?.station === station.station}
                onSelect={selectStation}
              />
            ))}
          </div>
        </div>

        {/* AQI Legend */}
        <div className="border-t border-slate-900/60 pt-4 mt-4 space-y-1.5">
          <span className="text-xs font-mono tracking-widest text-slate-400 uppercase block mb-1">CPCB Index Scale</span>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2">
            {[
              { color: '#10b981', label: 'Good'   },
              { color: '#84cc16', label: 'Satis.' },
              { color: '#eab308', label: 'Mod.'   },
              { color: '#f97316', label: 'Poor'   },
              { color: '#ef4444', label: 'V. Poor'},
              { color: '#a855f7', label: 'Severe' },
            ].map(({ color, label }) => (
              <div key={label} className="flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                <span className="text-xs text-slate-300">{label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 2. MAP CONTAINER */}
      <div className="flex-1 bg-[#090d16] border border-slate-900 rounded-xl overflow-hidden relative z-0 min-h-[350px]">

        {/* Active station HUD overlay */}
        {activeStation && activeTheme && (
          <div className="absolute top-4 right-4 z-40 bg-[#090d16]/95 border border-slate-800 rounded-xl p-4 shadow-xl max-w-xs">
            <h4 className="text-xs font-bold text-white uppercase tracking-wider">{activeStation.station}</h4>
            <div className="grid grid-cols-2 gap-4 mt-3 pt-3 border-t border-slate-900/60">
              <div>
                <span className="text-xs font-mono text-slate-400 uppercase block">AQI</span>
                <span className="text-lg font-bold" style={{ color: activeTheme.fill }}>{activeStation.cpcb_aqi}</span>
                <span className="text-xs font-mono text-slate-300 block">{activeTheme.text}</span>
              </div>
              <div>
                <span className="text-xs font-mono text-slate-400 uppercase block">PM2.5</span>
                <span className="text-lg font-bold text-amber-500">{activeStation.pm25} µg/m³</span>
              </div>
              {activeStation.no2 != null && (
                <div>
                  <span className="text-xs font-mono text-slate-400 uppercase block">NO₂</span>
                  <span className="text-base font-bold text-slate-200">{activeStation.no2} µg/m³</span>
                </div>
              )}
              {activeStation.o3 != null && (
                <div>
                  <span className="text-xs font-mono text-slate-400 uppercase block">Ozone</span>
                  <span className="text-base font-bold text-slate-200">{activeStation.o3} µg/m³</span>
                </div>
              )}
              {activeStation.temp != null && (
                <div>
                  <span className="text-xs font-mono text-slate-400 uppercase block">Temp</span>
                  <span className="text-base font-bold text-slate-200">{activeStation.temp} °C</span>
                </div>
              )}
              {activeStation.humidity != null && (
                <div>
                  <span className="text-xs font-mono text-slate-400 uppercase block">Humidity</span>
                  <span className="text-base font-bold text-slate-200">{activeStation.humidity}%</span>
                </div>
              )}
            </div>
          </div>
        )}

        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          className="w-full h-full"
          style={{ background: '#050811' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CartoDB</a> contributors'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <MapViewCoordinator center={mapCenter} zoom={mapZoom} />
          {filteredStations.map((station) => (
            <StationMarker key={station.station} station={station} />
          ))}
        </MapContainer>
      </div>

    </div>
  );
}
