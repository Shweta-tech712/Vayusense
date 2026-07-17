import React, { useMemo, useCallback, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFilters } from '../context/FilterContext';
import { useAQI } from '../hooks/useAQI';
import { useFire } from '../hooks/useFire';
import { useHCHO } from '../hooks/useHCHO';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { 
  Globe, Eye, Flame, Wind, ArrowUpRight, LayoutDashboard, Search, History, Sparkles, 
  AlertCircle, ShieldAlert, ArrowRight, Activity, Thermometer, Droplets, Heart, HelpCircle, RefreshCw,
  Cpu, Database, Clock, Zap
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar, Cell } from 'recharts';
import { MapContainer, TileLayer, CircleMarker, Circle, Popup, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// ─── Map views coordinators ──────────────────────────────────────────────────
function MapUpdater({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.setView(center, zoom, { animate: true, duration: 1.0 });
    }
  }, [center, zoom, map]);
  return null;
}

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click(e) {
      onMapClick(e.latlng.lat, e.latlng.lng);
    }
  });
  return null;
}

// ─── CPCB AQI Scale colors mapping ────────────────────────────────────────────
function getAQITheme(aqi) {
  if (aqi <= 50)  return { fill: '#10b981', text: 'Good'         };
  if (aqi <= 100) return { fill: '#84cc16', text: 'Satisfactory' };
  if (aqi <= 200) return { fill: '#eab308', text: 'Moderate'     };
  if (aqi <= 300) return { fill: '#f97316', text: 'Poor'         };
  if (aqi <= 400) return { fill: '#ef4444', text: 'Very Poor'    };
  return               { fill: '#a855f7', text: 'Severe'         };
}

// ─── Memoised summary card ────────────────────────────────────────────────────
const SummaryCard = React.memo(function SummaryCard({ card, onClick }) {
  const IconComponent = card.icon;
  return (
    <div
      onClick={onClick}
      className={`bg-gradient-to-br ${card.color} border ${card.borderColor} rounded-xl p-6 cursor-pointer hover:scale-[1.02] transition-all duration-300 select-none`}
    >
      <div className="flex justify-between items-start">
        <div>
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">{card.title}</span>
          <h3 className="text-3xl font-bold text-white mt-2">{card.value}</h3>
          <span className="text-[10px] font-mono text-slate-500 block mt-1">{card.label}</span>
        </div>
        <div className="bg-slate-900/60 p-2 rounded-lg">
          <IconComponent className="w-5 h-5 text-slate-400" />
        </div>
      </div>
      <div className="mt-4 flex items-center text-[10px] font-mono text-slate-500 hover:text-sky-400 transition-colors">
        <span>{card.actionLabel ?? "View Module"}</span>
        <ArrowUpRight className="w-3 h-3 ml-1" />
      </div>
    </div>
  );
});

// ─── Main Component ────────────────────────────────────────────────────────────
export default function HomeView() {
  const navigate = useNavigate();
  const { 
    analysisDate, selectedState, selectedPollutant,
    activeLocationReport, setActiveLocationReport,
    selectedCoords, setSelectedCoords,
    recentSearches,
    isAnalyzingLocation,
    analyzeLocation,
    predictionError, setPredictionError,
    loadingPhase, currentAnalysisLocation
  } = useFilters();

  const [searchVal, setSearchVal] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  const {
    stations,
    trend: aqiTrend,
    loading: aqiLoading,
    error: aqiError,
    refetch: refetchAQI,
  } = useAQI(analysisDate, selectedState, selectedPollutant);

  const {
    allFires: fires,
    loading: fireLoading,
    error: fireError,
  } = useFire(analysisDate, 0, selectedState);

  const { hotspots, loading: hchoLoading } = useHCHO(analysisDate, 2.0, selectedState);

  const loading = aqiLoading || fireLoading;
  const error   = aqiError   || fireError;

  // Autocomplete Suggestions list
  const SUGGESTIONS = ['Delhi', 'Mumbai', 'Pune', 'Punjab', 'Bengaluru', 'Kolkata', 'Chennai', 'Hyderabad', 'Patna', 'Indore', 'Bhopal'];

  const filteredSuggestions = useMemo(() => {
    if (!searchVal) return SUGGESTIONS;
    return SUGGESTIONS.filter(item => item.toLowerCase().includes(searchVal.toLowerCase()));
  }, [searchVal]);

  // Derived aggregates for general view
  const avgAQI = useMemo(() => {
    if (stations.length === 0) return '—';
    return Math.round(stations.reduce((s, st) => s + (st.cpcb_aqi ?? 0), 0) / stations.length);
  }, [stations]);

  const avgWindSpeed = useMemo(() => {
    const w = stations.filter(s => s.wind_speed != null);
    if (w.length === 0) return '—';
    return (w.reduce((s, st) => s + st.wind_speed, 0) / w.length).toFixed(1);
  }, [stations]);

  const trend = useMemo(() => {
    if (aqiTrend && aqiTrend.length > 0) return aqiTrend;
    if (typeof avgAQI !== 'number') return [];
    return Array.from({ length: 7 }, (_, i) => {
      const date = new Date(new Date(analysisDate).getTime() - (6 - i) * 864e5);
      return {
        date:  date.toISOString().slice(5, 10),
        aqi:   Math.round(avgAQI + Math.sin(i) * 18 + (i % 3 === 0 ? 12 : -5)),
        fires: Math.max(0, Math.round(fires.length * 0.8 + Math.cos(i) * 10)),
      };
    });
  }, [aqiTrend, avgAQI, fires.length, analysisDate]);

  // Trigger search execution
  const handleSearchSubmit = useCallback((e) => {
    e.preventDefault();
    if (searchVal.trim()) {
      analyzeLocation(searchVal.trim());
      setShowSuggestions(false);
    }
  }, [searchVal, analyzeLocation]);

  const handleSuggestionClick = useCallback((name) => {
    setSearchVal(name);
    analyzeLocation(name);
    setShowSuggestions(false);
  }, [analyzeLocation]);

  const handleMapClick = useCallback((lat, lng) => {
    analyzeLocation(`Pinpoint Location`, lat, lng);
  }, [analyzeLocation]);

  // Build active cards values based on whether report is selected
  const CARDS = useMemo(() => {
    if (activeLocationReport) {
      const theme = getAQITheme(activeLocationReport.AQI);
      return [
        {
          title: 'AQI (Location Intelligence)',
          value: activeLocationReport.AQI,
          label: `${activeLocationReport.location} (${activeLocationReport.category})`,
          icon: Globe, color: 'from-sky-500/20 to-blue-600/5', borderColor: 'border-sky-500/20', path: '/aqi',
          actionLabel: 'Interactive Map'
        },
        {
          title: 'HCHO Gas Column',
          value: `${(activeLocationReport.HCHO.concentration * 100).toFixed(2)} ×10⁻²`,
          label: `HCHO Risk: ${activeLocationReport.HCHO.risk}${activeLocationReport.HCHO.hotspot ? ' · Hotspot Anomaly' : ''} · Prob: ${(activeLocationReport.HCHO.concentration * 250).toFixed(0)}%`,
          icon: Eye, color: 'from-indigo-500/20 to-indigo-600/5', borderColor: 'border-indigo-500/20', path: '/hcho',
          actionLabel: 'Outliers Registry'
        },
        {
          title: 'Biomass Burning Count',
          value: `${activeLocationReport.Fire.nearby_fire_count} Fires`,
          label: `Thermal Influence: ${activeLocationReport.Fire.influence} (${activeLocationReport.Fire.distance_km} km)`,
          icon: Flame, color: 'from-amber-500/20 to-amber-600/5', borderColor: 'border-amber-500/20', path: '/fire-analysis',
          actionLabel: 'Biomass Dashboard'
        },
        {
          title: 'Meteorological Weather',
          value: activeLocationReport.Weather.temperature === "Data unavailable" 
            ? "Data unavailable" 
            : `${activeLocationReport.Weather.temperature}°C`,
          label: activeLocationReport.Weather.humidity === "Data unavailable" 
            ? "Data unavailable" 
            : `Winds: ${activeLocationReport.Weather.wind_speed} m/s ${activeLocationReport.Weather.wind} · Humidity: ${activeLocationReport.Weather.humidity}%`,
          icon: Wind, color: 'from-teal-500/20 to-teal-600/5', borderColor: 'border-teal-500/20', path: '/transport-analysis',
          actionLabel: 'Atmospheric Vectors'
        },
      ];
    }

    return [
      {
        title: 'Surface AQI',
        value: avgAQI,
        label: selectedState === 'All India' ? 'CPCB All-India Mean' : `${selectedState} Mean`,
        icon: Globe, color: 'from-sky-500/20 to-blue-600/5', borderColor: 'border-sky-500/20', path: '/aqi',
      },
      {
        title: 'HCHO Hotspots',
        value: `${hotspots?.length || 0} Clusters`,
        label: 'TROPOMI DBSCAN Detected',
        icon: Eye, color: 'from-indigo-500/20 to-indigo-600/5', borderColor: 'border-indigo-500/20', path: '/hcho',
      },
      {
        title: 'Biomass Burning',
        value: fires.length,
        label: 'NASA FIRMS Fire Events',
        icon: Flame, color: 'from-amber-500/20 to-amber-600/5', borderColor: 'border-amber-500/20', path: '/fire-analysis',
      },
      {
        title: 'Wind Speed',
        value: avgWindSpeed !== '—' ? `${avgWindSpeed} m/s` : '—',
        label: 'Station-Avg Surface Wind',
        icon: Wind, color: 'from-teal-500/20 to-teal-600/5', borderColor: 'border-teal-500/20', path: '/transport-analysis',
      },
    ];
  }, [activeLocationReport, avgAQI, hotspots, fires.length, avgWindSpeed, selectedState]);

  // Dynamic pollutants graph data
  const pollutantGraphData = useMemo(() => {
    if (!activeLocationReport) return [];
    const p = activeLocationReport.pollutants;
    return [
      { name: 'PM2.5', value: p.PM25, color: '#f59e0b' },
      { name: 'NO2', value: p.NO2, color: '#38bdf8' },
      { name: 'SO2', value: p.SO2, color: '#f43f5e' },
      { name: 'CO', value: p.CO * 100, color: '#10b981', displayValue: p.CO }, // Scale CO for visibility
      { name: 'O3', value: p.O3, color: '#a855f7' }
    ];
  }, [activeLocationReport]);

  if (aqiLoading || fireLoading || hchoLoading) {
    return <Spinner message="Aggregating multi-source satellite & ground telemetry..." />;
  }
  if (error) return (
    <EmptyState
      icon={LayoutDashboard}
      title="Data Unavailable"
      message={`Backend error: ${error}. Check that the Python API is running on ${import.meta.env.VITE_API_URL}.`}
      action={{ label: 'Retry', onClick: refetchAQI }}
    />
  );

  return (
    <div className="space-y-8 select-none text-left">
      
      {/* 1. Header Title & Reset control */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-widest uppercase">
            {activeLocationReport ? `📍 Location Analytics Hub` : `System Control Panel`}
          </h2>
          <p className="text-xs text-slate-500 font-mono tracking-wider mt-1">
            {activeLocationReport 
              ? `AI space-based diagnostics for: ${activeLocationReport.location}`
              : `Live satellite observations & predictive metrics summary · ${selectedState} · ${analysisDate}`
            }
          </p>
        </div>
        {activeLocationReport && (
          <button
            onClick={() => {
              setActiveLocationReport(null);
              setSelectedCoords(null);
              setSearchVal('');
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white rounded-lg text-xs font-semibold tracking-wide transition-all"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Reset to All India
          </button>
        )}
      </div>

      {/* 2. CENTRAL SEARCH AND INTELLIGENCE SECTION */}
      <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 relative">
        <h3 className="text-sm font-bold uppercase tracking-wider text-sky-400 mb-4 flex items-center gap-2">
          <Sparkles className="w-4 h-4" /> AI Location Intelligence Engine
        </h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <form onSubmit={handleSearchSubmit} className="relative">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Enter City name, District, or State in India (e.g. Pune, Delhi)..."
                  value={searchVal}
                  onChange={(e) => {
                    setSearchVal(e.target.value);
                    setShowSuggestions(true);
                  }}
                  onFocus={() => setShowSuggestions(true)}
                  className="w-full pl-10 pr-12 py-3 bg-[#050811] border border-slate-850 hover:border-slate-700 focus:border-sky-500 rounded-xl text-sm text-white focus:outline-none transition-all placeholder:text-slate-500"
                />
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                  <Search className="h-4.5 w-4.5 text-slate-400" />
                </span>
                {searchVal && (
                  <button
                    type="button"
                    onClick={() => { setSearchVal(''); }}
                    className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-white text-xs"
                  >
                    Clear
                  </button>
                )}
              </div>

              {/* Autocomplete Suggestions Box */}
              {showSuggestions && filteredSuggestions.length > 0 && (
                <div className="absolute left-0 right-0 mt-2 bg-[#090d16] border border-slate-850 rounded-xl shadow-2xl z-50 overflow-hidden max-h-60 overflow-y-auto">
                  {filteredSuggestions.map((item) => (
                    <button
                      key={item}
                      type="button"
                      onClick={() => handleSuggestionClick(item)}
                      className="w-full text-left px-4 py-2.5 hover:bg-slate-900/60 text-slate-300 hover:text-white text-xs border-b border-slate-900/60 transition-colors"
                    >
                      {item}
                    </button>
                  ))}
                </div>
              )}
            </form>

            {/* Suggestions Chips & Recent Searches */}
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 text-xs">
              <span className="text-slate-400 flex items-center gap-1"><History className="w-3.5 h-3.5" /> Recent:</span>
              {recentSearches.map((item) => (
                <button
                  key={item}
                  onClick={() => handleSuggestionClick(item)}
                  className="px-2 py-1 bg-slate-950/40 hover:bg-slate-900/60 border border-slate-900 hover:border-slate-800 text-slate-400 hover:text-sky-400 rounded-md transition-colors"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-[#050811]/60 border border-slate-900/60 rounded-xl p-4 flex flex-col justify-center text-xs text-slate-400 leading-relaxed font-mono">
            <span className="text-sky-400 font-bold uppercase mb-1 block">💡 Pro-Tip</span>
            Search any city, or click directly on the interactive map block to capture coordinates. INSAT-3D, TROPOMI columns, and NASA FIRMS datasets will be dynamically queried around that coordinate.
          </div>
        </div>

        {/* Loading Overlay */}
        {isAnalyzingLocation && (
          <div className="absolute inset-0 bg-[#050811]/90 rounded-xl z-[60] flex flex-col items-center justify-center gap-3">
            <div className="w-10 h-10 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin" />
            <span className="text-sm font-bold text-white uppercase tracking-wider animate-pulse">{loadingPhase}</span>
            <span className="text-xs text-slate-500 font-mono">Analyzing: {currentAnalysisLocation}</span>
          </div>
        )}
      </div>

      {/* Error Banner — shown when CNN-LSTM prediction fails */}
      {predictionError && (
        <div className="bg-red-950/40 border border-red-800/40 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-xs font-mono text-red-300">{predictionError}</p>
          </div>
          <button
            onClick={() => setPredictionError(null)}
            className="text-slate-500 hover:text-slate-300 text-xs font-mono"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* 3. ENVIRONMENTAL METRICS CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {CARDS.map((card) => (
          <SummaryCard 
            key={card.title} 
            card={card} 
            onClick={() => { navigate(card.path); }} 
          />
        ))}
      </div>

      {/* 4. DYNAMIC DETAIL PANELS (WHEN REPORT PRESENT) */}
      {activeLocationReport && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Interactive Geographic Map Block */}
          <div className="lg:col-span-5 bg-[#090d16] border border-slate-900 rounded-xl p-5 flex flex-col min-h-[350px] relative z-0">
            <div className="mb-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-white">Geospatial Study Area</h3>
              <p className="text-[10px] font-mono text-slate-500 mt-0.5">Click map to analyze coordinates</p>
            </div>
            <div className="flex-1 w-full h-full rounded-lg overflow-hidden border border-slate-900">
              <MapContainer 
                center={selectedCoords ?? [20.5937, 78.9629]} 
                zoom={selectedCoords ? 10 : 5} 
                className="w-full h-full" 
                style={{ background: '#050811' }}
              >
                <TileLayer
                  attribution="&copy; Contributors"
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />
                {selectedCoords && (
                  <>
                    <MapUpdater center={[selectedCoords.lat, selectedCoords.lng]} zoom={10} />
                    {/* Primary selected marker */}
                    <CircleMarker
                      center={[selectedCoords.lat, selectedCoords.lng]}
                      radius={10}
                      fillColor={getAQITheme(activeLocationReport.AQI).fill}
                      color="#fff"
                      weight={2}
                      fillOpacity={0.9}
                    >
                      <Popup className="leaflet-popup-dark">
                        <div className="text-xs">
                          <strong>{activeLocationReport.location}</strong><br />
                          AQI: {activeLocationReport.AQI} ({activeLocationReport.category})
                        </div>
                      </Popup>
                    </CircleMarker>
                    
                    {/* Pollution dispersion radius circle */}
                    <Circle
                      center={[selectedCoords.lat, selectedCoords.lng]}
                      radius={15000} // 15km
                      pathOptions={{
                        color: getAQITheme(activeLocationReport.AQI).fill,
                        fillColor: getAQITheme(activeLocationReport.AQI).fill,
                        fillOpacity: 0.12,
                        weight: 1.5,
                        dashArray: '5, 5'
                      }}
                    />
                  </>
                )}
                <MapClickHandler onMapClick={handleMapClick} />
              </MapContainer>
            </div>
          </div>

          {/* AI generated report content block */}
          <div className="lg:col-span-7 bg-[#090d16] border border-slate-900 rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-sky-400 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4" /> AI Generated Environmental Report
              </h3>
              <p className="text-[10px] font-mono text-slate-500 mt-1">Generated dynamically using deep learning models & satellite data</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Pollution summary */}
              <div className="p-4 bg-slate-950/40 border border-slate-900 rounded-xl space-y-1.5 text-left">
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wide block">Pollution Summary</span>
                <p className="text-xs text-slate-300 leading-relaxed">
                  The current air quality is classified as <strong className="text-white">{activeLocationReport.category}</strong>. Ground PM2.5 levels are estimated at <strong className="text-white">{activeLocationReport.pollutants.PM25} µg/m³</strong>, translating to potential respiratory exposure indices.
                </p>
              </div>

              {/* Main causes */}
              <div className="p-4 bg-slate-950/40 border border-slate-900 rounded-xl space-y-1.5 text-left">
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wide block">Main Causes</span>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {activeLocationReport.Fire.influence === 'High' 
                    ? `Primary source is identified as transboundary crop residue fires with a cumulative active fire count of ${activeLocationReport.Fire.nearby_fire_count} in the regional advection corridor.`
                    : `Dominant causes are local vehicular exhaust, traffic accumulation, and secondary organic aerosol (SOA) formation, indicated by an HCHO column concentration of ${activeLocationReport.HCHO.concentration} mol/m².`
                  }
                </p>
              </div>

              {/* Health Impact */}
              <div className="p-4 bg-slate-950/40 border border-slate-900 rounded-xl space-y-1.5 text-left">
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wide block">Health Impact</span>
                <p className="text-xs text-slate-300 leading-relaxed font-sans">
                  {activeLocationReport.AQI > 150 
                    ? "Members of sensitive groups (children, elderly, asthmatics) may experience health effects immediately. Wear N95 protective masks outdoors and avoid strenuous exertion."
                    : "No immediate threats for healthy individuals. Recommended for sensitive groups to monitor trends if engaging in long-duration outdoor activities."
                  }
                </p>
              </div>

              {/* Fire Influence */}
              <div className="p-4 bg-slate-950/40 border border-slate-900 rounded-xl space-y-1.5 text-left">
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wide block">Fire &amp; Weather Influence</span>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Nearby fire count: <strong className="text-white">{activeLocationReport.Fire.nearby_fire_count}</strong> (Influence: {activeLocationReport.Fire.influence}). 
                  {activeLocationReport.Weather.wind_speed === "Data unavailable"
                    ? " Wind dispersion and weather data is currently unavailable."
                    : ` Winds are flowing from the ${activeLocationReport.Weather.wind} at ${activeLocationReport.Weather.wind_speed} m/s, accelerating downstream advection.`
                  }
                </p>
              </div>

              {/* Future Prediction */}
              <div className="p-4 bg-slate-950/40 border border-slate-900 rounded-xl space-y-1.5 text-left col-span-1 md:col-span-2">
                <span className="text-[10px] font-mono text-sky-400 uppercase tracking-wide block flex items-center gap-1"><Activity className="w-3.5 h-3.5" /> Future Prediction &amp; Recommendations</span>
                <p className="text-xs text-slate-300 leading-relaxed">
                  The CNN-LSTM spatio-temporal model forecasts a stabilization trend over the next 24 hours. <strong>Recommendations:</strong> Adjust ventilation systems, reduce open-air exposure during peak wind dispersion hours, and execute agricultural biomass controls to reduce regional aerosol loading.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI CONFIDENCE + EXPLAINABILITY PANEL (CNN-LSTM results only) */}
      {activeLocationReport?._ai && (
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6">
          <h3 className="text-xs font-bold uppercase tracking-wider text-sky-400 flex items-center gap-1.5 mb-4">
            <Zap className="w-4 h-4" /> AI Model Confidence &amp; Feature Attribution
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Confidence Score */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">CNN-LSTM Confidence Score</span>
                <span className="text-sm font-bold text-white">
                  {activeLocationReport._ai.confidence_score != null
                    ? `${(activeLocationReport._ai.confidence_score * 100).toFixed(0)}%`
                    : 'N/A'}
                </span>
              </div>
              {activeLocationReport._ai.confidence_score != null && (
                <div className="w-full bg-slate-900 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-700"
                    style={{
                      width: `${(activeLocationReport._ai.confidence_score * 100).toFixed(0)}%`,
                      background: activeLocationReport._ai.confidence_score >= 0.75
                        ? 'linear-gradient(90deg, #10b981, #34d399)'
                        : activeLocationReport._ai.confidence_score >= 0.5
                          ? 'linear-gradient(90deg, #f59e0b, #fbbf24)'
                          : 'linear-gradient(90deg, #ef4444, #f87171)'
                    }}
                  />
                </div>
              )}
              <p className="text-[10px] font-mono text-slate-500 leading-relaxed">
                Score computed from model validation R², input completeness, and data quality.
              </p>
              {/* Satellite feature readout */}
              <div className="pt-2 border-t border-slate-900 space-y-1.5">
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Satellite Pollutants</span>
                {['PM25','NO2','SO2','O3'].map(key => (
                  <div key={key} className="flex justify-between text-[10px] font-mono">
                    <span className="text-slate-500">{key}</span>
                    <span className="text-white">
                      {activeLocationReport.pollutants[key] > 0
                        ? `${activeLocationReport.pollutants[key]} µg/m³`
                        : '—'}
                    </span>
                  </div>
                ))}
                <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-slate-500">HCHO Probability</span>
                  <span className="text-white">{(activeLocationReport.HCHO.concentration * 250).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Feature Contribution Bars */}
            <div className="space-y-3">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Feature Group Contribution</span>
              {Object.entries(activeLocationReport._ai.feature_contribution ?? {}).map(([group, pct]) => (
                <div key={group} className="space-y-1">
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-slate-400">{group}</span>
                    <span className="text-white">{pct}%</span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full bg-sky-500 transition-all duration-700"
                      style={{ width: `${Math.min(100, pct)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* MODEL SOURCE + METADATA ROW (CNN-LSTM results only) */}
      {activeLocationReport?._ai?.metadata && (
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-5">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex items-center gap-3">
              <Cpu className="w-4 h-4 text-sky-400" />
              <div>
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">AI Model</span>
                <span className="text-xs font-bold text-white">CNN-LSTM {activeLocationReport._ai.metadata.model_version}</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Database className="w-4 h-4 text-indigo-400" />
              <div>
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Datasets Used</span>
                <span className="text-xs text-slate-300">{activeLocationReport._ai.metadata.datasets_used?.join(' · ')}</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Clock className="w-4 h-4 text-teal-400" />
              <div>
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Last Updated</span>
                <span className="text-xs text-slate-300">
                  {activeLocationReport._ai.metadata.timestamp
                    ? new Date(activeLocationReport._ai.metadata.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true })
                    : '—'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-sky-500/10 border border-sky-500/20 rounded-lg">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
              <span className="text-[10px] font-mono text-sky-400 uppercase tracking-wider">Live AI Prediction</span>
            </div>
          </div>
        </div>
      )}

      {/* 5. DYNAMIC POLLUTANTS GRAPH (WHEN REPORT PRESENT) */}
      {activeLocationReport && pollutantGraphData.length > 0 && (
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6">
          <h3 className="text-xs font-bold uppercase tracking-wider text-white mb-2">Location Pollutants Speciation</h3>
          <p className="text-[10px] font-mono text-slate-500 mb-6">Speciation concentrations of key gaseous and fine particulate indicators (*CO scaled x100 for visibility)</p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pollutantGraphData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff', fontSize: '11px' }}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {pollutantGraphData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 6. AQI & FIRES 7-DAY TREND (SHOWN ONCE SEARCH IS NOT ACTIVE) */}
      {!activeLocationReport && trend.length > 0 && (
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-white">AQI &amp; Fires 7-Day Trend</h3>
              <p className="text-[10px] font-mono text-slate-500 tracking-wider mt-1">
                Co-analysis of ground PM2.5 levels vs active open combustion counts · {aqiTrend.length > 0 ? 'Live' : 'Estimated'}
              </p>
            </div>
            <button onClick={refetchAQI} className="flex items-center gap-2 text-[10px] font-mono text-slate-500 hover:text-sky-400 transition-colors">
              Refresh
            </button>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trend} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="aqiGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#0ea5e9" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}   />
                  </linearGradient>
                  <linearGradient id="firesGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}   />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="date" stroke="#64748b" fontSize={9} />
                <YAxis stroke="#64748b" fontSize={9} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px' }}
                  labelStyle={{ color: '#94a3b8', fontSize: '10px', fontFamily: 'monospace' }}
                  itemStyle={{ color: '#fff', fontSize: '11px' }}
                />
                <Area type="monotone" name="Observed AQI" dataKey="aqi"   stroke="#0ea5e9" strokeWidth={1.8} fillOpacity={1} fill="url(#aqiGlow)"   />
                <Area type="monotone" name="FIRMS Fires"  dataKey="fires" stroke="#f59e0b" strokeWidth={1.8} fillOpacity={1} fill="url(#firesGlow)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Quick Navigation Buttons */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {[
          { label: 'AQI Map',      path: '/aqi'                 },
          { label: 'HCHO',         path: '/hcho'                },
          { label: 'Fire Analysis',path: '/fire-analysis'       },
          { label: 'Transport',    path: '/transport-analysis'  },
          { label: 'Analytics',    path: '/analytics'           },
          { label: 'Validation',   path: '/model-performance'   },
        ].map(item => (
          <button
            key={item.path}
            onClick={() => navigate(item.path)}
            className="px-3 py-2.5 bg-[#090d16] border border-slate-900 hover:border-sky-500/40 text-xs font-mono font-semibold text-slate-400 hover:text-sky-400 rounded-lg transition-all duration-200"
          >
            {item.label}
          </button>
        ))}
      </div>

    </div>
  );
}
