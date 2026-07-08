import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { getAQIPrediction, PREDICTION_ERRORS } from '../services/predictionApi';

const FilterContext = createContext();

// ─── localStorage helpers ─────────────────────────────────────────────────────
const LS_KEY = 'vayusense_recent_searches';
const MAX_RECENT = 5;

function loadRecentSearches() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? JSON.parse(raw) : ['Delhi', 'Mumbai', 'Pune', 'Punjab'];
  } catch {
    return ['Delhi', 'Mumbai', 'Pune', 'Punjab'];
  }
}

function saveRecentSearches(list) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(list));
  } catch {
    // Ignore storage quota errors
  }
}

// ─── AQI category helper ──────────────────────────────────────────────────────
function hchoConcentrationFromProbability(prob) {
  // Reverse of: prob = clip(HCHO / 0.004, 0, 1)
  return parseFloat((prob * 0.004).toFixed(6));
}

// ─── Response normaliser ──────────────────────────────────────────────────────
/**
 * Converts the CNN-LSTM /api/predict/location response into the shape that
 * HomeView and other pages already expect from activeLocationReport.
 *
 * Existing shape (from /api/location-analysis):
 * { location, AQI, category, pollutants:{PM25,NO2,SO2,CO,O3},
 *   HCHO:{concentration,risk,hotspot}, Fire:{nearby_fire_count,influence,distance_km},
 *   Weather:{temperature,humidity,wind,wind_speed}, AI_Analysis }
 *
 * Extended shape adds:  _ai: { confidence_score, explainability, metadata, satellite, recommendation }
 */
function normalizeAIResponse(data) {
  const loc  = data.location  ?? {};
  const pred = data.prediction ?? {};
  const env  = data.environment ?? {};
  const expl = data.explainability ?? {};
  const meta = data.metadata ?? {};
  const sat  = data.satellite_features ?? {};   // populated when backend returns them

  const hchoProb = pred.HCHO_probability ?? 0;
  const hchoConc = hchoConcentrationFromProbability(hchoProb);

  return {
    // ── Core fields HomeView reads ───────────────────────────────────────────
    location: loc.name
      ? `${loc.name}${loc.state && loc.state !== 'Unknown' ? ', ' + loc.state : ''}`
      : 'Unknown Location',

    AQI:      Math.round(pred.AQI ?? 0),
    category: pred.category ?? 'Unknown',

    // Satellite pollutant features — use real values if returned, else 0
    pollutants: {
      PM25: +(pred.PM25 ?? sat.PM25 ?? 0).toFixed(1),
      NO2:  +(sat.NO2  ?? 0).toFixed(1),
      SO2:  +(sat.SO2  ?? 0).toFixed(1),
      CO:   +(sat.CO   ?? 0).toFixed(4),
      O3:   +(sat.O3   ?? 0).toFixed(1),
    },

    HCHO: {
      concentration: hchoConc,
      risk:    pred.HCHO_risk ?? 'Low',
      hotspot: hchoProb >= 0.7,
    },

    Fire: {
      nearby_fire_count: 0,           // not returned by CNN model; shown as N/A in UI
      influence:   env.fire_influence ?? 'Unknown',
      distance_km: 0,
    },

    Weather: {
      temperature: 0,                 // not in CNN response; shown as N/A
      humidity:    0,
      wind:        env.wind_transport ?? '—',
      wind_speed:  0,
    },

    AI_Analysis: data.recommendation ?? '',

    // ── Extended AI fields (picked up by new panels only) ───────────────────
    _ai: {
      confidence_score:      pred.confidence_score ?? null,
      feature_contribution:  expl.feature_contribution ?? {},
      aerosol_level:         env.aerosol_level ?? '—',
      recommendation:        data.recommendation ?? '',
      metadata: {
        model_version:    meta.model_version   ?? 'v1.0.0',
        dataset_version:  meta.dataset_version ?? 'v1',
        timestamp:        meta.prediction_timestamp ?? null,
        datasets_used:    ['Sentinel-5P', 'INSAT-3D', 'ERA5', 'NASA FIRMS', 'CPCB'],
      },
      // raw coords for map auto-zoom
      latitude:  loc.latitude  ?? null,
      longitude: loc.longitude ?? null,
    },
  };
}

const IS_PRODUCTION = import.meta.env.PROD === true;

// ─── Provider ─────────────────────────────────────────────────────────────────
export function FilterProvider({ children }) {
  const [analysisDate,       setAnalysisDate]      = useState(new Date().toISOString().split('T')[0]);
  const [selectedState,      setSelectedStateRaw]  = useState('All India');
  const [selectedDistrict,   setSelectedDistrict]  = useState('All Districts');
  const [selectedPollutant,  setSelectedPollutant] = useState('PM2.5');
  const [minFRP,             setMinFRP]            = useState(20);
  const [hchoThreshold,      setHchoThreshold]     = useState(2.0);
  const [isSidebarCollapsed, setIsSidebarCollapsed]= useState(false);
  const [searchQuery,        setSearchQuery]       = useState('');

  // AI Location Intelligence
  const [activeLocationReport, setActiveLocationReport] = useState(null);
  const [selectedCoords,       setSelectedCoords]       = useState(null);
  const [recentSearches,       setRecentSearches]       = useState(loadRecentSearches);
  const [isAnalyzingLocation,  setIsAnalyzingLocation]  = useState(false);
  const [predictionError,      setPredictionError]      = useState(null);

  // States to prevent location mismatch and show loading details
  const [currentAnalysisLocation, setCurrentAnalysisLocation] = useState('');
  const [loadingPhase,            setLoadingPhase]            = useState('');
  
  const abortControllerRef = useRef(null);

  const toggleSidebar = useCallback(() => {
    setIsSidebarCollapsed(prev => !prev);
  }, []);

  const setSelectedState = useCallback((state) => {
    setSelectedStateRaw(state);
    setSelectedDistrict('All Districts');
  }, []);

  /**
   * Main prediction entry point.
   * 1. Calls CNN-LSTM /api/predict/location
   * 2. Normalizes response
   * 3. Falls back to /api/location-analysis ONLY in development mode
   * 4. In production: shows typed error — no silent mock data
   */
  const analyzeLocation = useCallback(async (name, lat, lng) => {
    // 1. Cancel previous pending searches to prevent race conditions
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // 2. Clear old state immediately to ensure a fresh request state
    setActiveLocationReport(null);
    setSelectedCoords(null);
    setPredictionError(null);
    setCurrentAnalysisLocation(name);
    setIsAnalyzingLocation(true);
    setLoadingPhase('Initializing satellite scan...');

    // 3. Sequential loading experience update loop
    const loadingSteps = [
      'Initializing satellite scan...',
      'Fetching environmental parameters...',
      'Running CNN-LSTM prediction...',
      'Generating AQI intelligence report...'
    ];
    let stepIndex = 0;
    const intervalId = setInterval(() => {
      stepIndex++;
      if (stepIndex < loadingSteps.length) {
        setLoadingPhase(loadingSteps[stepIndex]);
      }
    }, 1000);

    const cleanUpRequest = () => {
      clearInterval(intervalId);
      if (abortControllerRef.current === controller) {
        setIsAnalyzingLocation(false);
      }
    };

    const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

    try {
      // ── 1. Try CNN-LSTM prediction endpoint ───────────────────────────────────
      const data = await getAQIPrediction(name, lat ?? null, lng ?? null, controller.signal);
      
      if (controller.signal.aborted) {
        cleanUpRequest();
        return;
      }

      // Check: Validate that API response location matches requested location
      const returnedLocName = data.location?.name ?? '';
      const queryNameLower = name.toLowerCase().trim();
      const isPinpoint = queryNameLower.includes("pinpoint");
      const isLocationMatched = isPinpoint ||
                                returnedLocName.toLowerCase().includes(queryNameLower) ||
                                queryNameLower.includes(returnedLocName.toLowerCase());

      if (!isLocationMatched) {
        console.warn(`Ignoring stale response: target was "${name}", but response returned "${returnedLocName}"`);
        cleanUpRequest();
        return;
      }

      const normalized = normalizeAIResponse(data);
      setActiveLocationReport(normalized);

      // Update map coords from response or explicit args
      const finalLat = normalized._ai.latitude  ?? lat  ?? 20.5937;
      const finalLng = normalized._ai.longitude ?? lng  ?? 78.9629;
      setSelectedCoords({ lat: finalLat, lng: finalLng });

      // Update localStorage recent searches
      const displayName = (normalized.location ?? name).split(',')[0].trim();
      setRecentSearches(prev => {
        const next = [displayName, ...prev.filter(x => x.toLowerCase() !== displayName.toLowerCase())]
          .slice(0, MAX_RECENT);
        saveRecentSearches(next);
        return next;
      });

    } catch (err) {
      if (controller.signal.aborted || err.name === 'CanceledError' || err.name === 'AbortError') {
        cleanUpRequest();
        return;
      }

      // ── 2. Development fallback to legacy endpoint ───────────────────────────
      if (!IS_PRODUCTION) {
        console.warn(`CNN-LSTM prediction failed (${err.type}): ${err.message}. Falling back to /location-analysis in dev mode.`);
        try {
          const resp = await fetch(`${API_BASE}/location-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location: name, latitude: lat, longitude: lng, date: 'current' }),
            signal: controller.signal
          });
          
          if (controller.signal.aborted) {
            cleanUpRequest();
            return;
          }

          if (resp.ok) {
            const legacy = await resp.json();
            
            // Double check location validity
            const fallbackLocName = legacy.location ?? '';
            const isFallbackMatched = isPinpoint ||
                                      fallbackLocName.toLowerCase().includes(queryNameLower) ||
                                      queryNameLower.includes(fallbackLocName.toLowerCase());
                                      
            if (!isFallbackMatched) {
              cleanUpRequest();
              return;
            }

            setActiveLocationReport(legacy);

            // Resolve coords from legacy city lookup
            const cityCoords = {
              "Delhi, National Capital Territory": { lat: 28.6139, lng: 77.2090 },
              "Mumbai, Maharashtra":               { lat: 19.0760, lng: 72.8777 },
              "Pune, Maharashtra":                 { lat: 18.5204, lng: 73.8567 },
              "Punjab, India":                     { lat: 31.1471, lng: 75.3412 },
              "Bengaluru, Karnataka":              { lat: 12.9716, lng: 77.5946 },
              "Kolkata, West Bengal":              { lat: 22.5726, lng: 88.3639 },
              "Chennai, Tamil Nadu":               { lat: 13.0827, lng: 80.2707 },
              "Hyderabad, Telangana":              { lat: 17.3850, lng: 78.4867 },
              "Patna, Bihar":                      { lat: 25.5941, lng: 85.1376 },
              "Indore, Madhya Pradesh":            { lat: 22.7196, lng: 75.8577 },
              "Bhopal, Madhya Pradesh":            { lat: 23.2599, lng: 77.4126 },
            };
            const coords = cityCoords[legacy.location];
            setSelectedCoords(coords ?? { lat: lat ?? 20.5937, lng: lng ?? 78.9629 });
          } else {
            setPredictionError("AI analysis unavailable for this location");
          }
        } catch (fallbackErr) {
          if (!controller.signal.aborted) {
            setPredictionError("AI analysis unavailable for this location");
          }
        }
      } else {
        // ── 3. Production: show typed error, no mock data ──────────────────────
        setPredictionError("AI analysis unavailable for this location");
      }
    } finally {
      cleanUpRequest();
    }
  }, []);

  const value = {
    analysisDate,       setAnalysisDate,
    selectedState,      setSelectedState,
    selectedDistrict,   setSelectedDistrict,
    selectedPollutant,  setSelectedPollutant,
    minFRP,             setMinFRP,
    hchoThreshold,      setHchoThreshold,
    isSidebarCollapsed, setIsSidebarCollapsed,
    toggleSidebar,
    searchQuery,        setSearchQuery,

    // AI Location intelligence
    activeLocationReport,    setActiveLocationReport,
    selectedCoords,          setSelectedCoords,
    recentSearches,          setRecentSearches,
    isAnalyzingLocation,
    analyzeLocation,
    predictionError,         setPredictionError,
    currentAnalysisLocation,
    loadingPhase,
  };

  return (
    <FilterContext.Provider value={value}>
      {children}
    </FilterContext.Provider>
  );
}

export function useFilters() {
  const context = useContext(FilterContext);
  if (!context) throw new Error('useFilters must be used within a FilterProvider');
  return context;
}
