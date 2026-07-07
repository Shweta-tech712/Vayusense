import React, { createContext, useContext, useState, useCallback } from 'react';

const FilterContext = createContext();

export function FilterProvider({ children }) {
  const [analysisDate,      setAnalysisDate]      = useState(new Date().toISOString().split('T')[0]);
  const [selectedState,     setSelectedStateRaw]  = useState('All India');
  const [selectedDistrict,  setSelectedDistrict]  = useState('All Districts');
  const [selectedPollutant, setSelectedPollutant] = useState('PM2.5');
  const [minFRP,            setMinFRP]            = useState(20);
  const [hchoThreshold,     setHchoThreshold]     = useState(2.0);
  const [isSidebarCollapsed,setIsSidebarCollapsed]= useState(false);
  const [searchQuery,       setSearchQuery]       = useState('');
  
  // AI Location Intelligence Central States
  const [activeLocationReport, setActiveLocationReport] = useState(null);
  const [selectedCoords, setSelectedCoords] = useState(null);
  const [recentSearches, setRecentSearches] = useState(['Delhi', 'Mumbai', 'Pune', 'Punjab']);
  const [isAnalyzingLocation, setIsAnalyzingLocation] = useState(false);

  const toggleSidebar = useCallback(() => {
    setIsSidebarCollapsed((prev) => !prev);
  }, []);

  // Reset district when state changes
  const setSelectedState = useCallback((state) => {
    setSelectedStateRaw(state);
    setSelectedDistrict('All Districts');
  }, []);

  const analyzeLocation = useCallback(async (name, lat, lng) => {
    setIsAnalyzingLocation(true);
    const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
    try {
      const response = await fetch(`${API_BASE}/location-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: name,
          latitude: lat,
          longitude: lng,
          date: 'current'
        })
      });
      if (response.ok) {
        const data = await response.json();
        setActiveLocationReport(data);
        
        // Save matching coords
        // If lat/lon were looked up by backend, use them, otherwise use the passed ones
        const resolvedLat = lat ?? (data.location.includes('(') ? parseFloat(data.location.split('(')[1].split('°N')[0]) : null);
        const resolvedLng = lng ?? (data.location.includes('(') ? parseFloat(data.location.split('°N, ')[1].split('°E')[0]) : null);
        
        // Default fallbacks from common cities if resolved is null
        const cityCoords = {
          "Delhi, National Capital Territory": { lat: 28.6139, lng: 77.2090 },
          "Mumbai, Maharashtra": { lat: 19.0760, lng: 72.8777 },
          "Pune, Maharashtra": { lat: 18.5204, lng: 73.8567 },
          "Punjab, India": { lat: 31.1471, lng: 75.3412 },
          "Bengaluru, Karnataka": { lat: 12.9716, lng: 77.5946 },
          "Kolkata, West Bengal": { lat: 22.5726, lng: 88.3639 },
          "Chennai, Tamil Nadu": { lat: 13.0827, lng: 80.2707 },
          "Hyderabad, Telangana": { lat: 17.3850, lng: 78.4867 },
          "Patna, Bihar": { lat: 25.5941, lng: 85.1376 },
          "Indore, Madhya Pradesh": { lat: 22.7196, lng: 75.8577 },
          "Bhopal, Madhya Pradesh": { lat: 23.2599, lng: 77.4126 }
        };

        let finalLat = resolvedLat ?? 20.5937;
        let finalLng = resolvedLng ?? 78.9629;
        
        if (cityCoords[data.location]) {
          finalLat = cityCoords[data.location].lat;
          finalLng = cityCoords[data.location].lng;
        }

        setSelectedCoords({ lat: finalLat, lng: finalLng });

        // Update recent searches
        setRecentSearches(prev => {
          const cleanName = data.location.split('(')[0].trim();
          const filtered = prev.filter(item => item.toLowerCase() !== cleanName.toLowerCase());
          return [cleanName, ...filtered].slice(0, 5);
        });
      } else {
        console.error("Failed to fetch location analysis");
      }
    } catch (err) {
      console.error("Error during location analysis:", err);
    } finally {
      setIsAnalyzingLocation(false);
    }
  }, []);

  const value = {
    analysisDate,      setAnalysisDate,
    selectedState,     setSelectedState,
    selectedDistrict,  setSelectedDistrict,
    selectedPollutant, setSelectedPollutant,
    minFRP,            setMinFRP,
    hchoThreshold,     setHchoThreshold,
    isSidebarCollapsed,setIsSidebarCollapsed,
    toggleSidebar,
    searchQuery,       setSearchQuery,
    
    // AI Location intelligence exports
    activeLocationReport, setActiveLocationReport,
    selectedCoords, setSelectedCoords,
    recentSearches, setRecentSearches,
    isAnalyzingLocation,
    analyzeLocation
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
