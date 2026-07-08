import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useFilters } from '../../context/FilterContext';
import ToastNotification from '../Common/ToastNotification';
import { 
  ChevronLeft, ChevronRight, Home, Map, Flame, Wind, 
  BarChart3, Brain, Database, BookOpen, Info, Settings, Search, Eye, Wifi, WifiOff
} from 'lucide-react';

const SIDEBAR_ITEMS = [
  { label: 'Home', path: '/dashboard', icon: Home },
  { label: 'AQI Map', path: '/aqi', icon: Map },
  { label: 'HCHO Hotspots', path: '/hcho', icon: Eye },
  { label: 'Fire Analysis', path: '/fire-analysis', icon: Flame },
  { label: 'Transport', path: '/transport-analysis', icon: Wind },
  { label: 'Analytics', path: '/analytics', icon: BarChart3 },
  { label: 'Model Validation', path: '/model-performance', icon: Brain },
  { label: 'Dataset Explorer', path: '/dataset-explorer', icon: Database },
  { label: 'Methodology', path: '/methodology', icon: BookOpen },
  { label: 'About', path: '/about', icon: Info },
  { label: 'Settings', path: '/settings', icon: Settings }
];

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { 
    analysisDate, setAnalysisDate,
    selectedState, setSelectedState,
    isSidebarCollapsed, toggleSidebar,
    searchQuery, setSearchQuery
  } = useFilters();

  // Backend connectivity probe – pings /api/health every 30s
  const [backendOnline, setBackendOnline] = useState(null);
  useEffect(() => {
    const API_BASE = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        setBackendOnline(res.ok);
      } catch {
        setBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  // Mock Indian States for filter
  const INDIAN_STATES = [
    'All India', 'Delhi', 'Maharashtra', 'Karnataka', 
    'West Bengal', 'Tamil Nadu', 'Telangana', 'Bihar'
  ];

  return (
    <div className="flex min-h-screen bg-[#050811] text-white font-sans overflow-hidden select-none">
      
      {/* 1. COLLAPSIBLE SIDEBAR */}
      <aside 
        className={`bg-[#090d16] border-r border-slate-900 flex flex-col justify-between transition-all duration-300 ${
          isSidebarCollapsed ? 'w-20' : 'w-64'
        }`}
      >
        <div>
          {/* Logo & Header */}
          <div className="flex items-center space-x-3 p-6 border-b border-slate-900/60 cursor-pointer" onClick={() => navigate('/')}>
            <div className="w-8 h-8 flex items-center justify-center bg-sky-950/40 border border-sky-800/40 rounded-lg shadow-md shrink-0">
              <svg viewBox="0 0 100 100" className="w-6 h-6 fill-sky-400 stroke-none">
                <path d="M63.8 38.2 C57 32 46.2 31.8 37.5 37 C29.5 41.8 28.2 50.8 33.8 55.2 C35.2 56.2 36 55.5 35.8 54 C33.8 45 42.5 40.5 58.5 39 C61.8 38.7 64 38.5 63.8 38.2 Z" />
                <path d="M36.2 61.8 C43 68 53.8 68.2 62.5 63 C70.5 58.2 71.8 49.2 66.2 44.8 C64.8 43.8 64 44.5 64.2 46 C66.2 55 57.5 59.5 41.5 61 C38.2 61.3 36 61.5 36.2 61.8 Z" />
                <path d="M24 67.5 C36.5 60 52 50.5 73.5 36.5 C75 35.5 75.5 35.8 74.2 37.2 C53.5 55.5 37.5 66.8 25 70 C24 70.3 23.5 69.5 24 67.5 Z" />
                <path d="M75.5 35.5 L73 37.2 L74 38.8 L77 37 L79.8 38.6 L83.5 34.5 L79.5 30.8 L75.5 34.5 Z" />
              </svg>
            </div>
            {!isSidebarCollapsed && (
              <div className="flex flex-col text-left">
                <span className="text-sm font-bold uppercase tracking-wide text-white leading-none">Vayusense</span>
                <span className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">ISRO SAC Portal</span>
              </div>
            )}
          </div>

          {/* Links list */}
          <nav className="p-4 space-y-1">
            {SIDEBAR_ITEMS.map((item) => {
              const IconComponent = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <button
                  key={item.label}
                  onClick={() => navigate(item.path)}
                  className={`w-full flex items-center space-x-4 px-4 py-3 rounded-lg text-sm font-semibold tracking-wide transition-all duration-300 ${
                    isActive 
                      ? 'bg-sky-600/15 border border-sky-500/25 text-sky-400' 
                      : 'border border-transparent text-slate-400 hover:text-white hover:bg-slate-900/30'
                  }`}
                  title={isSidebarCollapsed ? item.label : ''}
                >
                  <IconComponent className="w-5 h-5 shrink-0" />
                  {!isSidebarCollapsed && <span>{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Collapser Toggle Trigger */}
        <div className="p-4 border-t border-slate-900/60">
          <button
            onClick={toggleSidebar}
            className="w-full flex items-center justify-center py-2.5 bg-slate-900/40 border border-slate-850 hover:border-slate-750 text-slate-400 hover:text-white rounded-lg transition-all duration-300"
          >
            {isSidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>
      </aside>

      {/* 2. DYNAMIC WORKSPACE BODY CONTAINER */}
      <div className="flex-1 flex flex-col min-w-0">
        
        {/* Top filter navbar bar */}
        <header className="h-20 bg-[#090d16] border-b border-slate-900 px-6 flex items-center justify-between z-40 shrink-0">
          
          {/* Left placeholder to preserve space structure */}
          <div />

          {/* Right Filters Control center */}
          <div className="flex items-center space-x-4">

            {/* Backend status badge */}
            {backendOnline !== null && (
              <div
                className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-semibold"
                style={{
                  backgroundColor: backendOnline ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)',
                  border: `1px solid ${backendOnline ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)'}`,
                  color: backendOnline ? '#10b981' : '#ef4444',
                }}
                title={backendOnline ? 'Backend API connected' : 'Backend offline — using mock data'}
              >
                {backendOnline
                  ? <><Wifi className="w-3 h-3" /> API LIVE</>
                  : <><WifiOff className="w-3 h-3" /> OFFLINE</>}
              </div>
            )}
            
            {/* State filter selector */}
            <select
              value={selectedState}
              onChange={(e) => setSelectedState(e.target.value)}
              className="bg-[#050811] border border-slate-850 hover:border-slate-750 px-3 py-2 rounded-lg text-xs font-semibold text-slate-300 focus:outline-none cursor-pointer"
            >
              {INDIAN_STATES.map((st) => (
                <option key={st} value={st}>{st}</option>
              ))}
            </select>

            {/* Date filter picker */}
            <input
              type="date"
              value={analysisDate}
              onChange={(e) => setAnalysisDate(e.target.value)}
              className="bg-[#050811] border border-slate-850 hover:border-slate-750 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-300 focus:outline-none cursor-pointer"
            />

          </div>
        </header>

        {/* Dynamic Nested Output pane with smooth Page Transitions */}
        <main className="flex-1 overflow-y-auto p-8 relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.22, ease: 'easeInOut' }}
              className="h-full"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>

      </div>

      {/* Global toast notification HUD */}
      <ToastNotification />

    </div>
  );
}
