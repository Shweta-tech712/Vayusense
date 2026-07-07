import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldAlert } from 'lucide-react';

export default function NotFound() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#050811] text-white p-8 select-none text-center">
      <ShieldAlert className="w-16 h-16 text-sky-500 mb-6 animate-pulse" />
      <h2 className="text-2xl font-bold uppercase tracking-widest text-white mb-2">404 - Area Mass Unmapped</h2>
      <p className="text-xs text-slate-500 font-mono tracking-wide max-w-sm mb-8 leading-relaxed">
        The geospatial coordinate path you requested is outside the mission operational boundaries.
      </p>
      <button
        onClick={() => navigate('/')}
        className="px-8 py-3 bg-sky-600 hover:bg-sky-500 text-xs font-bold uppercase tracking-widest text-white rounded-xl transition-all duration-300 shadow-[0_4px_25px_rgba(14,165,233,0.15)]"
      >
        Return to Mission Control
      </button>
    </div>
  );
}
