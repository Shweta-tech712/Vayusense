import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

const NAV_LINKS = [
  { label: 'Home', path: '/' },
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'AQI Map', path: '/aqi' },
  { label: 'HCHO Hotspots', path: '/hcho' },
  { label: 'Fire Analysis', path: '/fire-analysis' },
  { label: 'Transport', path: '/transport-analysis' },
  { label: 'Analytics', path: '/analytics' },
  { label: 'About', path: '/about' }
];

export default function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 30);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav 
      className={`fixed top-0 left-0 w-full z-50 transition-all duration-500 border-b select-none ${
        isScrolled 
          ? 'bg-[#050811]/80 border-slate-900/60 backdrop-blur-md py-4' 
          : 'bg-transparent border-transparent py-6'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 md:px-12 flex items-center justify-between">
        
        {/* Brand Logo & Name */}
        <div className="flex items-center space-x-3 cursor-pointer group" onClick={() => navigate('/')}>
          <div className="w-9 h-9 flex items-center justify-center bg-sky-950/40 border border-sky-800/40 rounded-lg group-hover:border-amber-500/50 transition-all duration-300 shadow-[0_0_12px_rgba(56,189,248,0.08)]">
            <svg viewBox="0 0 100 100" className="w-6 h-6 fill-sky-400 stroke-none group-hover:fill-amber-500 transition-colors duration-300">
              <path d="M63.8 38.2 C57 32 46.2 31.8 37.5 37 C29.5 41.8 28.2 50.8 33.8 55.2 C35.2 56.2 36 55.5 35.8 54 C33.8 45 42.5 40.5 58.5 39 C61.8 38.7 64 38.5 63.8 38.2 Z" />
              <path d="M36.2 61.8 C43 68 53.8 68.2 62.5 63 C70.5 58.2 71.8 49.2 66.2 44.8 C64.8 43.8 64 44.5 64.2 46 C66.2 55 57.5 59.5 41.5 61 C38.2 61.3 36 61.5 36.2 61.8 Z" />
              <path d="M24 67.5 C36.5 60 52 50.5 73.5 36.5 C75 35.5 75.5 35.8 74.2 37.2 C53.5 55.5 37.5 66.8 25 70 C24 70.3 23.5 69.5 24 67.5 Z" />
              <path d="M75.5 35.5 L73 37.2 L74 38.8 L77 37 L79.8 38.6 L83.5 34.5 L79.5 30.8 L75.5 34.5 Z" />
            </svg>
          </div>
          <div className="flex flex-col text-left">
            <span className="text-base font-extrabold tracking-widest uppercase text-white leading-none">Vayusense</span>
            <span className="text-[11px] font-mono tracking-wider text-slate-350 uppercase mt-0.5">ISRO SAC Portal</span>
          </div>
        </div>

        {/* Desktop Links */}
        <div className="hidden lg:flex items-center space-x-8">
          {NAV_LINKS.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <button 
                key={link.label}
                onClick={() => navigate(link.path)}
                className={`text-[15px] font-semibold tracking-wide transition-all duration-350 cursor-pointer bg-transparent border-none focus:outline-none ${
                  isActive 
                    ? 'text-sky-400 underline underline-offset-8 decoration-sky-400/80 decoration-2' 
                    : 'text-slate-350 hover:text-white'
                }`}
              >
                {link.label}
              </button>
            );
          })}
        </div>

        {/* Action Link & Mobile Toggle */}
        <div className="flex items-center space-x-4">
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noreferrer"
            className="hidden sm:inline-flex items-center space-x-2 border border-slate-800 hover:border-amber-500/50 bg-[#090d16]/40 hover:bg-[#0c121e]/60 px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider text-slate-300 hover:text-white transition-all duration-300 shadow-sm"
          >
            <span>GitHub</span>
          </a>

          {/* Hamburger Icon */}
          <button 
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-1.5 rounded-lg bg-slate-900/40 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 lg:hidden transition-all duration-300"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

      </div>

      {/* Mobile Drawer menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="w-full bg-[#050811]/95 border-b border-slate-900/60 backdrop-blur-lg lg:hidden"
          >
            <div className="px-6 py-8 flex flex-col space-y-4 text-left">
              {NAV_LINKS.map((link) => {
                const isActive = location.pathname === link.path;
                return (
                  <button 
                    key={link.label}
                    onClick={() => {
                      setMobileMenuOpen(false);
                      navigate(link.path);
                    }}
                    className={`text-sm font-medium tracking-wide transition-colors duration-300 block text-left border-none bg-transparent focus:outline-none ${
                      isActive ? 'text-sky-400' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {link.label}
                  </button>
                );
              })}
              <hr className="border-slate-900/60" />
              <a 
                href="https://github.com" 
                target="_blank" 
                rel="noreferrer"
                className="w-full text-center py-3 border border-slate-850 hover:border-amber-500/50 rounded-lg text-xs font-bold uppercase tracking-wider text-slate-300 hover:text-white block transition-all duration-300"
              >
                GitHub Repository
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
