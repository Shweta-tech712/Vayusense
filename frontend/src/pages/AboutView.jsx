import React from 'react';
import { motion } from 'framer-motion';
import { 
  Globe, Sparkles, Database, Landmark, Heart, Cpu, 
  Activity, ShieldAlert, Award, Compass, ShieldCheck, Flame 
} from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: 'easeOut'
    }
  }
};

export default function AboutView() {
  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-10 select-none text-left max-w-5xl pb-12"
    >
      
      {/* 1. PREMIUM HEADER SECTION */}
      <motion.div variants={itemVariants} className="relative z-10">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-950/40 border border-sky-800/35 text-sky-400 text-[10px] font-mono uppercase tracking-widest mb-3 shadow-[0_0_12px_rgba(56,189,248,0.06)]">
          <Sparkles className="w-3.5 h-3.5" /> Mission Specifications
        </div>
        <h2 className="text-3xl font-extrabold text-white tracking-tight uppercase">
          About Vayusense
        </h2>
        <p className="text-xs text-slate-400 font-mono tracking-wider mt-1.5 max-w-2xl leading-relaxed">
          The central remote sensing command center for surface AQI calculations &amp; Formaldehyde hotspot diagnostics across India.
        </p>
      </motion.div>

      {/* 2. PREMIUM HERO OVERVIEW CARD */}
      <motion.div 
        variants={itemVariants}
        className="relative group bg-[#090d16]/80 border border-slate-900 rounded-2xl p-8 overflow-hidden backdrop-blur-md shadow-2xl transition-all duration-300 hover:border-slate-800"
      >
        {/* Glow effect */}
        <div className="absolute -top-[120px] -right-[120px] w-[240px] h-[240px] rounded-full blur-[100px] bg-sky-500/10 pointer-events-none transition-all duration-500 group-hover:bg-sky-500/15" />
        
        <div className="relative z-10 flex flex-col md:flex-row gap-6 items-start">
          <div className="p-3.5 rounded-xl bg-sky-950/40 border border-sky-850 shadow-inner">
            <Globe className="w-8 h-8 text-sky-400" />
          </div>
          <div className="space-y-3 flex-1">
            <h3 className="text-lg font-bold text-white tracking-wide flex items-center gap-2">
              The Vayusense Project
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              <strong>Vayusense</strong> is a state-of-the-art environmental intelligence platform developed for the **Development of Surface AQI &amp; Identification of HCHO Hotspots over India using Satellite Data**. 
            </p>
            <p className="text-sm text-slate-350 leading-relaxed font-sans">
              By fusing spaceborne optical depth retrieval instruments, tropospheric volatile gaseous trace measurements, and boundary meteorological vector fields, Vayusense provides real-time pollution speciation, agricultural stubble fire tracking, and deep learning-based air quality predictions.
            </p>
          </div>
        </div>
      </motion.div>

      {/* 3. CORE FRAMEWORK GRID */}
      <motion.div variants={itemVariants} className="space-y-4">
        <h3 className="text-xs font-mono tracking-widest text-slate-400 uppercase">Scientific Architecture</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          {/* Deep Learning Core */}
          <div className="bg-[#090d16]/60 border border-slate-900 rounded-xl p-6 space-y-4 hover:border-slate-800 transition-all duration-300 flex flex-col justify-between">
            <div className="space-y-2">
              <div className="w-10 h-10 flex items-center justify-center rounded-lg bg-sky-950/40 border border-sky-900/30 text-sky-400">
                <Cpu className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-bold text-white uppercase tracking-wider">Deep Learning Predictor</h4>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
              Utilizes a hybrid CNN-LSTM model processing 11x11 grid spatial patches and a 7-day temporal lag (T=7) to predict ground-level Surface AQI, preventing data leakage.
            </p>
          </div>

          {/* Data Fusion Core */}
          <div className="bg-[#090d16]/60 border border-slate-900 rounded-xl p-6 space-y-4 hover:border-slate-800 transition-all duration-300 flex flex-col justify-between">
            <div className="space-y-2">
              <div className="w-10 h-10 flex items-center justify-center rounded-lg bg-indigo-950/40 border border-indigo-900/30 text-indigo-400">
                <Database className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-bold text-white uppercase tracking-wider">Multi-Sensor Fusion</h4>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
              Consumes datasets from INSAT-3D/MODIS Aerosol Depth, TROPOMI columns (HCHO, NO₂, SO₂, CO, O₃), NASA FIRMS thermal points, and ERA5 winds.
            </p>
          </div>

          {/* Meteorological Core */}
          <div className="bg-[#090d16]/60 border border-slate-900 rounded-xl p-6 space-y-4 hover:border-slate-800 transition-all duration-300 flex flex-col justify-between">
            <div className="space-y-2">
              <div className="w-10 h-10 flex items-center justify-center rounded-lg bg-teal-950/40 border border-teal-900/30 text-teal-400">
                <Compass className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-bold text-white uppercase tracking-wider">Advection Tracer</h4>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
              Traces crop-residue open combustion plumes using ERA5 boundary layer wind velocity components to map dispersal vectors across state lines.
            </p>
          </div>

        </div>
      </motion.div>

      {/* 4. MISSION METRICS */}
      <motion.div 
        variants={itemVariants} 
        className="grid grid-cols-2 md:grid-cols-4 gap-6 bg-slate-950/30 border border-slate-900/80 rounded-2xl p-6"
      >
        <div className="text-center md:text-left space-y-1">
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">CPCB Stations</span>
          <h4 className="text-2xl font-black text-white">500+</h4>
          <span className="text-[9px] font-mono text-sky-500">Validation Points</span>
        </div>
        <div className="text-center md:text-left space-y-1 border-l border-slate-900 pl-4">
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">S5P TROPOMI</span>
          <h4 className="text-2xl font-black text-white">Daily</h4>
          <span className="text-[9px] font-mono text-indigo-400">Trace Gas Mappings</span>
        </div>
        <div className="text-center md:text-left space-y-1 border-l border-slate-900 pl-4">
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">FRP Threshold</span>
          <h4 className="text-2xl font-black text-white">&gt;70%</h4>
          <span className="text-[9px] font-mono text-amber-500">Thermal Confidence</span>
        </div>
        <div className="text-center md:text-left space-y-1 border-l border-slate-900 pl-4">
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Grid Scale</span>
          <h4 className="text-2xl font-black text-white">10 km</h4>
          <span className="text-[9px] font-mono text-teal-400">Spatial Study Area</span>
        </div>
      </motion.div>

      {/* 5. METADATA CREDITS */}
      <motion.div 
        variants={itemVariants}
        className="border-t border-slate-900/60 pt-8 grid grid-cols-1 md:grid-cols-2 gap-8 text-xs text-slate-400 leading-relaxed"
      >
        <div className="flex gap-3">
          <Landmark className="w-5 h-5 text-slate-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="text-slate-200 font-bold uppercase tracking-wider text-[11px] block">Lead Agency</span>
            <p className="font-sans text-slate-400">Space Applications Centre (ISRO SAC), Ahmedabad, Gujarat. Fostering aerospace engineering architectures to address atmospheric challenges.</p>
          </div>
        </div>
        
        <div className="flex gap-3">
          <ShieldCheck className="w-5 h-5 text-slate-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="text-slate-200 font-bold uppercase tracking-wider text-[11px] block">Collaborative Integration</span>
            <p className="font-sans text-slate-400">Consumes telemetry feeds from Central Pollution Control Board (CPCB) India, ESA Copernicus Open Access Hub, and NASA LANCE thermal anomaly alerts.</p>
          </div>
        </div>
      </motion.div>

    </motion.div>
  );
}
