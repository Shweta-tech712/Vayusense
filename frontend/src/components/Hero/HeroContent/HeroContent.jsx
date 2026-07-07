import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

export default function HeroContent({ isTabActive }) {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.12
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 25 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] }
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate={isTabActive ? "visible" : "hidden"}
      className="flex flex-col space-y-8 text-left max-w-2xl"
    >
      
      {/* Target Mission tag */}
      <motion.div variants={itemVariants} className="inline-flex items-center self-start">
        <span className="text-xs font-mono tracking-wider text-amber-400 border border-amber-500/35 bg-amber-500/5 px-3 py-1 rounded-full uppercase">
          AI & Satellite Remote Sensing Initiative
        </span>
      </motion.div>

      {/* Primary Project Title */}
      <motion.h1 
        variants={itemVariants} 
        className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.08] select-none"
      >
        Development of <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-400 via-blue-500 to-indigo-500">Surface AQI</span> & Identification of HCHO Hotspots over India using Satellite Data
      </motion.h1>

      {/* Comprehensive Subtitle */}
      <motion.p 
        variants={itemVariants} 
        className="text-slate-200 text-base md:text-[17px] leading-relaxed font-normal"
      >
        An AI-powered satellite monitoring platform that predicts Surface Air Quality Index, detects Formaldehyde hotspots, identifies biomass burning regions, and analyzes pollution transport using INSAT-3D, Sentinel-5P, NASA FIRMS, ERA5, and Deep Learning.
      </motion.p>

      {/* Button CTAs Group */}
      <motion.div 
        variants={itemVariants}
        className="flex flex-wrap gap-4 pt-2"
      >
        <button 
          onClick={() => navigate('/dashboard')}
          className="px-7 py-4 bg-sky-600 hover:bg-sky-500 border border-sky-400/20 rounded-xl font-semibold text-xs uppercase tracking-wider text-white transition-all duration-300 flex items-center space-x-2 group shadow-[0_4px_25px_rgba(14,165,233,0.15)] hover:shadow-[0_4px_35px_rgba(14,165,233,0.35)]"
        >
          <span>Explore Dashboard</span>
          <ArrowRight className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-1" />
        </button>
        
        <button 
          onClick={() => navigate('/methodology')}
          className="px-7 py-4 bg-transparent hover:bg-slate-900/60 border border-slate-800 hover:border-slate-700 rounded-xl font-semibold text-xs uppercase tracking-wider text-slate-300 hover:text-white transition-all duration-300"
        >
          View Methodology
        </button>
      </motion.div>

    </motion.div>
  );
}
