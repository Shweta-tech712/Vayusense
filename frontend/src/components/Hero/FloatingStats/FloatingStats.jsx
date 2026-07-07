import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Globe, Radio, Flame, TestTube, HardDrive, Wind } from 'lucide-react';

const STATS_DATA = [
  {
    icon: Globe,
    title: 'Surface AQI',
    value: '0.84 R²',
    desc: 'Validation correlation index mapped across 500+ CPCB stations.',
    floatDuration: 6.2,
    path: '/aqi'
  },
  {
    icon: Radio,
    title: 'Sentinel-5P',
    value: 'TROPOMI Feed',
    desc: 'Daily retrieval layers for NO2 and formaldehyde column counts.',
    floatDuration: 7.8,
    path: '/dataset-explorer'
  },
  {
    icon: Flame,
    title: 'Active Fire Events',
    value: 'NASA FIRMS',
    desc: 'Active fire hotspot buffers joined in crop residue burning belts.',
    floatDuration: 6.9,
    path: '/fire-analysis'
  },
  {
    icon: TestTube,
    title: 'HCHO Hotspots',
    value: 'DBSCAN Clusters',
    desc: 'Statistical outlier groupings and organic volatile anomalies.',
    floatDuration: 8.4,
    path: '/hcho'
  },
  {
    icon: HardDrive,
    title: 'Satellite Coverage',
    value: 'MODIS & INSAT-3D',
    desc: 'Spatially aligned AOD retrieval profiles at 1km grid increments.',
    floatDuration: 7.2,
    path: '/dataset-explorer'
  },
  {
    icon: Wind,
    title: 'Wind Analysis',
    value: 'ERA5 Advection',
    desc: 'Boundary layer wind transport vectors tracing plume advections.',
    floatDuration: 8.0,
    path: '/transport-analysis'
  }
];

export default function FloatingStats({ isTabActive }) {
  const navigate = useNavigate();

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-xl select-none">
      {STATS_DATA.map((stat, idx) => {
        const IconComponent = stat.icon;
        
        return (
          <motion.div
            key={stat.title}
            // Float loop on the Y-axis
            animate={isTabActive ? {
              y: [0, -12, 0],
            } : {}}
            transition={{
              duration: stat.floatDuration,
              repeat: Infinity,
              ease: "easeInOut",
              delay: idx * 0.1
            }}
            whileHover={{ 
              y: -4, 
              borderColor: 'rgba(245, 158, 11, 0.45)', // Golden border on hover
              boxShadow: '0 0 25px rgba(245, 158, 11, 0.1)'
            }}
            onClick={() => navigate(stat.path)}
            className="bg-[#090d16] border border-slate-900 rounded-xl p-5 text-left flex flex-col justify-between transition-all duration-300 relative overflow-hidden group cursor-pointer shadow-lg"
            style={{
              willChange: 'transform',
              boxShadow: '0 6px 35px rgba(0,0,0,0.6)'
            }}
          >
            {/* Top Indicator border (Golden highlight) */}
            <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-amber-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono tracking-wide text-slate-300 uppercase">{stat.title}</span>
              <div className="p-2 rounded-lg bg-sky-950/30 border border-sky-900/20 group-hover:border-amber-500/25 transition-all duration-300">
                <IconComponent className="w-4 h-4 text-sky-400 group-hover:text-amber-500 transition-colors duration-300" />
              </div>
            </div>
            
            <div>
              <h4 className="text-xl font-bold text-white tracking-tight mb-1 group-hover:text-sky-400 transition-colors duration-300">
                {stat.value}
              </h4>
              <p className="text-slate-300 text-[13px] leading-relaxed">
                {stat.desc}
              </p>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
