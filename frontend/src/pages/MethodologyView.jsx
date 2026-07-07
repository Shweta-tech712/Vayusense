import React from 'react';
import { BookOpen, GitFork, Cpu, Layers } from 'lucide-react';

export default function MethodologyView() {
  return (
    <div className="space-y-8 select-none text-left max-w-4xl">
      
      {/* 1. HEADER */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-widest uppercase">Scientific Methodology</h2>
        <p className="text-xs text-slate-500 font-mono tracking-wider mt-1">
          Deep learning models and multi-satellite datasets fusion pipeline
        </p>
      </div>

      {/* 2. CORE METHODOLOGY STACK */}
      <div className="space-y-6">
        
        {/* Step 1: Multi-satellite retrieval */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 flex items-start space-x-4">
          <div className="p-3 bg-sky-950/40 border border-sky-900/20 rounded-lg text-sky-400 shrink-0">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold uppercase tracking-wider text-slate-200">1. Data Harmonization Layer</h3>
            <p className="text-sm text-slate-300 leading-relaxed mt-2">
              We ingest and spatially align heterogeneous remote sensing products at a unified grid spacing. This includes daily aerosol optical depth (AOD) from MODIS, Formaldehyde (HCHO) column counts from TROPOMI (Sentinel-5P), active fire count thermal anomalies from NASA FIRMS, and boundary layer wind fields from ERA5 reanalysis datasets.
            </p>
          </div>
        </div>

        {/* Step 2: CNN-LSTM prediction */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 flex items-start space-x-4">
          <div className="p-3 bg-indigo-950/40 border border-indigo-900/20 rounded-lg text-indigo-400 shrink-0">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold uppercase tracking-wider text-slate-200">2. Deep Spatial-Temporal Regression</h3>
            <p className="text-sm text-slate-300 leading-relaxed mt-2">
              A hybrid Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM) model captures the spatial grids and temporal dependencies of particulate matter. Conv2D layers extract spatial features from satellite retrievals, which are fed into LSTM networks to model transport lags and predict surface PM2.5 concentrations.
            </p>
          </div>
        </div>

        {/* Step 3: DBSCAN Outliers */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 flex items-start space-x-4">
          <div className="p-3 bg-amber-950/40 border border-amber-900/20 rounded-lg text-amber-400 shrink-0">
            <GitFork className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold uppercase tracking-wider text-slate-200">3. Density-Based Clustering (DBSCAN)</h3>
            <p className="text-sm text-slate-300 leading-relaxed mt-2">
              Formaldehyde (HCHO) hotspots are identified using Density-Based Spatial Clustering of Applications with Noise (DBSCAN). By analyzing TROPOMI retrievals and fire radiative power indices, the algorithm clusters organic pollutant anomalies, filtering out background noise.
            </p>
          </div>
        </div>

      </div>

    </div>
  );
}
