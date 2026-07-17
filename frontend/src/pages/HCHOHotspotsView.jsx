import React, { useMemo, memo } from 'react';
import { useFilters } from '../context/FilterContext';
import { useHCHO } from '../hooks/useHCHO';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { useMap, MapContainer, TileLayer, Polygon, Popup, CircleMarker } from 'react-leaflet';
import { RefreshCw, AlertTriangle, Eye } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

const STATE_COORDS = {
  'All India': { center: [22.0, 79.0], zoom: 5 },
  'Delhi': { center: [28.6139, 77.2090], zoom: 9 },
  'Maharashtra': { center: [19.7515, 75.7139], zoom: 6 },
  'Karnataka': { center: [15.3173, 75.7139], zoom: 6 },
  'West Bengal': { center: [22.9868, 87.8550], zoom: 7 },
  'Tamil Nadu': { center: [11.1271, 78.6569], zoom: 6 },
  'Telangana': { center: [18.1124, 79.0193], zoom: 6 },
  'Bihar': { center: [25.0961, 85.3131], zoom: 7 }
};

// Component to dynamically update map center
function MapUpdater({ center, zoom }) {
  const map = useMap();
  React.useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

// ─── Memoised grid point marker ───────────────────────────────────────────────
const GridMarker = memo(function GridMarker({ pt, idx }) {
  const val = pt.hcho_vcd ?? pt.density ?? 0;
  return (
    <CircleMarker
      key={pt.id ?? `grd-${idx}`}
      center={[pt.latitude, pt.longitude]}
      radius={12}
      fillColor="#7c3aed"
      color="transparent"
      weight={0}
      fillOpacity={Math.min(0.22 + val / 10, 0.85)}
    >
      <Popup className="leaflet-popup-dark">
        <div className="text-xs font-sans">
          <strong>Tropospheric HCHO Column</strong>
          <div className="mt-1 text-indigo-400 font-mono">{val.toFixed(3)} 10⁻⁴ mol/m²</div>
          {pt.quality_flag != null && (
            <div className="mt-0.5 text-[10px] text-slate-500">Quality flag: {pt.quality_flag}</div>
          )}
        </div>
      </Popup>
    </CircleMarker>
  );
});

// ─── Memoised DBSCAN cluster polygon ──────────────────────────────────────────
const ClusterPolygon = memo(function ClusterPolygon({ cluster }) {
  return (
    <Polygon
      key={cluster.cluster_id}
      positions={cluster.coordinates}
      pathOptions={{ fillColor: '#7c3aed', fillOpacity: 0.15, color: '#a855f7', weight: 1.5, dashArray: '4,4' }}
    >
      <Popup className="leaflet-popup-dark">
        <div className="text-xs font-sans">
          <h4 className="font-bold text-slate-800 border-b pb-1 mb-1">
            {cluster.label ?? `DBSCAN Cluster #${cluster.cluster_id}`}
          </h4>
          <div className="mt-1.5 space-y-1">
            <div>Core Points: <strong>{cluster.point_count}</strong></div>
            <div>Biomass Fire Links: <strong>{cluster.fire_count} sources</strong></div>
            <div>Cumulative FRP: <strong>{Math.round(cluster.cumulative_frp)} MW</strong></div>
            {cluster.mean_hcho != null && (
              <div>Mean HCHO VCD: <strong className="text-indigo-400">{cluster.mean_hcho.toFixed(2)} ×10⁻⁴ mol/m²</strong></div>
            )}
          </div>
        </div>
      </Popup>
    </Polygon>
  );
});

// ─── Main Component ────────────────────────────────────────────────────────────
export default function HCHOHotspotsView() {
  const { analysisDate, hchoThreshold, setHchoThreshold, selectedState, searchQuery } = useFilters();
  const { hotspots, grid, seasonal, loading, error, refetch } = useHCHO(analysisDate, hchoThreshold, selectedState);

  // Client-side threshold and search query filtering for instant slider/search response
  const filteredDensity  = useMemo(() => {
    let res = grid.filter(pt => (pt.hcho_vcd ?? pt.density ?? 0) >= hchoThreshold);
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      res = res.filter(pt => (pt.id || '').toLowerCase().includes(q));
    }
    return res;
  }, [grid, hchoThreshold, searchQuery]);

  const filteredClusters = useMemo(() => {
    let res = hotspots.filter(c => (c.mean_hcho ?? c.cumulative_frp / 150) >= hchoThreshold);
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      res = res.filter(c => 
        (c.label || '').toLowerCase().includes(q) || 
        String(c.cluster_id).includes(q)
      );
    }
    return res;
  }, [hotspots, hchoThreshold, searchQuery]);

  if (loading) return <Spinner message="Querying Sentinel-5P HCHO Column Feeds..." />;

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <AlertTriangle className="w-8 h-8 text-amber-500" />
      <p className="text-slate-400 text-sm">{error}</p>
      <button onClick={refetch} className="flex items-center gap-2 px-4 py-2 bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 rounded-lg text-xs hover:bg-indigo-600/30 transition-colors">
        <RefreshCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-12rem)] select-none text-left">

      {/* 1. CONTROL PANEL */}
      <div className="w-full lg:w-88 bg-[#090d16] border border-slate-900 rounded-xl p-5 flex flex-col justify-between shrink-0 overflow-y-auto">
        <div className="space-y-6">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">Formaldehyde Clusters</h3>
            <p className="text-xs text-slate-400 font-mono tracking-wide mt-1">
              DBSCAN spatial outlier detection from Sentinel-5P TROPOMI VCD retrievals
            </p>
          </div>

          {/* Threshold slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-400">Min HCHO Column</span>
              <span className="font-mono font-bold text-sky-400">{hchoThreshold.toFixed(1)} ×10⁻⁴ mol/m²</span>
            </div>
            <input
              type="range" min="1.0" max="5.0" step="0.1"
              value={hchoThreshold}
              onChange={(e) => setHchoThreshold(parseFloat(e.target.value))}
              className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
            <span className="text-xs font-mono text-slate-400 block">
              Threshold applied client-side for instant response
            </span>
          </div>

          {/* Live stats */}
          <div className="p-3 bg-slate-950/40 border border-slate-900 rounded-lg grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-xs font-mono text-slate-400 uppercase block">Clusters Detected</span>
              <span className="text-xl font-bold text-indigo-400">{filteredClusters.length}</span>
            </div>
            <div>
              <span className="text-xs font-mono text-slate-400 uppercase block">Grid Points</span>
              <span className="text-xl font-bold text-white">{filteredDensity.length}</span>
            </div>
          </div>

          {/* Cluster list */}
          <div className="border-t border-slate-900/60 pt-4">
            <h4 className="text-xs font-mono tracking-widest text-slate-400 uppercase mb-2">Active Clusters</h4>
            {filteredClusters.length === 0 ? (
              <EmptyState
                icon={Eye}
                title="No Clusters"
                message="Raise the HCHO threshold or select a different date."
              />
            ) : (
              <div className="space-y-2 max-h-[140px] overflow-y-auto pr-1">
                {filteredClusters.map((cluster) => (
                  <div key={cluster.cluster_id} className="p-3 bg-slate-950/40 border border-slate-900 rounded-lg">
                    <div className="flex justify-between items-center gap-2 text-xs font-bold text-slate-300">
                      <span className="truncate flex-1 min-w-0" title={cluster.label ?? `Cluster #${cluster.cluster_id}`}>{cluster.label ?? `Cluster #${cluster.cluster_id}`}</span>
                      <span className="text-xs font-mono text-indigo-400 shrink-0">Active</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-slate-900/40 text-xs font-mono text-slate-400">
                      <div>Points: <span className="text-slate-300">{cluster.point_count}</span></div>
                      <div>FRP: <span className="text-slate-300">{Math.round(cluster.cumulative_frp)} MW</span></div>
                      {cluster.mean_hcho != null && (
                        <div className="col-span-2">Mean VCD: <span className="text-indigo-300">{cluster.mean_hcho.toFixed(2)}</span></div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-slate-900/60 pt-4 text-xs text-slate-400 leading-relaxed font-mono">
          Formaldehyde (HCHO) is a tracer for VOC emissions. DBSCAN isolates spatial anomalies in Sentinel-5P satellite retrievals.
        </div>
      </div>

      {/* 2. MAP */}
      <div className="flex-1 bg-[#090d16] border border-slate-900 rounded-xl overflow-hidden relative z-0 min-h-[350px]">
        <MapContainer center={STATE_COORDS[selectedState]?.center || [20.5937, 78.9629]} zoom={STATE_COORDS[selectedState]?.zoom || 5} className="w-full h-full" style={{ background: '#050811' }}>
          <MapUpdater center={STATE_COORDS[selectedState]?.center || [20.5937, 78.9629]} zoom={STATE_COORDS[selectedState]?.zoom || 5} />
          <TileLayer
            attribution="&copy; Contributors"
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          {filteredDensity.map((pt, idx) => (
            <GridMarker key={pt.id ?? idx} pt={pt} idx={idx} />
          ))}
          {filteredClusters.map((cluster) => (
            <ClusterPolygon key={cluster.cluster_id} cluster={cluster} />
          ))}
        </MapContainer>
      </div>

    </div>
  );
}
