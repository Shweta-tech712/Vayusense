import React, { useState, useMemo, useCallback } from 'react';
import { useFilters } from '../context/FilterContext';
import { useAQI } from '../hooks/useAQI';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { Download, Search, Filter, RefreshCw, AlertTriangle, Database } from 'lucide-react';

// AQI badge styling
function AQIBadge({ category }) {
  const clsMap = {
    'Good':        'bg-emerald-950/30 text-emerald-400 border border-emerald-900/40',
    'Satisfactory':'bg-lime-950/30 text-lime-400 border border-lime-900/40',
    'Moderate':    'bg-yellow-950/30 text-yellow-400 border border-yellow-900/40',
    'Poor':        'bg-orange-950/30 text-orange-400 border border-orange-900/40',
    'Very Poor':   'bg-red-950/30 text-red-400 border border-red-900/40',
    'Severe':      'bg-purple-950/30 text-purple-400 border border-purple-900/40',
  };
  return (
    <span className={`inline-block text-xs font-mono font-bold px-2 py-0.5 rounded ${clsMap[category] ?? 'bg-slate-900/30 text-slate-400 border border-slate-800'}`}>
      {category ?? 'Unknown'}
    </span>
  );
}

// Table row — memoised to prevent re-renders when search changes
const StationTableRow = React.memo(function StationTableRow({ row }) {
  return (
    <tr className="hover:bg-slate-950/20 transition-colors">
      <td className="p-4 font-semibold text-slate-200">{row.station}</td>
      <td className="p-4 text-center font-mono text-slate-400 text-xs">
        {row.latitude?.toFixed(3)}, {row.longitude?.toFixed(3)}
      </td>
      <td className="p-4 text-center text-slate-300 font-mono">{row.pm25 ?? '—'}</td>
      <td className="p-4 text-center text-slate-300 font-mono">{row.pm10 ?? '—'}</td>
      <td className="p-4 text-center text-slate-300 font-mono">{row.no2  ?? '—'}</td>
      <td className="p-4 text-center text-slate-300 font-mono">{row.o3   ?? '—'}</td>
      {row.so2   != null && <td className="p-4 text-center text-slate-300 font-mono">{row.so2}</td>}
      {row.co    != null && <td className="p-4 text-center text-slate-300 font-mono">{row.co}</td>}
      {row.temp  != null && <td className="p-4 text-center text-slate-300 font-mono">{row.temp} °C</td>}
      <td className="p-4 text-center font-bold text-sky-400 font-mono">{row.cpcb_aqi ?? '—'}</td>
      <td className="p-4 text-right"><AQIBadge category={row.aqi_category} /></td>
    </tr>
  );
});

export default function DatasetExplorerView() {
  const { analysisDate, selectedState, selectedPollutant, searchQuery } = useFilters();
  const { stations, loading, error, refetch } = useAQI(analysisDate, selectedState, selectedPollutant);

  const [localSearch, setLocalSearch] = useState('');
  const [sortKey,     setSortKey]     = useState('cpcb_aqi');
  const [sortDesc,    setSortDesc]    = useState(true);

  const handleSort = useCallback((key) => {
    setSortKey(prev => {
      if (prev === key) setSortDesc(d => !d);
      else setSortDesc(true);
      return key;
    });
  }, []);

  // Filtered + sorted rows — memoised to avoid re-sorting on every keystroke
  const filteredData = useMemo(() => {
    const q = (searchQuery || localSearch).toLowerCase();
    let rows = q
      ? stations.filter(s => s.station.toLowerCase().includes(q))
      : [...stations];

    rows.sort((a, b) => {
      const av = a[sortKey] ?? 0;
      const bv = b[sortKey] ?? 0;
      return sortDesc ? (bv > av ? 1 : -1) : (av > bv ? 1 : -1);
    });
    return rows;
  }, [stations, localSearch, searchQuery, sortKey, sortDesc]);

  // Extra columns present in the backend response
  const hasSO2  = stations.some(s => s.so2  != null);
  const hasCO   = stations.some(s => s.co   != null);
  const hasTemp = stations.some(s => s.temp != null);

  // CSV export
  const handleDownloadCSV = useCallback(() => {
    if (filteredData.length === 0) return;
    const headers = ['Station','Lat','Lon','PM2.5','PM10','NO2','O3',
      hasSO2 ? 'SO2' : null, hasCO ? 'CO' : null, hasTemp ? 'Temp_C' : null,
      'CPCB_AQI','AQI_Category'
    ].filter(Boolean).join(',');

    const rows = filteredData.map(r => [
      `"${r.station.replace(/"/g, '')}"`,
      r.latitude?.toFixed(4), r.longitude?.toFixed(4),
      r.pm25 ?? '', r.pm10 ?? '', r.no2 ?? '', r.o3 ?? '',
      hasSO2  ? (r.so2  ?? '') : null,
      hasCO   ? (r.co   ?? '') : null,
      hasTemp ? (r.temp ?? '') : null,
      r.cpcb_aqi ?? '', `"${r.aqi_category ?? ''}"`
    ].filter(v => v !== null).join(','));

    const csv = [headers, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `isro_aqi_${selectedState.replace(/\s+/g,'_')}_${analysisDate}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [filteredData, analysisDate, selectedState, hasSO2, hasCO, hasTemp]);

  if (loading) return <Spinner message="Indexing Geospatial Observation Databases..." />;

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <AlertTriangle className="w-8 h-8 text-amber-500" />
      <p className="text-slate-400 text-sm">{error}</p>
      <button onClick={refetch} className="flex items-center gap-2 px-4 py-2 bg-sky-600/20 border border-sky-500/30 text-sky-400 rounded-lg text-xs hover:bg-sky-600/30 transition-colors">
        <RefreshCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );

  if (stations.length === 0) return (
    <EmptyState
      icon={Database}
      title="No Station Records"
      message={`No CPCB observations found for "${selectedState}" on ${analysisDate}.`}
      action={{ label: 'Retry', onClick: refetch }}
    />
  );

  return (
    <div className="space-y-6 select-none text-left">

      {/* Header & Export */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-widest uppercase">Dataset Explorer</h2>
          <p className="text-xs text-slate-500 font-mono tracking-wider mt-1">
            Live backend observation matrices · {selectedState} · {analysisDate} · {selectedPollutant}
          </p>
        </div>
        <button
          onClick={handleDownloadCSV}
          className="px-5 py-2.5 bg-sky-600 hover:bg-sky-500 border border-sky-400/20 rounded-lg text-xs font-semibold uppercase tracking-wider text-white transition-all duration-300 flex items-center space-x-2"
        >
          <Download className="w-4 h-4" />
          <span>Export CSV ({filteredData.length} rows)</span>
        </button>
      </div>

      {/* Filter Bar */}
      <div className="bg-[#090d16] border border-slate-900 rounded-xl p-4 flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="relative w-full sm:w-80">
          <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-slate-500" />
          </span>
          <input
            type="text"
            placeholder="Search stations..."
            value={localSearch}
            onChange={(e) => setLocalSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-[#050811] border border-slate-850 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-sky-500/50 transition-colors"
          />
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono text-slate-400 shrink-0">
          <Filter className="w-4 h-4 text-slate-500" />
          <span>{filteredData.length} / {stations.length} records · sorted by {sortKey} {sortDesc ? '↓' : '↑'}</span>
        </div>
      </div>

      {/* Data Table */}
      <div className="bg-[#090d16] border border-slate-900 rounded-xl overflow-hidden">
        <div className="overflow-x-auto max-h-[500px]">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-950/60 border-b border-slate-900 text-xs font-mono text-slate-350 uppercase tracking-wider sticky top-0 z-10">
                <th className="p-4 font-semibold">Station Name</th>
                <th className="p-4 font-semibold text-center">Lat / Lon</th>
                <th className="p-4 font-semibold text-center cursor-pointer hover:text-white" onClick={() => handleSort('pm25')}>PM2.5</th>
                <th className="p-4 font-semibold text-center cursor-pointer hover:text-white" onClick={() => handleSort('pm10')}>PM10</th>
                <th className="p-4 font-semibold text-center cursor-pointer hover:text-white" onClick={() => handleSort('no2')}>NO₂</th>
                <th className="p-4 font-semibold text-center cursor-pointer hover:text-white" onClick={() => handleSort('o3')}>O₃</th>
                {hasSO2  && <th className="p-4 font-semibold text-center cursor-pointer hover:text-white" onClick={() => handleSort('so2')}>SO₂</th>}
                {hasCO   && <th className="p-4 font-semibold text-center cursor-pointer hover:text-white" onClick={() => handleSort('co')}>CO</th>}
                {hasTemp && <th className="p-4 font-semibold text-center">Temp</th>}
                <th className="p-4 font-semibold text-center cursor-pointer hover:text-white" onClick={() => handleSort('cpcb_aqi')}>CPCB AQI</th>
                <th className="p-4 font-semibold text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-900/60 text-xs">
              {filteredData.length === 0 ? (
                <tr>
                  <td colSpan="11" className="p-8 text-center text-slate-500 italic font-mono">
                    No records match "{localSearch}".
                  </td>
                </tr>
              ) : (
                filteredData.map((row) => (
                  <StationTableRow key={row.station} row={row} />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
