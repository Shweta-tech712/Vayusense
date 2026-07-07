import React from 'react';

// Animated Spinner Loop
export function Spinner({ message = "Retrieving Satellite Telemetry..." }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 select-none w-full min-h-[300px]">
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 rounded-full border-4 border-slate-900" />
        <div className="absolute inset-0 rounded-full border-4 border-sky-500 border-t-transparent animate-spin" />
      </div>
      <p className="mt-4 text-xs font-mono tracking-widest text-slate-400 uppercase">{message}</p>
    </div>
  );
}

// Card skeleton panel representing loading dashboards
export function SkeletonCard() {
  return (
    <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 w-full animate-pulse flex flex-col justify-between min-h-[160px]">
      <div className="flex justify-between items-center mb-4">
        <div className="h-2.5 w-24 bg-slate-800 rounded" />
        <div className="h-8 w-8 bg-slate-800 rounded-lg" />
      </div>
      <div className="space-y-3">
        <div className="h-5 w-36 bg-slate-800 rounded" />
        <div className="h-3 w-full bg-slate-800 rounded" />
      </div>
    </div>
  );
}

export function SkeletonGrid({ count = 6 }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 w-full">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}

/**
 * EmptyState — shown when a backend API returns an empty array.
 * icon: a Lucide React component class
 * title: short heading
 * message: descriptive sentence
 * action: optional { label, onClick } object for a retry / reset button
 */
export function EmptyState({ icon: Icon, title = 'No Data Found', message = 'The backend returned no records for the selected filters.', action }) {
  return (
    <div className="flex flex-col items-center justify-center p-16 select-none w-full min-h-[300px] text-center">
      {Icon && (
        <div className="w-14 h-14 rounded-2xl bg-slate-900/60 border border-slate-800 flex items-center justify-center mb-5">
          <Icon className="w-7 h-7 text-slate-500" />
        </div>
      )}
      <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider">{title}</h3>
      <p className="text-xs font-mono text-slate-500 mt-2 max-w-xs leading-relaxed">{message}</p>
      {action && (
        <button
          onClick={action.onClick}
          className="mt-5 flex items-center gap-2 px-4 py-2 bg-sky-600/20 border border-sky-500/30 text-sky-400 rounded-lg text-xs font-semibold hover:bg-sky-600/30 transition-colors"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
