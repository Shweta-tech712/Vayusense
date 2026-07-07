import React, { useMemo } from 'react';
import { useModel } from '../hooks/useModel';
import { Spinner, EmptyState } from '../components/Common/Loader';
import { RefreshCw, AlertTriangle, TrendingUp } from 'lucide-react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  Tooltip, CartesianGrid, ScatterChart, Scatter
} from 'recharts';

// ─── Stat card (pure component — no state) ────────────────────────────────────
function StatCard({ label, value, desc }) {
  return (
    <div className="bg-[#090d16] border border-slate-900 rounded-xl p-5">
      <span className="text-xs font-mono text-slate-400 uppercase block">{label}</span>
      <h4 className="text-2xl font-bold text-white tracking-tight mt-2">{value}</h4>
      <span className="text-xs font-mono text-slate-300 block mt-1">{desc}</span>
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────────
export default function ValidationView() {
  const { metrics, lossCurve, residuals, loading, error, refetch } = useModel();

  // Recharts loss data — memoised so area chart doesn't re-render on non-related state
  const lossData = useMemo(() => {
    if (!lossCurve?.epochs) return [];
    return lossCurve.epochs.map((epoch, idx) => ({
      epoch,
      train: parseFloat((lossCurve.train_loss?.[idx] ?? 0).toFixed(4)),
      val:   parseFloat((lossCurve.val_loss?.[idx]   ?? 0).toFixed(4)),
    }));
  }, [lossCurve]);

  // Residual scatter data — support flat number array or object array from backend
  const residualData = useMemo(() => {
    if (!residuals || residuals.length === 0) return [];
    return residuals.map((res, idx) => ({
      index: idx + 1,
      value: parseFloat(
        typeof res === 'number'
          ? res.toFixed(2)
          : (res.residual ?? res.error ?? 0).toFixed(2)
      ),
      observed:  typeof res === 'object' ? res.observed  : null,
      predicted: typeof res === 'object' ? res.predicted : null,
    }));
  }, [residuals]);

  const STATS = useMemo(() => [
    {
      label: 'R² Correlation',
      value: metrics?.r2 != null ? metrics.r2.toFixed(4) : '—',
      desc:  'Coefficient of Determination',
    },
    {
      label: 'Mean Absolute Error',
      value: metrics?.mae != null ? `${metrics.mae.toFixed(2)} ppb` : '—',
      desc:  'Mean prediction magnitude error',
    },
    {
      label: 'Root Mean Sq. Error',
      value: metrics?.rmse != null ? `${metrics.rmse.toFixed(2)} ppb` : '—',
      desc:  'Standard deviation of residuals',
    },
    {
      label: 'Pearson Coefficient',
      value: metrics?.pearson != null ? metrics.pearson.toFixed(4) : '—',
      desc:  'Linear strength coefficient',
    },
  ], [metrics]);

  if (loading) return <Spinner message="Querying CNN-LSTM Loss Logs and Weights..." />;

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <AlertTriangle className="w-8 h-8 text-amber-500" />
      <p className="text-slate-400 text-sm">{error}</p>
      <button onClick={refetch} className="flex items-center gap-2 px-4 py-2 bg-sky-600/20 border border-sky-500/30 text-sky-400 rounded-lg text-xs hover:bg-sky-600/30 transition-colors">
        <RefreshCw className="w-3.5 h-3.5" /> Retry
      </button>
    </div>
  );

  return (
    <div className="space-y-8 select-none text-left">

      {/* 1. HEADER */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-widest uppercase">Deep Learning Model Performance</h2>
        <p className="text-xs text-slate-500 font-mono tracking-wider mt-1">
          CNN-LSTM regression validation statistics and convergence history
          {metrics?.model_name && ` · ${metrics.model_name}`}
          {metrics?.training_date && ` · Trained ${metrics.training_date}`}
        </p>
      </div>

      {/* 2. STAT CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {STATS.map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </div>

      {/* Extra metrics from backend if present */}
      {(metrics?.mbe != null || metrics?.nrmse != null || metrics?.ioa != null) && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {metrics.mbe   != null && <StatCard label="Mean Bias Error" value={`${metrics.mbe.toFixed(2)} ppb`}  desc="Systematic over/under-prediction" />}
          {metrics.nrmse != null && <StatCard label="NRMSE"          value={`${(metrics.nrmse * 100).toFixed(1)}%`}  desc="Normalised RMSE" />}
          {metrics.ioa   != null && <StatCard label="Index of Agree." value={metrics.ioa.toFixed(4)} desc="Willmott Index of Agreement" />}
          {metrics.train_samples != null && <StatCard label="Train Samples" value={metrics.train_samples.toLocaleString()} desc="Total training observations" />}
          {metrics.test_samples  != null && <StatCard label="Test Samples"  value={metrics.test_samples.toLocaleString()}  desc="Hold-out test set size" />}
        </div>
      )}

      {/* 3. CHARTS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Loss Convergence */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 h-[380px] flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">Training Loss Convergence</h3>
            <p className="text-xs font-mono text-slate-400 mt-1">MSE loss per epoch — train vs validation</p>
          </div>
          {lossData.length === 0 ? (
            <EmptyState icon={TrendingUp} title="No Loss Data" message="Backend did not return loss curve epochs." />
          ) : (
            <div className="flex-1 h-full min-h-[250px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={lossData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="trainGlow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#0ea5e9" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}    />
                    </linearGradient>
                    <linearGradient id="valGlow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}    />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff', fontSize: '11px' }}
                  />
                  <Area type="monotone" name="Train Loss" dataKey="train" stroke="#0ea5e9" strokeWidth={1.8} fillOpacity={1} fill="url(#trainGlow)" />
                  <Area type="monotone" name="Val. Loss"  dataKey="val"   stroke="#ef4444" strokeWidth={1.8} fillOpacity={1} fill="url(#valGlow)"   />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Residual Scatter */}
        <div className="bg-[#090d16] border border-slate-900 rounded-xl p-6 h-[380px] flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">Error Residuals (ŷ − y)</h3>
            <p className="text-xs font-mono text-slate-400 mt-1">
              Distribution of individual prediction errors
              {residualData.length > 0 && ` · ${residualData.length} samples`}
            </p>
          </div>
          {residualData.length === 0 ? (
            <EmptyState icon={TrendingUp} title="No Residual Data" message="Backend returned no residual error samples." />
          ) : (
            <div className="flex-1 h-full min-h-[250px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                  <XAxis type="number" dataKey="index" name="Sample Index" stroke="#64748b" fontSize={11} />
                  <YAxis type="number" dataKey="value" name="Residual Error" stroke="#64748b" fontSize={11} />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.1)' }}
                    contentStyle={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff', fontSize: '11px' }}
                    formatter={(val, name, props) => {
                      const { payload } = props;
                      if (payload.observed != null) {
                        return [`ŷ=${payload.predicted?.toFixed(1)}, y=${payload.observed?.toFixed(1)}`, 'Prediction'];
                      }
                      return [val, name];
                    }}
                  />
                  <Scatter name="Residual Value" data={residualData} fill="#f59e0b" fillOpacity={0.65} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
