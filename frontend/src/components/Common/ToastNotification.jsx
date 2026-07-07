import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, XCircle, Info, X } from 'lucide-react';

const ICONS = {
  success: { icon: CheckCircle, color: '#10b981', bg: 'rgba(16,185,129,0.08)',  border: 'rgba(16,185,129,0.25)'  },
  warning: { icon: AlertTriangle, color: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.25)' },
  error:   { icon: XCircle,      color: '#ef4444', bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.25)'   },
  info:    { icon: Info,          color: '#38bdf8', bg: 'rgba(56,189,248,0.08)', border: 'rgba(56,189,248,0.25)'  },
};

let toastIdCounter = 0;

/**
 * ToastNotification — mounts in Layout and listens for the global
 * CustomEvent("show-toast") dispatched by axiosInstance error interceptors
 * and any component that calls window.dispatchEvent(toastEvent).
 *
 * Usage from anywhere in the app:
 *   window.dispatchEvent(new CustomEvent('show-toast', {
 *     detail: { message: 'Something went wrong', type: 'error', duration: 4000 }
 *   }));
 */
export default function ToastNotification() {
  const [toasts, setToasts] = useState([]);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback(({ message, type = 'info', duration = 4500 }) => {
    const id = ++toastIdCounter;
    setToasts((prev) => [...prev.slice(-4), { id, message, type }]); // Max 5 at once
    setTimeout(() => removeToast(id), duration);
  }, [removeToast]);

  // Listen to global custom events from axiosInstance or any component
  useEffect(() => {
    const handler = (e) => addToast(e.detail || {});
    window.addEventListener('show-toast', handler);
    return () => window.removeEventListener('show-toast', handler);
  }, [addToast]);

  return (
    <div
      aria-live="polite"
      className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-3 items-end pointer-events-none"
    >
      <AnimatePresence>
        {toasts.map((toast) => {
          const cfg = ICONS[toast.type] || ICONS.info;
          const IconComp = cfg.icon;
          return (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, x: 80, scale: 0.9 }}
              animate={{ opacity: 1, x: 0,  scale: 1   }}
              exit={{    opacity: 0, x: 80,  scale: 0.9 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="pointer-events-auto flex items-start gap-3 max-w-xs rounded-xl px-4 py-3 shadow-2xl"
              style={{
                backgroundColor: cfg.bg,
                border: `1px solid ${cfg.border}`,
                backdropFilter: 'blur(12px)',
              }}
            >
              <IconComp
                className="shrink-0 mt-0.5"
                style={{ color: cfg.color, width: 16, height: 16 }}
              />
              <p className="text-xs font-medium text-slate-200 leading-relaxed flex-1">
                {toast.message}
              </p>
              <button
                onClick={() => removeToast(toast.id)}
                className="shrink-0 text-slate-500 hover:text-slate-200 transition-colors mt-0.5"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

/**
 * Programmatic helper — call from any component without needing the event API.
 * import { showToast } from '../components/Common/ToastNotification';
 * showToast({ message: 'Saved!', type: 'success' });
 */
export function showToast({ message, type = 'info', duration = 4500 }) {
  window.dispatchEvent(new CustomEvent('show-toast', { detail: { message, type, duration } }));
}
