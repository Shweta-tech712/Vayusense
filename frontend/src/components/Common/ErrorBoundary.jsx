import React, { Component } from 'react';
import { AlertCircle } from 'lucide-react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an exception: ", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 bg-red-950/20 border border-red-900/30 rounded-xl max-w-xl mx-auto my-8 select-none">
          <AlertCircle className="w-12 h-12 text-red-500 mb-4 animate-bounce" />
          <h3 className="text-lg font-bold text-white mb-2">Component Execution Fault</h3>
          <p className="text-xs text-slate-400 text-center mb-6 leading-relaxed">
            The system encountered a rendering error while loading this panel. This is often caused by missing geospatial bounds or connection issues.
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-6 py-2.5 bg-red-900 hover:bg-red-800 text-xs font-bold uppercase tracking-wider text-white rounded-lg transition-colors duration-300"
          >
            Attempt Reload
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
