import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { FilterProvider } from './context/FilterContext';
import ErrorBoundary from './components/Common/ErrorBoundary';
import ScrollToTop from './components/Common/ScrollToTop';
import { Spinner } from './components/Common/Loader';

// Eager load layout coordinates
import Layout from './components/Dashboard/Layout';

// Lazy-load all nested router components to optimize load times
const Hero = lazy(() => import('./components/Hero/Hero'));
const HomeView = lazy(() => import('./pages/HomeView'));
const AQIMapView = lazy(() => import('./pages/AQIMapView'));
const HCHOHotspotsView = lazy(() => import('./pages/HCHOHotspotsView'));
const FireAnalysisView = lazy(() => import('./pages/FireAnalysisView'));
const TransportView = lazy(() => import('./pages/TransportView'));
const AnalyticsView = lazy(() => import('./pages/AnalyticsView'));
const ValidationView = lazy(() => import('./pages/ValidationView'));
const DatasetExplorerView = lazy(() => import('./pages/DatasetExplorerView'));
const MethodologyView = lazy(() => import('./pages/MethodologyView'));
const AboutView = lazy(() => import('./pages/AboutView'));
const SettingsView = lazy(() => import('./pages/SettingsView'));
const NotFound = lazy(() => import('./pages/NotFound'));

function App() {
  return (
    <FilterProvider>
      <Router>
        <ScrollToTop />
        <Suspense fallback={<Spinner message="Initializing Satellite Telemetry..." />}>
          <Routes>
            {/* Landing hero page */}
            <Route path="/" element={<Hero />} />

            {/* Shared dashboard layout coordinates (Pathless parent route) */}
            <Route element={
              <ErrorBoundary>
                <Layout />
              </ErrorBoundary>
            }>
              <Route path="/dashboard" element={<HomeView />} />
              <Route path="/aqi" element={<AQIMapView />} />
              <Route path="/hcho" element={<HCHOHotspotsView />} />
              <Route path="/fire-analysis" element={<FireAnalysisView />} />
              <Route path="/transport-analysis" element={<TransportView />} />
              <Route path="/analytics" element={<AnalyticsView />} />
              <Route path="/model-performance" element={<ValidationView />} />
              <Route path="/dataset-explorer" element={<DatasetExplorerView />} />
              <Route path="/methodology" element={<MethodologyView />} />
              <Route path="/about" element={<AboutView />} />
              <Route path="/settings" element={<SettingsView />} />
            </Route>

            {/* 404 Fallback page */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </Router>
    </FilterProvider>
  );
}

export default App;
