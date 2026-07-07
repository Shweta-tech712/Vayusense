import React, { useState, useEffect, useRef, memo } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

// Import sub-components
import Navbar from './Navbar/Navbar';
import BackgroundVideo from './BackgroundVideo/BackgroundVideo';
import HeroContent from './HeroContent/HeroContent';
import FloatingStats from './FloatingStats/FloatingStats';
import ScrollIndicator from './ScrollIndicator/ScrollIndicator';

// ----------------------------------------------------------------------
// GRAPHICS OVERLAY CONTAINER (Stars, Grid, Earth Glow & Orbit Lines)
// ----------------------------------------------------------------------
const BackgroundGraphics = memo(() => {
  return (
    <div className="absolute inset-0 z-10 pointer-events-none overflow-hidden select-none">
      
      {/* Dynamic Keyframes Sheet */}
      <style>{`
        @keyframes drift {
          0% { transform: translate3d(0, 0, 0) rotate(0deg); }
          50% { transform: translate3d(20px, -20px, 0) rotate(180deg); }
          100% { transform: translate3d(0, 0, 0) rotate(360deg); }
        }
        @keyframes pulseGlow {
          0%, 100% { opacity: 0.22; transform: scale(1) translate3d(0,0,0); }
          50% { opacity: 0.38; transform: scale(1.08) translate3d(8px, -4px, 0); }
        }
        @keyframes lightSweep {
          0%, 100% { transform: rotate(-35deg) translate3d(-3%, -3%, 0); opacity: 0.12; }
          50% { transform: rotate(-31deg) translate3d(3%, 3%, 0); opacity: 0.22; }
        }
        @keyframes orbitFlow {
          to { stroke-dashoffset: -1000; }
        }
        .bg-star {
          will-change: transform;
          animation: drift 25s infinite linear;
        }
        .sky-light-beam {
          will-change: transform, opacity;
          animation: lightSweep 18s infinite ease-in-out;
        }
        .horizon-earth-glow {
          will-change: transform, opacity;
          animation: pulseGlow 12s infinite ease-in-out;
        }
        .satellite-track {
          stroke-dasharray: 12, 12;
          animation: orbitFlow 40s linear infinite;
        }
      `}</style>

      {/* Earth Horizon Glow */}
      <div 
        className="horizon-earth-glow absolute -bottom-[350px] left-1/2 -translate-x-1/2 w-[900px] h-[500px] rounded-full blur-[150px]"
        style={{
          background: 'radial-gradient(circle, rgba(14, 165, 233, 0.18) 0%, rgba(3, 105, 161, 0.03) 70%, transparent 100%)'
        }}
      />

      {/* Sweeping searchlight rays */}
      <div 
        className="sky-light-beam absolute -top-[20%] -left-[10%] w-[60%] h-[150%] blur-[120px] origin-top"
        style={{
          background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.05) 0%, transparent 70%)'
        }}
      />

      {/* Slowly translating stars */}
      <div className="bg-star absolute top-[15%] left-[20%] w-[3px] h-[3px] bg-sky-400 rounded-full blur-[1px]" />
      <div className="bg-star absolute top-[45%] left-[75%] w-[2px] h-[2px] bg-white rounded-full opacity-60" style={{ animationDelay: '-5s', animationDuration: '30s' }} />
      <div className="bg-star absolute top-[70%] left-[15%] w-[4px] h-[4px] bg-blue-400 rounded-full blur-[1px]" style={{ animationDelay: '-12s', animationDuration: '22s' }} />
      <div className="bg-star absolute top-[25%] left-[80%] w-[3px] h-[3px] bg-sky-300 rounded-full blur-[0.5px]" style={{ animationDelay: '-8s', animationDuration: '28s' }} />
      <div className="bg-star absolute top-[80%] left-[60%] w-[2px] h-[2px] bg-white rounded-full opacity-85" style={{ animationDelay: '-18s', animationDuration: '35s' }} />

    </div>
  );
});

BackgroundGraphics.displayName = 'BackgroundGraphics';


// ----------------------------------------------------------------------
// 5. MAIN INTEGRATED HERO PAGE
// ----------------------------------------------------------------------
export default function Hero() {
  const [isTabActive, setIsTabActive] = useState(true);
  const containerRef = useRef(null);
  
  // Track page scroll position for scroll-linked fades and shifts
  const { scrollY } = useScroll();
  const heroOpacity = useTransform(scrollY, [0, 420], [1, 0]);
  const heroTranslateY = useTransform(scrollY, [0, 420], [0, -70]);
  const bgBlur = useTransform(scrollY, [0, 420], [0, 16]);

  // Hook into Browser tab active lifecycle
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsTabActive(!document.hidden);
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // Performance-optimized cursor parallax tracking
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!containerRef.current) return;
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      
      const nx = (clientX / innerWidth) - 0.5;
      const ny = (clientY / innerHeight) - 0.5;
      
      containerRef.current.style.setProperty('--mx', nx.toFixed(3));
      containerRef.current.style.setProperty('--my', ny.toFixed(3));
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div 
      ref={containerRef}
      className="relative min-h-screen w-full bg-black text-white font-sans overflow-hidden flex flex-col justify-between select-none"
      style={{
        '--mx': 0,
        '--my': 0
      }}
    >
      {/* Layer 1: Fullscreen Video Background */}
      <BackgroundVideo />

      {/* Layer 2: Black + Space Blue Gradient Mask (Slightly lighter overlay for video visibility) */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/60 to-[#070b16]/40 z-10 pointer-events-none" />

      {/* Layer 3: Mouse-move Parallax Stars & Grid overlay */}
      <div 
        className="absolute inset-0 z-20 pointer-events-none transition-transform duration-300 ease-out"
        style={{
          transform: 'translate3d(calc(var(--mx) * 15px), calc(var(--my) * 15px), 0)'
        }}
      >
        <BackgroundGraphics />
      </div>

      {/* Layer 4: Sticky Blur Navbar */}
      <Navbar />

      {/* Layer 5: Hero Content & Floating Cards container */}
      <motion.div
        style={{ opacity: heroOpacity, y: heroTranslateY, filter: `blur(${bgBlur}px)` }}
        className="relative z-30 w-full min-h-screen max-w-7xl mx-auto px-6 md:px-12 pt-32 pb-10 flex flex-col justify-between"
      >
        
        {/* Main Columns Content block */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center my-auto w-full">
          
          {/* Left Column: Mission Description and Copywriting */}
          <div className="lg:col-span-9 lg:col-start-1">
            <HeroContent isTabActive={isTabActive} />
          </div>

        </div>

        {/* Bouncing scroll cue at base of viewport */}
        <ScrollIndicator isTabActive={isTabActive} />

      </motion.div>
      
    </div>
  );
}
