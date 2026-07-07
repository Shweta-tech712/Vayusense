import React, { useEffect, useRef, useState, memo } from 'react';

const VIDEO_URL = "https://www.image2url.com/r2/default/videos/1782802354454-ce248e2a-5fd1-4fa5-a702-b6a5de0db0cb.mp4";

const BackgroundVideo = memo(() => {
  const containerRef = useRef(null);
  const videoRef = useRef(null);
  const [isIntersecting, setIsIntersecting] = useState(true);
  const [isTabActive, setIsTabActive] = useState(true);

  // 1. Monitor Browser Tab Focus (Page Visibility API)
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsTabActive(!document.hidden);
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // 2. Monitor Scroll Position (Intersection Observer API)
  useEffect(() => {
    if (!containerRef.current) return;
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
      },
      { threshold: 0.15 } // Trigger when at least 15% of the fold is visible
    );

    observer.observe(containerRef.current);
    return () => {
      if (containerRef.current) {
        observer.unobserve(containerRef.current);
      }
    };
  }, []);

  // 3. Play/Pause based on visibility states
  useEffect(() => {
    if (!videoRef.current) return;

    if (isIntersecting && isTabActive) {
      videoRef.current.play().catch((err) => {
        console.warn("Video playback was deferred by client policy: ", err);
      });
    } else {
      videoRef.current.pause();
    }
  }, [isIntersecting, isTabActive]);

  return (
    <div 
      ref={containerRef}
      className="absolute inset-0 w-full h-full z-0 overflow-hidden bg-black select-none pointer-events-none"
    >
      <video
        ref={videoRef}
        src={VIDEO_URL}
        autoPlay
        muted
        loop
        playsInline
        className="w-full h-full object-cover opacity-80"
        style={{ filter: 'brightness(0.7) contrast(1.05)' }}
      />
    </div>
  );
});

BackgroundVideo.displayName = 'BackgroundVideo';

export default BackgroundVideo;
