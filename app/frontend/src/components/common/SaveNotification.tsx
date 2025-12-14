import React, { useEffect, useState, useRef } from 'react';

function SaveNotification({ 
  show, 
  onDismiss
}: { 
  show: boolean; 
  onDismiss: () => void;
}) {
  const [opacity, setOpacity] = useState(0);
  const dismissTimerRef = useRef<NodeJS.Timeout | null>(null);
  const fadeOutTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Handle showing and auto-dismissing the notification
  useEffect(() => {
    // Clear any existing timers when the effect runs
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
    
    if (fadeOutTimerRef.current) {
      clearTimeout(fadeOutTimerRef.current);
      fadeOutTimerRef.current = null;
    }
    
    if (show) {
      // Show the notification
      setOpacity(1);
      
      // Set timer to start fade out
      dismissTimerRef.current = setTimeout(() => {
        setOpacity(0);
        
        // Set timer to remove from DOM after fade completes
        fadeOutTimerRef.current = setTimeout(() => {
          onDismiss();
        }, 300); // Match the duration of the CSS transition
      }, 2000);
    } else {
      // If show becomes false from outside, ensure we fade out properly
      setOpacity(0);
      
      // Ensure component is fully removed after transition
      fadeOutTimerRef.current = setTimeout(() => {
        onDismiss();
      }, 300);
    }
    
    // Cleanup all timers when component unmounts or effect reruns
    return () => {
      if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
      if (fadeOutTimerRef.current) clearTimeout(fadeOutTimerRef.current);
    };
  }, [show, onDismiss]);

  // Don't render anything if not showing and fully transparent
  if (!show && opacity === 0) return null;
  
  return (
    <div 
      className="fixed left-1/2 transform -translate-x-1/2 top-4 bg-green-50 border border-green-500 text-green-700 px-4 py-2 rounded shadow-md flex items-center transition-opacity duration-300 ease-in-out z-10"
      style={{ opacity }}
      data-testid="save-notification"
    >
      <span className="font-medium">Saved successfully</span>
      <svg 
        className="w-4 h-4 ml-1 text-green-600" 
        fill="none" 
        stroke="currentColor" 
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          strokeWidth="2" 
          d="M5 13l4 4L19 7"
        />
      </svg>
    </div>
  );
}

export default SaveNotification;