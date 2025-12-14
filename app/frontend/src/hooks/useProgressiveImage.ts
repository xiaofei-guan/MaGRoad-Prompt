import { useState, useEffect, useCallback, useRef } from 'react';
import { ChunkedDownloadProgress } from '../types';
import { preloadImageWithTimeout, loadImageOptimized } from '../services/api';

interface UseProgressiveImageOptions {
  imageUrl?: string;
  imageId?: string; // Add imageId for chunked downloads
  preferWebP?: boolean; // Whether to prefer WebP format
  timeout?: number; // Single timeout for image loading
}

interface UseProgressiveImageReturn {
  loadingState: {
    isLoaded: boolean;
    isLoading: boolean;
    error?: string;
    isChunkedLoading: boolean;
    chunkedProgress: number;
    totalChunks: number;
    completedChunks: number;
    downloadSpeed: number;
    estimatedTimeRemaining: number;
  };
  currentImageElement: HTMLImageElement | null;
  loadImage: () => void;
  reset: () => void;
}

export const useProgressiveImage = ({
  imageUrl,
  imageId,
  preferWebP = false,
  timeout = 30000 // 30 seconds default timeout
}: UseProgressiveImageOptions): UseProgressiveImageReturn => {
  const [loadingState, setLoadingState] = useState({
    isLoaded: false,
    isLoading: false,
    error: undefined as string | undefined,
    isChunkedLoading: false,
    chunkedProgress: 0,
    totalChunks: 0,
    completedChunks: 0,
    downloadSpeed: 0,
    estimatedTimeRemaining: 0
  });

  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null);

  // Use refs to track current URLs to prevent unnecessary re-renders
  const urlsRef = useRef({ imageUrl, imageId });
  const isLoadingRef = useRef(false);
  const loadSeqRef = useRef(0);

  // Update refs when props change
  useEffect(() => {
    urlsRef.current = { imageUrl, imageId };
  }, [imageUrl, imageId]);

  // Throttled chunked download progress updates to avoid excessive re-renders
  const lastProgressRef = useRef<{ ts: number; pct: number }>({ ts: 0, pct: 0 });
  const progressRafRef = useRef<number | null>(null);
  const handleChunkedProgress = useCallback((progress: ChunkedDownloadProgress) => {
    const now = performance.now();
    const dt = now - lastProgressRef.current.ts;
    const dp = Math.abs(progress.percentage - lastProgressRef.current.pct);
    if (dt < 80 && dp < 0.5) return; // throttle: 80ms or 0.5%
    lastProgressRef.current = { ts: now, pct: progress.percentage };
    if (progressRafRef.current !== null) cancelAnimationFrame(progressRafRef.current);
    progressRafRef.current = requestAnimationFrame(() => {
      setLoadingState(prev => ({
        ...prev,
        isChunkedLoading: true,
        chunkedProgress: progress.percentage,
        totalChunks: progress.totalChunks,
        completedChunks: progress.completedChunks,
        downloadSpeed: progress.downloadSpeed,
        estimatedTimeRemaining: progress.estimatedTimeRemaining
      }));
    });
  }, []);

  // Simplified image loading function
  const loadImage = useCallback(async (): Promise<HTMLImageElement | null> => {
    // Prevent multiple concurrent loads
    if (isLoadingRef.current) {
      return null;
    }

    const { imageUrl: url, imageId: id } = urlsRef.current;
    
    if (!url && !id) {
      console.warn('No image URL or ID provided');
      return null;
    }

    let mySeq = 0;
    try {
      isLoadingRef.current = true;
      mySeq = ++loadSeqRef.current;
      
      setLoadingState(prev => ({ 
        ...prev, 
        isLoading: true, 
        error: undefined,
        isChunkedLoading: false,
        chunkedProgress: 0,
        completedChunks: 0,
        totalChunks: 0,
        downloadSpeed: 0,
        estimatedTimeRemaining: 0
      }));
      
      let imageElement: HTMLImageElement;

      // Prefer a direct load with automatic fallback inside loadImageOptimized
      if (id) {
        try {
          imageElement = await loadImageOptimized(id, preferWebP, handleChunkedProgress);
        } catch (error) {
          console.warn('Optimized load failed, falling back to direct URL:', error);
          if (url) imageElement = await preloadImageWithTimeout(url, timeout);
          else throw error;
        }
      } else if (url) {
        // Use regular loading for URL-based images
        imageElement = await preloadImageWithTimeout(url, timeout);
      } else {
        throw new Error('No valid image source provided');
      }
      
      if (mySeq === loadSeqRef.current) {
        setImageElement(imageElement);
        setLoadingState(prev => ({
          ...prev,
          isLoaded: true,
          isLoading: false,
          isChunkedLoading: false
        }));
      }

      console.log('Loaded image successfully');
      return imageElement;
    } catch (error) {
      const errorMessage = `Failed to load image: ${error instanceof Error ? error.message : 'Unknown error'}`;
      console.error(errorMessage);
      
      if (mySeq === loadSeqRef.current) {
        setLoadingState(prev => ({
          ...prev,
          isLoading: false,
          isChunkedLoading: false,
          error: errorMessage
        }));
      }
      
      return null;
    } finally {
      // Only clear loading flag if this is the latest request
      if (mySeq === loadSeqRef.current) {
        isLoadingRef.current = false;
      }
    }
  }, [preferWebP, handleChunkedProgress, timeout]);

  const reset = useCallback(() => {
    isLoadingRef.current = false;
    if (progressRafRef.current !== null) {
      cancelAnimationFrame(progressRafRef.current);
      progressRafRef.current = null;
    }
    setLoadingState({
      isLoaded: false,
      isLoading: false,
      error: undefined,
      isChunkedLoading: false,
      chunkedProgress: 0,
      totalChunks: 0,
      completedChunks: 0,
      downloadSpeed: 0,
      estimatedTimeRemaining: 0
    });
    setImageElement(null);
  }, []);

  // Auto-load image when URLs change
  useEffect(() => {
    // Reset state when URLs change
    reset();
    
    // Load image with a small delay to ensure component is mounted
    const timeoutId = setTimeout(() => {
      if (imageUrl || imageId) {
        loadImage();
      }
    }, 10);

    return () => {
      clearTimeout(timeoutId);
      isLoadingRef.current = false;
      if (progressRafRef.current !== null) {
        cancelAnimationFrame(progressRafRef.current);
        progressRafRef.current = null;
      }
    };
  }, [imageUrl, imageId, loadImage, reset]);

  return {
    loadingState,
    currentImageElement: imageElement,
    loadImage,
    reset
  };
}; 