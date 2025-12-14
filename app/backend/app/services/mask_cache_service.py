import time
import threading
import gzip
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger(__name__)

class MaskCacheEntry:
    """
    Individual cache entry for compressed mask data.
    Stores masks in compressed format to reduce memory usage.
    """
    
    def __init__(self, road_mask: Optional[np.ndarray], kp_mask: Optional[np.ndarray]):
        self.timestamp = time.time()
        self.access_count = 0
        
        # **🔥 MEMORY OPTIMIZATION**: Store compressed data instead of raw arrays
        self._compressed_road_mask = self._compress_mask(road_mask) if road_mask is not None else None
        self._compressed_kp_mask = self._compress_mask(kp_mask) if kp_mask is not None else None
        
        # Store metadata for decompression
        self._road_mask_shape = road_mask.shape if road_mask is not None else None
        self._kp_mask_shape = kp_mask.shape if kp_mask is not None else None
        
        # Log compression statistics
        if road_mask is not None or kp_mask is not None:
            self._log_compression_stats(road_mask, kp_mask)
    
    def _compress_mask(self, mask_array: np.ndarray) -> bytes:
        """
        Compress mask data using gzip compression.
        
        Args:
            mask_array: Input mask array
            
        Returns:
            Compressed bytes data
        """
        try:
            # Convert to boolean for better compression
            mask_bool = mask_array.astype(np.bool_)
            # Use compression level 3 for balance between size and speed
            return gzip.compress(mask_bool.tobytes(), compresslevel=3)
        except Exception as e:
            logger.error(f"Failed to compress mask: {e}")
            # Fallback to storing as bytes without compression
            return mask_array.astype(np.bool_).tobytes()
    
    def _decompress_mask(self, compressed_data: bytes, shape: Tuple[int, int]) -> np.ndarray:
        """
        Decompress mask data.
        
        Args:
            compressed_data: Compressed bytes data
            shape: Original mask shape
            
        Returns:
            Decompressed numpy array
        """
        try:
            # Try gzip decompression first
            decompressed_bytes = gzip.decompress(compressed_data)
            return np.frombuffer(decompressed_bytes, dtype=np.bool_).reshape(shape)
        except Exception:
            # Fallback: assume it's uncompressed bytes
            return np.frombuffer(compressed_data, dtype=np.bool_).reshape(shape)
    
    def _log_compression_stats(self, road_mask: Optional[np.ndarray], kp_mask: Optional[np.ndarray]) -> None:
        """Log compression statistics for monitoring"""
        try:
            original_size = 0
            compressed_size = 0
            
            if road_mask is not None:
                original_size += road_mask.nbytes
                compressed_size += len(self._compressed_road_mask)
            
            if kp_mask is not None:
                original_size += kp_mask.nbytes
                compressed_size += len(self._compressed_kp_mask)
            
            if original_size > 0:
                compression_ratio = compressed_size / original_size
                saved_mb = (original_size - compressed_size) / (1024 * 1024)
                logger.debug(f"Cache compression: {original_size} → {compressed_size} bytes "
                           f"(ratio: {compression_ratio:.3f}, saved: {saved_mb:.1f}MB)")
                
        except Exception as e:
            logger.warning(f"Failed to log compression stats: {e}")
    
    @property
    def road_mask(self) -> Optional[np.ndarray]:
        """
        Get road mask, decompressing on demand.
        
        Returns:
            Decompressed road mask array or None
        """
        if self._compressed_road_mask is None:
            return None
        return self._decompress_mask(self._compressed_road_mask, self._road_mask_shape)
    
    @property  
    def kp_mask(self) -> Optional[np.ndarray]:
        """
        Get keypoint mask, decompressing on demand.
        
        Returns:
            Decompressed keypoint mask array or None
        """
        if self._compressed_kp_mask is None:
            return None
        return self._decompress_mask(self._compressed_kp_mask, self._kp_mask_shape)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics for this cache entry.
        
        Returns:
            Dictionary with memory usage information
        """
        road_compressed_size = len(self._compressed_road_mask) if self._compressed_road_mask else 0
        kp_compressed_size = len(self._compressed_kp_mask) if self._compressed_kp_mask else 0
        
        return {
            'road_mask_compressed_bytes': road_compressed_size,
            'kp_mask_compressed_bytes': kp_compressed_size,
            'total_compressed_bytes': road_compressed_size + kp_compressed_size,
            'has_road_mask': self._compressed_road_mask is not None,
            'has_kp_mask': self._compressed_kp_mask is not None
        }
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        expiry_time = self.timestamp + (settings.MASK_CACHE_EXPIRY_HOURS * 3600)
        return time.time() > expiry_time
    
    def access(self) -> None:
        """Mark this entry as accessed"""
        self.access_count += 1

class MaskCacheService:
    """
    Service for caching mask data to avoid repeated transmission.
    Supports both regular and large images with automatic cleanup.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._cache: Dict[str, MaskCacheEntry] = {}
            self._cache_lock = threading.RLock()
            self._last_cleanup = time.time()
            self.initialized = True
            logger.info("MaskCacheService initialized")
    
    def cache_masks(
        self, 
        image_id: str, 
        road_mask: Optional[np.ndarray], 
        kp_mask: Optional[np.ndarray]
    ) -> None:
        """
        Cache mask data for the given image ID.
        
        Args:
            image_id: The image identifier
            road_mask: Road mask numpy array (optional)
            kp_mask: Keypoint mask numpy array (optional)
        """
        try:
            with self._cache_lock:
                # Perform cleanup if needed
                self._cleanup_if_needed()
                
                # Create cache entry
                entry = MaskCacheEntry(road_mask, kp_mask)
                self._cache[image_id] = entry
                
                # Enforce cache size limit
                self._enforce_size_limit()
                
                logger.info(f"Cached masks for image_id: {image_id} (road_mask: {road_mask is not None}, kp_mask: {kp_mask is not None})")
                logger.debug(f"Cache size: {len(self._cache)}")
                
        except Exception as e:
            logger.error(f"Error caching masks for image_id {image_id}: {e}")
    
    def get_cached_masks(self, image_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve cached mask data for the given image ID.
        
        Args:
            image_id: The image identifier
            
        Returns:
            Tuple of (road_mask, kp_mask) or (None, None) if not cached
        """
        try:
            with self._cache_lock:
                entry = self._cache.get(image_id)
                
                if entry is None:
                    logger.debug(f"No cached masks found for image_id: {image_id}")
                    return None, None
                
                if entry.is_expired():
                    logger.info(f"Cached masks expired for image_id: {image_id}, removing from cache")
                    del self._cache[image_id]
                    return None, None
                
                # Mark as accessed
                entry.access()
                
                logger.info(f"Retrieved cached masks for image_id: {image_id} (road_mask: {entry.road_mask is not None}, kp_mask: {entry.kp_mask is not None})")
                return entry.road_mask, entry.kp_mask
                
        except Exception as e:
            logger.error(f"Error retrieving cached masks for image_id {image_id}: {e}")
            return None, None
    
    def has_cached_masks(self, image_id: str) -> bool:
        """
        Check if masks are cached for the given image ID.
        
        Args:
            image_id: The image identifier
            
        Returns:
            True if valid cached masks exist, False otherwise
        """
        with self._cache_lock:
            entry = self._cache.get(image_id)
            return entry is not None and not entry.is_expired()
    
    def remove_cached_masks(self, image_id: str) -> None:
        """
        Remove cached masks for the given image ID.
        
        Args:
            image_id: The image identifier
        """
        try:
            with self._cache_lock:
                if image_id in self._cache:
                    del self._cache[image_id]
                    logger.info(f"Removed cached masks for image_id: {image_id}")
        except Exception as e:
            logger.error(f"Error removing cached masks for image_id {image_id}: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached masks"""
        try:
            with self._cache_lock:
                cache_size = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared mask cache ({cache_size} entries removed)")
        except Exception as e:
            logger.error(f"Error clearing mask cache: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Perform cleanup if enough time has passed"""
        current_time = time.time()
        if current_time - self._last_cleanup > settings.MASK_CACHE_CLEANUP_INTERVAL:
            self._cleanup_expired_entries()
            self._last_cleanup = current_time
    
    def _cleanup_expired_entries(self) -> None:
        """Remove expired cache entries"""
        try:
            expired_keys = []
            for image_id, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(image_id)
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size by removing least recently used entries"""
        try:
            if len(self._cache) > settings.MASK_CACHE_MAX_SIZE:
                # Sort by access count (ascending) and timestamp (ascending)
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: (x[1].access_count, x[1].timestamp)
                )
                
                # Remove oldest/least accessed entries
                excess_count = len(self._cache) - settings.MASK_CACHE_MAX_SIZE
                for i in range(excess_count):
                    image_id, _ = sorted_items[i]
                    del self._cache[image_id]
                
                logger.info(f"Enforced cache size limit: removed {excess_count} entries")
                
        except Exception as e:
            logger.error(f"Error enforcing cache size limit: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics including memory usage.
        
        Returns:
            Dictionary with cache statistics and memory usage information
        """
        with self._cache_lock:
            total_memory = 0
            entries_info = {}
            
            for image_id, entry in self._cache.items():
                memory_usage = entry.get_memory_usage()
                total_memory += memory_usage['total_compressed_bytes']
                
                entries_info[image_id] = {
                    'has_road_mask': memory_usage['has_road_mask'],
                    'has_kp_mask': memory_usage['has_kp_mask'],
                    'access_count': entry.access_count,
                    'age_seconds': time.time() - entry.timestamp,
                    'memory_usage_bytes': memory_usage['total_compressed_bytes'],
                    'road_mask_bytes': memory_usage['road_mask_compressed_bytes'],
                    'kp_mask_bytes': memory_usage['kp_mask_compressed_bytes'],
                    'is_expired': entry.is_expired()
                }
            
            stats = {
                'total_entries': len(self._cache),
                'total_memory_bytes': total_memory,
                'total_memory_mb': round(total_memory / (1024 * 1024), 2),
                'average_memory_per_entry_mb': round((total_memory / len(self._cache)) / (1024 * 1024), 2) if len(self._cache) > 0 else 0,
                'max_size': settings.MASK_CACHE_MAX_SIZE,
                'expiry_hours': settings.MASK_CACHE_EXPIRY_HOURS,
                'last_cleanup': datetime.fromtimestamp(self._last_cleanup).isoformat(),
                'entries_by_image': entries_info
            }
            
            return stats

# Global instance
mask_cache_service = MaskCacheService() 