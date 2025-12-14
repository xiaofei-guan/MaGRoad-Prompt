import axios, { type AxiosProgressEvent, type AxiosResponse } from 'axios';
// Ensure correct import path for types
import { 
  ImageListResponse, 
  UploadResponse, 
  FeatureStatusResponse, 
  PrecomputeResponse, 
  RoadNetworkData, 
  Prompt, 
  RoadNetGenerationResponse, 
  SaveRoadNetworkPayload, 
  SaveRoadNetworkResponse,
  // Add chunked download types
  ImageFileInfo,
  ChunkedDownloadOptions,
  ChunkedDownloadProgress,
  ImageChunk,
  ExportOptions,
  ExportResponseMeta,
  DeleteOptions,
  DeleteResponseMeta
} from '../types'; 

// ... (API_BASE_URL, apiClient, handleError remain the same)
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds
  headers: {
    'Content-Type': 'application/json',
  }
});

const handleError = (error: any, context: string) => {
  console.error(`API Error [${context}]:`, error);
  if (axios.isAxiosError(error)) {
    // Re-throw the original Axios error to allow callers to inspect response status etc.
    throw error;
  } else if (error instanceof Error) {
    throw error;
  } else {
    throw new Error('An unexpected error occurred');
  }
};

// Export imageApi directly
export const imageApi = {
  // ... (uploadImages remains the same)
    uploadImages: async (files: File[]): Promise<UploadResponse> => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file, file.name);
    });

    try {
      const response = await apiClient.post<UploadResponse>('/images/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        // For large multi-file uploads, allow a much longer timeout
        timeout: 0,
      });
      return response.data;
    } catch (error) {
      handleError(error, 'uploadImages');
      throw error;
    }
  },
  /**
   * (removed) getDevice: backend device display removed by request
   */
  /**
   * Delete images and optionally their annotations.
   * POST /images/delete with JSON body { scope, image_id?, delete_annotations }
   */
  deleteImages: async (options: DeleteOptions): Promise<DeleteResponseMeta> => {
    try {
      const body: any = {
        scope: options.scope === 'all' ? 'all' : 'current',
        delete_annotations: !!options.deleteAnnotations,
      };
      if (options.scope === 'current' && options.imageId) body.image_id = options.imageId;

      const response = await apiClient.post<DeleteResponseMeta>('/images/delete', body);
      return response.data;
    } catch (error) {
      handleError(error, 'deleteImages');
      throw error;
    }
  },
  /**
   * Upload a single image file with progress callback and extended timeout.
   */
  uploadSingleImage: async (
    file: File,
    onProgress?: (uploadedBytes: number, totalBytes: number, file: File) => void
  ): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('files', file, file.name);

    try {
      const response: AxiosResponse<UploadResponse> = await apiClient.post('/images/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 0, // disable client timeout for large files
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (onProgress && typeof progressEvent.loaded === 'number' && typeof progressEvent.total === 'number') {
            onProgress(progressEvent.loaded, progressEvent.total, file);
          }
        },
      });
      return response.data as UploadResponse;
    } catch (error) {
      handleError(error, 'uploadSingleImage');
      throw error;
    }
  },
  // ... (listImages remains the same)
    listImages: async (): Promise<ImageListResponse> => {
    try {
      const response = await apiClient.get<ImageListResponse>('/images/list');
      return response.data;
    } catch (error) {
      handleError(error, 'listImages');
      throw error;
    }
  },
  // ... (startPrecomputeFeature remains the same)
   startPrecomputeFeature: async (imageId: string): Promise<PrecomputeResponse> => {
    try {
      const response = await apiClient.post<PrecomputeResponse>('/images/precompute-feature', { image_id: imageId });
      return response.data;
    } catch (error) {
      handleError(error, 'startPrecomputeFeature');
      throw error;
    }
  },
  // ... (checkFeatureStatus remains the same)
  checkFeatureStatus: async (imageId: string): Promise<FeatureStatusResponse> => {
    try {
      const response = await apiClient.get<FeatureStatusResponse>(`/images/feature-status/${imageId}`);
      return response.data;
    } catch (error) {
      handleError(error, 'checkFeatureStatus');
      throw error;
    }
  },
  /**
   * Recompute features: reset existing features and start computing again.
   */
  recomputeFeatures: async (imageId: string): Promise<PrecomputeResponse> => {
    try {
      const response = await apiClient.post<PrecomputeResponse>(`/images/recompute-feature`, { image_id: imageId });
      return response.data;
    } catch (error) {
      handleError(error, 'recomputeFeatures');
      throw error;
    }
  },
};

// Export roadNetworkApi directly
export const roadNetworkApi = {
   /**
   * Generates road network based on image and prompts.
   */
  generateRoadNetwork: async (imageId: string, prompts: any[]) => {
    try {
      const response = await fetch(`${API_BASE_URL}/road-network/generate-from-prompts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_id: imageId,
          prompts: prompts
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to generate road network: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API error in generateRoadNetwork:', error);
      throw error;
    }
  },

  /**
   * Saves the edited road network.
   */
  saveRoadNetwork: async (imageId: string, geojsonData: any, prompts: any[], roadMask?: number[][], kpMask?: number[][]) => {
    try {
      const response = await fetch(`${API_BASE_URL}/road-network/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_id: imageId,
          geojson_data: geojsonData,
          prompts: prompts,
          road_mask: roadMask,
          kp_mask: kpMask
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to save road network: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API error in saveRoadNetwork:', error);
      throw error;
    }
  },

  loadSavedRoadNetwork: async (imageId: string, abortSignal?: AbortSignal) => {
    try {
      // Create an AbortController for timeout handling if no external signal provided
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 1 minute timeout

      // Combine external abort signal with timeout controller
      // Note: AbortSignal.any is not widely supported, use manual approach for better compatibility
      if (abortSignal) {
        // If external signal is already aborted, abort our controller immediately
        if (abortSignal.aborted) {
          controller.abort();
        } else {
          // Listen for external signal abort and propagate it
          abortSignal.addEventListener('abort', () => {
            controller.abort();
          }, { once: true });
        }
      }

              const response = await fetch(`${API_BASE_URL}/road-network/${imageId}`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
          signal: controller.signal,
        });

      clearTimeout(timeoutId);

      if (response.status === 404) {
        // No saved network found, return null to indicate this
        return null;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to load road network: ${response.status}`);
      }

      return await response.json();
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log(`Load request aborted for image ${imageId}`);
        throw error; // Re-throw AbortError for proper handling in caller
      }
      
      console.error('API error in loadSavedRoadNetwork:', error);
      throw error;
    }
  }
};

// Add this new function
export const generateRoadNetworkAPI = async (imageId: string, prompts: Prompt[]): Promise<RoadNetworkData | null> => {
  console.log(`[API] Requesting network generation for imageId: ${imageId}, prompts:`, prompts);
  try {
    const response = await axios.post<RoadNetGenerationResponse>(
      `${API_BASE_URL}/road-network/generate-from-prompts`,
      { image_id: imageId, prompts: prompts.map(p => ({ x: p.x, y: p.y, label: p.label })) } // Ensure prompts are in backend format
    );
    console.log("[API] Network generation response:", response.data);
    return response.data.geojson_data;
  } catch (error) {
    console.error("[API] Error generating road network:", error);
    // Consider re-throwing a more specific error or handling it as per application needs
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(error.response.data.detail || 'Failed to generate road network');
    } else {
      throw new Error('Failed to generate road network due to an unknown error.');
    }
  }
};

// --- Start of New API Function ---
export const saveRoadNetworkAPI = async (
  imageId: string,
  networkData: RoadNetworkData,
  prompts?: any[],
  roadMask?: boolean[][] | number[][],  // DEPRECATED - backend uses cached data
  kpMask?: boolean[][] | number[][]     // DEPRECATED - backend uses cached data
): Promise<SaveRoadNetworkResponse> => {
  console.log(`[API] Requesting to save road network for imageId: ${imageId} (using backend cache for masks)`);
  const requestBody: SaveRoadNetworkPayload = {
    image_id: imageId,
    geojson_data: networkData,
    prompts: prompts,
    road_mask: roadMask,   // Only sent as fallback for backwards compatibility
    kp_mask: kpMask,       // Only sent as fallback for backwards compatibility
    use_cached_masks: true // Prioritize backend cached masks
  };
  try {
    const response = await axios.post<SaveRoadNetworkResponse>(
      `${API_BASE_URL}/road-network/save`,
      requestBody
    );
    console.log("[API] Save road network response:", response.data);
    return response.data;
  } catch (error) {
    console.error("[API] Error saving road network:", error);
    if (axios.isAxiosError(error) && error.response) {
      // Assuming the backend returns a similar error structure with 'detail'
      throw new Error(error.response.data.detail || 'Failed to save road network');
    } else {
      throw new Error('Failed to save road network due to an unknown error.');
    }
  }
};
// --- End of New API Function --- 

// Image URL utility function (simplified - now only supports original resolution)
export const getImageUrl = (imageId: string): string => {
  const baseUrl = API_BASE_URL.replace('/api', '');
  return `${baseUrl}/api/images/${imageId}`;
};

// Explicit WebP route kept for backward compatibility, but not preferred by default
export const getImageWebPUrl = (imageId: string): string => {
  const baseUrl = API_BASE_URL.replace('/api', '');
  return `${baseUrl}/api/images/${imageId}/webp`;
};

// Preload image with timeout and error handling
export const preloadImageWithTimeout = (url: string, timeout: number = 10000): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    let isResolved = false;

    const cleanup = () => {
      img.onload = null;
      img.onerror = null;
    };

    const timeoutId = setTimeout(() => {
      if (!isResolved) {
        isResolved = true;
        cleanup();
        reject(new Error(`Image load timeout after ${timeout}ms: ${url}`));
      }
    }, timeout);

    img.onload = () => {
      if (!isResolved) {
        isResolved = true;
        cleanup();
        clearTimeout(timeoutId);
        resolve(img);
      }
    };

    img.onerror = () => {
      if (!isResolved) {
        isResolved = true;
        cleanup();
        clearTimeout(timeoutId);
        reject(new Error(`Failed to load image: ${url}`));
      }
    };

    // Enable CORS for cross-origin images if needed
    img.crossOrigin = 'anonymous';
    img.src = url;
  });
};

// Get image file information for chunked downloads
export const getImageInfo = async (imageId: string): Promise<ImageFileInfo> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/images/${imageId}/info`);
    return response.data;
  } catch (error) {
    console.error('Failed to get image info:', error);
    throw error;
  }
};

// Download a single chunk with Range request
export const downloadChunk = async (
  url: string, 
  start: number, 
  end: number,
  preferWebP: boolean = false
): Promise<ArrayBuffer> => {
  try {
    const headers: Record<string, string> = {
      'Range': `bytes=${start}-${end}`
    };

    // Avoid preferring WebP by default to reduce server-side conversion overhead
    if (preferWebP) headers['Accept'] = 'image/webp,image/*,*/*;q=0.8';

    const response = await axios.get(url, {
      headers,
      responseType: 'arraybuffer',
      timeout: 30000 // 30 second timeout per chunk
    });

    if (response.status === 206 || response.status === 200) {
      return response.data;
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error) {
    console.error(`Failed to download chunk ${start}-${end}:`, error);
    throw error;
  }
};

// Chunked download with parallel processing
export const downloadImageChunked = async (
  imageId: string,
  options: ChunkedDownloadOptions
): Promise<HTMLImageElement> => {
  const {
    chunkSize,
    maxConcurrentChunks,
    maxRetries,
    retryDelay,
    // preferWebP (unused; default disabled for performance)
    onProgress,
    onChunkComplete,
    onComplete,
    onError
  } = options;

  try {
    // Get image info first
    const imageInfo = await getImageInfo(imageId);
    
    if (!imageInfo.supports_range) {
      throw new Error('Server does not support Range requests');
    }

    // Choose original format and URL
    const fileSize = imageInfo.original_size;
    const imageUrl = getImageUrl(imageId);

    // Calculate chunks
    const totalChunks = Math.ceil(fileSize / chunkSize);
    const chunks: ImageChunk[] = [];

    for (let i = 0; i < totalChunks; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize - 1, fileSize - 1);
      chunks.push({
        index: i,
        start,
        end,
        size: end - start + 1,
        status: 'pending',
        retries: 0
      });
    }

    let completedChunks = 0;
    let downloadedSize = 0;
    const startTime = Date.now();
    const chunkResults = new Array(totalChunks);
    let activeDownloads = 0;

    const updateProgress = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      const downloadSpeed = downloadedSize / elapsed;
      const remainingSize = fileSize - downloadedSize;
      const estimatedTimeRemaining = downloadSpeed > 0 ? remainingSize / downloadSpeed : 0;

      const progress: ChunkedDownloadProgress = {
        totalSize: fileSize,
        downloadedSize,
        percentage: (downloadedSize / fileSize) * 100,
        completedChunks,
        totalChunks,
        downloadSpeed,
        estimatedTimeRemaining,
        activeDownloads
      };

      onProgress?.(progress);
    };

    // Download chunk with retry logic
    const downloadChunkWithRetry = async (chunk: ImageChunk): Promise<void> => {
      activeDownloads++;
      chunk.status = 'downloading';

      try {
        const chunkStart = Date.now();
        const data = await downloadChunk(imageUrl, chunk.start, chunk.end, false);
        
        chunk.data = data;
        chunk.status = 'completed';
        chunk.downloadTime = Date.now() - chunkStart;
        chunkResults[chunk.index] = data;
        
        completedChunks++;
        downloadedSize += chunk.size;
        
        onChunkComplete?.(chunk);
        updateProgress();
        
      } catch (error) {
        chunk.retries++;
        
        if (chunk.retries <= maxRetries) {
          // Retry after delay
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          return downloadChunkWithRetry(chunk);
        } else {
          chunk.status = 'error';
          throw new Error(`Failed to download chunk ${chunk.index} after ${maxRetries} retries: ${error}`);
        }
      } finally {
        activeDownloads--;
      }
    };

    // Process chunks with concurrency limit
    const downloadPromises: Promise<void>[] = [];
    let chunkIndex = 0;

    const processNextChunk = async (): Promise<void> => {
      if (chunkIndex < chunks.length) {
        const chunk = chunks[chunkIndex++];
        await downloadChunkWithRetry(chunk);
        
        // Process next chunk if more are available
        if (chunkIndex < chunks.length) {
          return processNextChunk();
        }
      }
    };

    // Start initial concurrent downloads
    for (let i = 0; i < Math.min(maxConcurrentChunks, chunks.length); i++) {
      downloadPromises.push(processNextChunk());
    }

    // Wait for all downloads to complete
    await Promise.all(downloadPromises);

    // Combine chunks into a single blob
    const imageBlob = new Blob(chunkResults);
    
    onComplete?.(imageBlob);

    // Create image element from blob
    return new Promise((resolve, reject) => {
      const img = new Image();
      const blobUrl = URL.createObjectURL(imageBlob);
      
      img.onload = () => {
        URL.revokeObjectURL(blobUrl);
        resolve(img);
      };
      
      img.onerror = () => {
        URL.revokeObjectURL(blobUrl);
        reject(new Error('Failed to create image from downloaded chunks'));
      };
      
      img.src = blobUrl;
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    onError?.(errorMessage);
    throw error;
  }
};

// Streaming download with progress (single request, no Range). Shows speed and percentage.
export const downloadImageStreaming = async (
  imageId: string,
  onProgress?: (progress: ChunkedDownloadProgress) => void
): Promise<HTMLImageElement> => {
  const imageUrl = getImageUrl(imageId);
  const response = await fetch(imageUrl);
  if (!response.ok || !response.body) {
    // Fallback: load directly if streaming unsupported
    return preloadImageWithTimeout(imageUrl, 30000);
  }

  const contentType = response.headers.get('Content-Type') || undefined;
  const totalHeader = response.headers.get('Content-Length') || response.headers.get('content-length');
  const totalSize = totalHeader ? parseInt(totalHeader, 10) : 0;

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let downloadedSize = 0;
  const startTime = Date.now();

  const report = () => {
    const elapsed = Math.max(0.001, (Date.now() - startTime) / 1000);
    const downloadSpeed = downloadedSize / elapsed;
    const remainingSize = totalSize > 0 ? totalSize - downloadedSize : 0;
    const estimatedTimeRemaining = downloadSpeed > 0 && totalSize > 0 ? remainingSize / downloadSpeed : 0;
    onProgress?.({
      totalSize: totalSize || downloadedSize, // Avoid 0 to keep UI sane
      downloadedSize,
      percentage: totalSize > 0 ? (downloadedSize / totalSize) * 100 : 0,
      completedChunks: downloadedSize,
      totalChunks: totalSize,
      downloadSpeed,
      estimatedTimeRemaining,
      activeDownloads: 1
    });
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      downloadedSize += value.length;
      report();
    }
  }

  const blob = new Blob(chunks, { type: contentType });
  return new Promise((resolve, reject) => {
    const img = new Image();
    const blobUrl = URL.createObjectURL(blob);
    img.onload = () => {
      URL.revokeObjectURL(blobUrl);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(blobUrl);
      reject(new Error('Failed to create image from streamed data'));
    };
    img.crossOrigin = 'anonymous';
    img.src = blobUrl;
  });
};

// Optimized image loading with automatic chunked download for large images
export const loadImageOptimized = async (
  imageId: string,
  _preferWebP: boolean = false,
  onProgress?: (progress: ChunkedDownloadProgress) => void
): Promise<HTMLImageElement> => {
  try {
    // Decide path based on file size for better UX: use streaming with progress for large files
    const imageInfo = await getImageInfo(imageId);
    const fileSize = imageInfo.original_size;
    if (fileSize > 2 * 1024 * 1024) {
      // Show progress via streaming
      return await downloadImageStreaming(imageId, onProgress);
    }
    // Small files: direct load
    const directUrl = getImageUrl(imageId);
    return await preloadImageWithTimeout(directUrl, 30000);
  } catch (error) {
    console.error(`Failed to load image ${imageId}:`, error);
    // Ultimate fallback: try chunked if available
    try {
      const imageInfo = await getImageInfo(imageId);
      return await downloadImageChunked(imageId, {
        chunkSize: imageInfo.recommended_chunk_size,
        maxConcurrentChunks: Math.min(imageInfo.max_concurrent_chunks, 6),
        maxRetries: 3,
        retryDelay: 1000,
        preferWebP: false,
        onProgress
      });
    } catch (e) {
      throw error;
    }
  }
}; 

// --- Export API ---
export const exportRoadNetworks = async (options: ExportOptions): Promise<{ blob: Blob; meta: ExportResponseMeta }> => {
  const url = `${API_BASE_URL}/road-network/export`;
  const body: any = {
    scope: options.scope === 'all' ? 'all' : 'current',
    coordinate_format: options.coordinateFormat,
  };
  if (options.scope === 'current' && options.imageId) {
    body.image_id = options.imageId;
  }
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const msg = await response.text();
    throw new Error(msg || `Export failed with status ${response.status}`);
  }
  const blob = await response.blob();
  // Stats from header
  const statsHeader = response.headers.get('X-Export-Stats');
  let meta: ExportResponseMeta = { exported: 0, missing: 0, filename: options.scope === 'all' ? 'all.zip' : 'export.pickle' };
  if (statsHeader) {
    try {
      meta = JSON.parse(statsHeader);
    } catch {}
  }
  return { blob, meta };
};