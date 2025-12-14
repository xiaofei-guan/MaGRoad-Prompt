export interface Point {
  x: number;
  y: number;
}

export type PromptType = 'positive' | 'negative';

export interface Prompt extends Point {
  id: string;
  type: PromptType;
  label: 0 | 1;
}

export interface ImageInfo {
  id: string;
  filename: string;
  original_filename: string; // Original file name before hashing
  hasFeature: boolean; // Frontend uses camelCase
  url: string; // Backend provides this URL
  name: string;
  features_computed?: boolean;
  features_computing?: boolean;
  error_computing_features?: string | null;
  width?: number;
  height?: number;
  isAnnotated: boolean; // Whether this image has been annotated
  // Chunked download info (multi-resolution URLs removed - now only original)
  originalSize?: number;  // Original file size in bytes
  webpSize?: number;      // WebP file size in bytes
  webpAvailable?: boolean; // Whether WebP version is available
  supportsRange?: boolean; // Whether server supports Range requests
}

export type FeatureStatus = 'none' | 'computing' | 'ready' | 'error';

// === Road Network Types ===

// Individual node in the road network
export interface RoadNode {
    id: number;
    x: number;
    y: number;
}

// Individual edge connecting two nodes
export interface RoadEdge {
    id: number;
    source: number;  // Node ID
    target: number;  // Node ID
}

// GeoJSON Feature (for backward compatibility)
export interface GeoJSONFeature {
    type: 'Feature';
    geometry: {
        type: 'LineString';
        coordinates: [number, number][];
    };
    properties: Record<string, any>;
}

// Updated GeoJSON FeatureCollection for node-edge format only
export interface GeoJSONFeatureCollection {
    type: 'FeatureCollection';
    properties?: {
        format_version?: string;
        total_nodes?: number;
        total_edges?: number;
        coordinate_system?: string;
        image_id?: string;
        [key: string]: any;
    };
    // Node-edge format (v2.0+)
    nodes: RoadNode[];
    edges: RoadEdge[];
    // Keep features for backward compatibility with GeoJSON spec
    features: GeoJSONFeature[];
}

// Type alias for the road network data
export type RoadNetworkData = GeoJSONFeatureCollection;

// Legacy type for backward compatibility
export interface RoadFeature extends GeoJSONFeature {}

// --- API Response Types (matching backend schemas) ---

export interface ImageListResponse {
  images: ImageInfo[];
}

export interface UploadResponse {
  message: string;
  image_ids: string[];
  images: ImageInfo[];
}

export interface FeatureStatusResponse {
  image_id?: string;
  status: 'none' | 'pending' | 'processing' | 'computing' | 'completed' | 'ready' | 'failed' | 'error';
  message: string;
  error?: string;
}

// Reusing FeatureStatusResponse for Precompute start response
export interface PrecomputeResponse {
    message: string;
}

export interface SaveResponse {
    status: string; // e.g., "success"
    message: string;
}

// Matches backend schema app.schemas.roadnet.RoadNetGenerationResponse
export interface RoadNetGenerationResponse {
    image_id: string;
    geojson_data: GeoJSONFeatureCollection;
    prompts?: Prompt[]; // Prompts associated with this generated/loaded network
    road_mask?: boolean[][] | number[][] | string | null; // Support compressed, boolean, or number arrays
    kp_mask?: boolean[][] | number[][] | string | null;   // Support compressed, boolean, or number arrays
    road_mask_metadata?: { shape: [number, number]; dtype: string; compressed: boolean };
    kp_mask_metadata?: { shape: [number, number]; dtype: string; compressed: boolean };
}

// Matches backend schema app.schemas.roadnet.SaveRoadNetworkRequest
export interface SaveRoadNetworkPayload {
    image_id: string;
    geojson_data: GeoJSONFeatureCollection;
    prompts?: Prompt[];
    road_mask?: boolean[][] | number[][]; // DEPRECATED - backend uses cached data
    kp_mask?: boolean[][] | number[][];   // DEPRECATED - backend uses cached data
    use_cached_masks?: boolean;          // Whether to use backend cached masks
}

// Matches backend schema app.schemas.roadnet.SaveRoadNetworkResponse
export interface SaveRoadNetworkResponse {
  status: string; // e.g., "success"
  message: string;
  file_path?: string; // Optional, as per Pydantic schema
}

// --- Export types ---
export type ExportScope = 'current' | 'all';
export type CoordinateFormat = 'xy' | 'rc';

export interface ExportOptions {
  scope: ExportScope;
  imageId?: string;
  coordinateFormat: CoordinateFormat; // 'xy' or 'rc'
}

export interface ExportResponseMeta {
  exported: number;
  missing: number;
  filename: string; // suggested download filename
}

// --- Delete types ---
export type DeleteScope = 'current' | 'all';

export interface DeleteOptions {
  scope: DeleteScope;
  imageId?: string; // required if scope is current
  deleteAnnotations: boolean; // whether to also delete annotations
}

export interface DeleteResponseMeta {
  deleted: number;
  skipped: number;
  message?: string;
}

// Note: ImageResolution and ImageLoadingState types removed - no longer using progressive loading

// Chunk download types
export interface ImageChunk {
  index: number;
  start: number;
  end: number;
  size: number;
  data?: ArrayBuffer;
  status: 'pending' | 'downloading' | 'completed' | 'error';
  retries: number;
  downloadTime?: number; // milliseconds
}

export interface ChunkedDownloadOptions {
  chunkSize: number; // Size of each chunk in bytes
  maxConcurrentChunks: number; // Max number of chunks to download simultaneously
  maxRetries: number; // Max retries per chunk
  retryDelay: number; // Delay between retries in ms
  preferWebP: boolean; // Whether to prefer WebP format
  onProgress?: (progress: ChunkedDownloadProgress) => void;
  onChunkComplete?: (chunk: ImageChunk) => void;
  onComplete?: (imageBlob: Blob) => void;
  onError?: (error: string) => void;
}

export interface ChunkedDownloadProgress {
  totalSize: number;
  downloadedSize: number;
  percentage: number;
  completedChunks: number;
  totalChunks: number;
  downloadSpeed: number; // bytes per second
  estimatedTimeRemaining: number; // seconds
  activeDownloads: number;
}

export interface ImageFileInfo {
  image_id: string;
  original_size: number;
  webp_size?: number;
  webp_available: boolean;
  supports_range: boolean;
  recommended_chunk_size: number;
  max_concurrent_chunks: number;
}

// Ensure ImageInfo, ImageListResponse, UploadResponse, FeatureStatusResponse, PrecomputeResponse are defined above or imported
// Ensure Point, PromptType, Prompt, PointPrompt (if used by frontend directly), RoadNetworkData are defined above. 