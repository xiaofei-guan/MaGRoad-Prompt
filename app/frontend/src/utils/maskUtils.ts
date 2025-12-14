/**
 * Utilities for efficient mask data processing and rendering
 * Handles compression/decompression and optimized rendering for large images
 */

import pako from 'pako';

// Types for mask data handling
export interface MaskMetadata {
    shape: [number, number];
    dtype: string;
    compressed: boolean;
}

export interface MaskData {
    data: boolean[][] | string;
    metadata?: MaskMetadata;
}

/**
 * Decompress mask data from base64 encoded gzip format
 */
export function decompressMaskData(
    compressedData: string, 
    metadata: MaskMetadata
): boolean[][] {
    try {
        // Decode base64
        const binaryString = atob(compressedData);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        // Decompress with pako (gzip)
        const decompressed = pako.inflate(bytes);
        
        // Convert to boolean array
        const [height, width] = metadata.shape;
        const result: boolean[][] = [];
        
        for (let y = 0; y < height; y++) {
            const row: boolean[] = [];
            for (let x = 0; x < width; x++) {
                const index = y * width + x;
                row.push(decompressed[index] === 1);
            }
            result.push(row);
        }
        
        return result;
    } catch (error) {
        console.error('Failed to decompress mask data:', error);
        return [];
    }
}

/**
 * Process mask data (decompress if needed)
 */
export function processMaskData(maskData: MaskData): boolean[][] | null {
    if (!maskData.data) return null;
    
    // If data is already decompressed (array format)
    if (Array.isArray(maskData.data)) {
        return maskData.data;
    }
    
    // If data is compressed (string format)
    if (typeof maskData.data === 'string' && maskData.metadata) {
        return decompressMaskData(maskData.data, maskData.metadata);
    }
    
    return null;
}

/**
 * Create optimized canvas for mask rendering using ImageData
 * This runs in the main thread but with optimized pixel operations
 */
export function createMaskCanvas(
    maskData: boolean[][],
    width: number,
    height: number,
    color: { r: number; g: number; b: number; a: number }
): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    
    // Create ImageData for efficient pixel manipulation
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    
    // Optimized pixel filling using typed array operations
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const maskValue = (y < maskData.length && x < maskData[y].length) 
                ? maskData[y][x] 
                : false;
            
            if (maskValue) {
                const idx = (y * width + x) * 4;
                data[idx] = color.r;     // R
                data[idx + 1] = color.g; // G
                data[idx + 2] = color.b; // B
                data[idx + 3] = color.a; // A
            }
            // Transparent pixels are already 0 by default
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

/**
 * Worker script for mask processing (as string to avoid separate file)
 * This can be used with Web Workers for large mask processing
 */
export const maskWorkerScript = `
self.onmessage = function(e) {
    const { maskData, width, height, color, taskId } = e.data;
    
    try {
        // Create ImageData buffer
        const buffer = new ArrayBuffer(width * height * 4);
        const data = new Uint8ClampedArray(buffer);
        
        // Process mask data
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const maskValue = (y < maskData.length && x < maskData[y].length) 
                    ? maskData[y][x] 
                    : false;
                
                if (maskValue) {
                    const idx = (y * width + x) * 4;
                    data[idx] = color.r;     // R
                    data[idx + 1] = color.g; // G
                    data[idx + 2] = color.b; // B
                    data[idx + 3] = color.a; // A
                }
            }
        }
        
        // Send back the processed data
        self.postMessage({
            taskId,
            imageData: { data: data, width: width, height: height },
            success: true
        });
        
    } catch (error) {
        self.postMessage({
            taskId,
            error: error.message,
            success: false
        });
    }
};
`;

/**
 * Process mask using Web Worker for large images
 */
export function processMaskWithWorker(
    maskData: boolean[][],
    width: number,
    height: number,
    color: { r: number; g: number; b: number; a: number }
): Promise<ImageData> {
    return new Promise((resolve, reject) => {
        const worker = new Worker(
            URL.createObjectURL(new Blob([maskWorkerScript], { type: 'application/javascript' }))
        );
        
        const taskId = Math.random().toString(36);
        
        worker.onmessage = (e) => {
            const { taskId: responseTaskId, imageData, success, error } = e.data;
            
            if (responseTaskId === taskId) {
                worker.terminate();
                
                if (success) {
                    resolve(new ImageData(
                        new Uint8ClampedArray(imageData.data),
                        imageData.width,
                        imageData.height
                    ));
                } else {
                    reject(new Error(error));
                }
            }
        };
        
        worker.onerror = (error) => {
            worker.terminate();
            reject(error);
        };
        
        worker.postMessage({ maskData, width, height, color, taskId });
    });
}

/**
 * Determine if mask should use worker processing based on size
 */
export function shouldUseWorkerProcessing(width: number, height: number): boolean {
    const pixelCount = width * height;
    return pixelCount > 5_000_000; // Use worker for > 5M pixels
}

/**
 * Estimate mask processing time for UI feedback
 */
export function estimateMaskProcessingTime(width: number, height: number): number {
    const pixelCount = width * height;
    // Rough estimate: ~1ms per 1k pixels
    return Math.max(100, pixelCount / 100_000);
} 