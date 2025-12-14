// Rendering optimization utilities for road network visualization
import { RoadNetworkData } from '../types';

export interface ViewportBounds {
    left: number;
    top: number;
    right: number;
    bottom: number;
}

export interface LODConfig {
    showPoints: boolean;
    showAllPoints: boolean;
    lineWidth: number;
    pointRadius: number;
    maxFeatures: number;
    simplificationTolerance: number;
}

export enum DetailLevel {
    MINIMAL = 0,    // Only major roads, no points
    LOW = 1,        // Lines only, simplified
    MEDIUM = 2,     // Lines + key points
    HIGH = 3,       // Lines + most points  
    FULL = 4        // All details
}

/**
 * Calculate current viewport bounds in image coordinates
 */
export function calculateViewportBounds(
    stageRef: React.RefObject<any>,
    imageWidth: number,
    imageHeight: number,
    padding: number = 0.1
): ViewportBounds | null {
    const stage = stageRef.current;
    if (!stage) return null;

    const scale = stage.scaleX();
    const position = stage.position();
    const stageWidth = stage.width();
    const stageHeight = stage.height();

    // Calculate visible area in image coordinates
    const left = (-position.x / scale) - (stageWidth * padding / scale);
    const top = (-position.y / scale) - (stageHeight * padding / scale);
    const right = (-position.x + stageWidth) / scale + (stageWidth * padding / scale);
    const bottom = (-position.y + stageHeight) / scale + (stageHeight * padding / scale);

    // Clamp to image bounds
    return {
        left: Math.max(0, left),
        top: Math.max(0, top),
        right: Math.min(imageWidth, right),
        bottom: Math.min(imageHeight, bottom)
    };
}

/**
 * Determine detail level based on zoom scale
 */
export function getDetailLevel(scale: number): DetailLevel {
    if (scale < 0.15) return DetailLevel.MINIMAL;
    if (scale < 0.30) return DetailLevel.LOW;
    if (scale < 0.50) return DetailLevel.MEDIUM;
    if (scale < 1.0) return DetailLevel.HIGH;
    return DetailLevel.FULL;
}

/**
 * Get LOD configuration based on detail level
 */
export function getLODConfig(detailLevel: DetailLevel, scale: number): LODConfig {
    const basePointRadius = 3;
    const baseLineWidth = 2;

    switch (detailLevel) {
        case DetailLevel.MINIMAL:
            return {
                showPoints: false,
                showAllPoints: false,
                lineWidth: Math.max(1, baseLineWidth / scale),
                pointRadius: 0,
                maxFeatures: 200,
                simplificationTolerance: 10.0 / scale
            };

        case DetailLevel.LOW:
            return {
                showPoints: false,
                showAllPoints: false,
                lineWidth: Math.max(1, baseLineWidth / scale),
                pointRadius: 0,
                maxFeatures: 400,
                simplificationTolerance: 5.0 / scale
            };

        case DetailLevel.MEDIUM:
            return {
                showPoints: true,
                showAllPoints: false,
                lineWidth: Math.max(1.5, baseLineWidth / scale),
                pointRadius: Math.max(2, basePointRadius / scale),
                maxFeatures: 600,
                simplificationTolerance: 2.0 / scale
            };

        case DetailLevel.HIGH:
            return {
                showPoints: true,
                showAllPoints: true,
                lineWidth: Math.max(1.5, baseLineWidth / scale),
                pointRadius: Math.max(2.5, basePointRadius / scale),
                maxFeatures: 1000,
                simplificationTolerance: 1.0 / scale
            };

        case DetailLevel.FULL:
        default:
            return {
                showPoints: true,
                showAllPoints: true,
                lineWidth: Math.max(2, baseLineWidth / scale),
                pointRadius: Math.max(3, basePointRadius / scale),
                maxFeatures: Infinity,
                simplificationTolerance: 0
            };
    }
}

/**
 * Check if a line string intersects with viewport bounds
 */
export function lineIntersectsViewport(
    coordinates: [number, number][],
    viewport: ViewportBounds
): boolean {
    if (coordinates.length === 0) return false;

    // Quick bounding box check first
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of coordinates) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }

    // If bounding box doesn't intersect viewport, line doesn't intersect
    if (maxX < viewport.left || minX > viewport.right || 
        maxY < viewport.top || minY > viewport.bottom) {
        return false;
    }

    return true;
}

/**
 * Simplify line coordinates based on tolerance (Douglas-Peucker-like)
 */
export function simplifyLineCoordinates(
    coordinates: [number, number][],
    tolerance: number
): [number, number][] {
    if (tolerance <= 0 || coordinates.length <= 2) {
        return coordinates;
    }

    // Simple distance-based simplification
    const simplified: [number, number][] = [coordinates[0]];
    let lastPoint = coordinates[0];

    for (let i = 1; i < coordinates.length - 1; i++) {
        const currentPoint = coordinates[i];
        const distance = Math.sqrt(
            Math.pow(currentPoint[0] - lastPoint[0], 2) + 
            Math.pow(currentPoint[1] - lastPoint[1], 2)
        );

        if (distance >= tolerance) {
            simplified.push(currentPoint);
            lastPoint = currentPoint;
        }
    }

    // Always include the last point
    if (coordinates.length > 1) {
        simplified.push(coordinates[coordinates.length - 1]);
    }

    return simplified;
}

/**
 * Filter road network features based on viewport and LOD
 */
export function filterRoadNetworkFeatures(
    roadNetwork: RoadNetworkData,
    viewport: ViewportBounds | null,
    lodConfig: LODConfig
): RoadNetworkData {
    if (!roadNetwork || !roadNetwork.features) {
        return roadNetwork;
    }

    let filteredFeatures = roadNetwork.features;

    // Apply viewport culling
    if (viewport) {
        filteredFeatures = filteredFeatures.filter(feature => {
            if (feature.geometry.type === 'LineString') {
                const coordinates = feature.geometry.coordinates as [number, number][];
                return lineIntersectsViewport(coordinates, viewport);
            }
            return true;
        });
    }

    // Apply feature limit for performance
    if (filteredFeatures.length > lodConfig.maxFeatures) {
        filteredFeatures = filteredFeatures.slice(0, lodConfig.maxFeatures);
    }

    // Apply line simplification
    if (lodConfig.simplificationTolerance > 0) {
        filteredFeatures = filteredFeatures.map(feature => {
            if (feature.geometry.type === 'LineString') {
                const coordinates = feature.geometry.coordinates as [number, number][];
                const simplified = simplifyLineCoordinates(coordinates, lodConfig.simplificationTolerance);
                return {
                    ...feature,
                    geometry: {
                        ...feature.geometry,
                        coordinates: simplified
                    }
                };
            }
            return feature;
        });
    }

    return {
        ...roadNetwork,
        features: filteredFeatures
    };
}

/**
 * Debounce function for performance optimization
 */
export function debounce<T extends (...args: any[]) => void>(
    func: T,
    wait: number
): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

/**
 * Throttle function for performance optimization
 */
export function throttle<T extends (...args: any[]) => void>(
    func: T,
    limit: number
): (...args: Parameters<T>) => void {
    let inThrottle: boolean;
    return (...args: Parameters<T>) => {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
} 