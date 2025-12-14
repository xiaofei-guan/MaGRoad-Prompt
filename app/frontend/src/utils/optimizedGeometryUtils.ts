// Optimized geometry utilities for high-performance road network operations
import RBush from 'rbush';
import { RoadNode, RoadEdge } from '../types';

export interface Point {
    x: number;
    y: number;
}

// Enhanced types for spatial indexing
interface SpatialNode extends Point {
    nodeId: number;
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
}

interface SpatialEdge {
    edgeId: number;
    startNode: RoadNode;
    endNode: RoadNode;
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
}

/**
 * Calculate distance between two points
 */
export function pointDistance(p1: Point, p2: Point): number {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate distance from a point to a line segment
 */
export function pointToLineDistance(
    px: number, py: number,
    x1: number, y1: number,
    x2: number, y2: number
): number {
    const A = px - x1;
    const B = py - y1;
    const C = x2 - x1;
    const D = y2 - y1;
    
    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    
    if (lenSq === 0) return Math.sqrt(A * A + B * B);
    
    let param = dot / lenSq;
    param = Math.max(0, Math.min(1, param));
    
    const xx = x1 + param * C;
    const yy = y1 + param * D;
    
    const dx = px - xx;
    const dy = py - yy;
    
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Normalize rectangle coordinates to ensure proper bounds
 */
export function normalizeRectangle(start: Point, end: Point): { 
    topLeft: Point; 
    bottomRight: Point; 
    width: number; 
    height: number 
} {
    const minX = Math.min(start.x, end.x);
    const maxX = Math.max(start.x, end.x);
    const minY = Math.min(start.y, end.y);
    const maxY = Math.max(start.y, end.y);
    
    return {
        topLeft: { x: minX, y: minY },
        bottomRight: { x: maxX, y: maxY },
        width: maxX - minX,
        height: maxY - minY
    };
}

/**
 * Check if a point is inside a rectangle defined by two corner points
 */
export function isPointInRectangle(
    point: Point,
    rectStart: Point,
    rectEnd: Point
): boolean {
    const minX = Math.min(rectStart.x, rectEnd.x);
    const maxX = Math.max(rectStart.x, rectEnd.x);
    const minY = Math.min(rectStart.y, rectEnd.y);
    const maxY = Math.max(rectStart.y, rectEnd.y);
    
    return point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY;
}

/**
 * High-performance geometry operations manager using spatial indexing
 */
export class OptimizedGeometryManager {
    private nodeIndex: RBush<SpatialNode>;
    private edgeIndex: RBush<SpatialEdge>;
    private adjacencyList: Map<number, Set<number>>;
    private edgeConnections: Map<number, Set<number>>;
    private nodeDegrees: Map<number, number>;
    private isInitialized = false;

    constructor() {
        this.nodeIndex = new RBush<SpatialNode>();
        this.edgeIndex = new RBush<SpatialEdge>();
        this.adjacencyList = new Map();
        this.edgeConnections = new Map();
        this.nodeDegrees = new Map();
    }

    /**
     * Initialize or update the spatial indexes and adjacency data
     */
    initialize(nodes: RoadNode[], edges: RoadEdge[]): void {
        // Clear existing data
        this.nodeIndex.clear();
        this.edgeIndex.clear();
        this.adjacencyList.clear();
        this.edgeConnections.clear();
        this.nodeDegrees.clear();

        // Build spatial index for nodes
        const spatialNodes: SpatialNode[] = nodes.map(node => ({
            ...node,
            nodeId: node.id,
            minX: node.x,
            minY: node.y,
            maxX: node.x,
            maxY: node.y
        }));
        
        this.nodeIndex.load(spatialNodes);

        // Build spatial index for edges and adjacency data
        const spatialEdges: SpatialEdge[] = [];
        
        for (const edge of edges) {
            const startNode = nodes.find(n => n.id === edge.source);
            const endNode = nodes.find(n => n.id === edge.target);
            
            if (startNode && endNode) {
                // Add to spatial index
                spatialEdges.push({
                    edgeId: edge.id,
                    startNode,
                    endNode,
                    minX: Math.min(startNode.x, endNode.x),
                    minY: Math.min(startNode.y, endNode.y),
                    maxX: Math.max(startNode.x, endNode.x),
                    maxY: Math.max(startNode.y, endNode.y)
                });

                // Build adjacency list
                if (!this.adjacencyList.has(edge.source)) {
                    this.adjacencyList.set(edge.source, new Set());
                }
                if (!this.adjacencyList.has(edge.target)) {
                    this.adjacencyList.set(edge.target, new Set());
                }
                
                this.adjacencyList.get(edge.source)!.add(edge.target);
                this.adjacencyList.get(edge.target)!.add(edge.source);

                // Build edge connections
                if (!this.edgeConnections.has(edge.source)) {
                    this.edgeConnections.set(edge.source, new Set());
                }
                if (!this.edgeConnections.has(edge.target)) {
                    this.edgeConnections.set(edge.target, new Set());
                }
                
                this.edgeConnections.get(edge.source)!.add(edge.id);
                this.edgeConnections.get(edge.target)!.add(edge.id);
            }
        }

        this.edgeIndex.load(spatialEdges);

        // Precompute node degrees
        for (const [nodeId, neighbors] of this.adjacencyList) {
            this.nodeDegrees.set(nodeId, neighbors.size);
        }

        this.isInitialized = true;
    }

    /**
     * Fast nearest node search using spatial indexing
     */
    findNearestNode(point: Point, threshold: number = 15, excludeNodeId?: number): { node: RoadNode; distance: number } | null {
        if (!this.isInitialized) return null;

        const searchBounds = {
            minX: point.x - threshold,
            minY: point.y - threshold,
            maxX: point.x + threshold,
            maxY: point.y + threshold
        };

        const candidates = this.nodeIndex.search(searchBounds);
        
        let nearestNode = null;
        let minDistance = threshold;

        for (const candidate of candidates) {
            if (excludeNodeId && candidate.nodeId === excludeNodeId) continue;
            
            const distance = pointDistance(point, candidate);
            if (distance < minDistance) {
                minDistance = distance;
                nearestNode = { 
                    node: { id: candidate.nodeId, x: candidate.x, y: candidate.y },
                    distance 
                };
            }
        }

        return nearestNode;
    }

    /**
     * Fast nearest edge search using spatial indexing
     */
    findNearestEdge(point: Point, threshold: number = 10): { edge: RoadEdge; distance: number } | null {
        if (!this.isInitialized) return null;

        const searchBounds = {
            minX: point.x - threshold,
            minY: point.y - threshold,
            maxX: point.x + threshold,
            maxY: point.y + threshold
        };

        const candidates = this.edgeIndex.search(searchBounds);
        
        let nearestEdge = null;
        let minDistance = threshold;

        for (const candidate of candidates) {
            const distance = pointToLineDistance(
                point.x, point.y,
                candidate.startNode.x, candidate.startNode.y,
                candidate.endNode.x, candidate.endNode.y
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestEdge = { 
                    edge: { 
                        id: candidate.edgeId,
                        source: candidate.startNode.id,
                        target: candidate.endNode.id
                    },
                    distance 
                };
            }
        }

        return nearestEdge;
    }

    /**
     * Fast neighbor nodes lookup using adjacency list
     */
    getNeighborNodes(nodeId: number): number[] {
        const neighbors = this.adjacencyList.get(nodeId);
        return neighbors ? Array.from(neighbors) : [];
    }

    /**
     * Fast node degree lookup using precomputed degrees
     */
    getNodeDegree(nodeId: number): number {
        return this.nodeDegrees.get(nodeId) || 0;
    }

    /**
     * Fast connected edges lookup using edge connections
     */
    findConnectedEdges(nodeId: number): number[] {
        const connectedEdges = this.edgeConnections.get(nodeId);
        return connectedEdges ? Array.from(connectedEdges) : [];
    }

    /**
     * Fast nodes in rectangle search using spatial indexing
     */
    getNodesInRectangle(rectStart: Point, rectEnd: Point): RoadNode[] {
        if (!this.isInitialized) return [];

        const bounds = normalizeRectangle(rectStart, rectEnd);
        const searchBounds = {
            minX: bounds.topLeft.x,
            minY: bounds.topLeft.y,
            maxX: bounds.bottomRight.x,
            maxY: bounds.bottomRight.y
        };

        const candidates = this.nodeIndex.search(searchBounds);
        
        return candidates.map((candidate: SpatialNode) => ({
            id: candidate.nodeId,
            x: candidate.x,
            y: candidate.y
        }));
    }

    /**
     * Fast nodes at position search using spatial indexing
     */
    findNodesAtPosition(position: Point, tolerance: number = 0.1): RoadNode[] {
        if (!this.isInitialized) return [];

        const searchBounds = {
            minX: position.x - tolerance,
            minY: position.y - tolerance,
            maxX: position.x + tolerance,
            maxY: position.y + tolerance
        };

        const candidates = this.nodeIndex.search(searchBounds);
        
        return candidates
            .filter((candidate: SpatialNode) => pointDistance(position, candidate) <= tolerance)
            .map((candidate: SpatialNode) => ({
                id: candidate.nodeId,
                x: candidate.x,
                y: candidate.y
            }));
    }

    /**
     * Efficient snap node detection with spatial indexing
     */
    getNearestSnapNode(point: Point, snapDistance: number, excludeNodeId?: number): RoadNode | null {
        const result = this.findNearestNode(point, snapDistance, excludeNodeId);
        return result?.node || null;
    }

    /**
     * Optimized node merging with adjacency list updates
     */
    mergeNodes(
        targetNodeId: number,
        nodesToMergeIds: number[],
        nodes: RoadNode[],
        edges: RoadEdge[]
    ): { updatedNodes: RoadNode[]; updatedEdges: RoadEdge[] } {
        if (nodesToMergeIds.length === 0) {
            return { updatedNodes: nodes, updatedEdges: edges };
        }

        const mergeIdsSet = new Set(nodesToMergeIds);
        
        // Filter out nodes to be merged (except target)
        const updatedNodes = nodes.filter(node => 
            node.id === targetNodeId || !mergeIdsSet.has(node.id)
        );

        // Update edges efficiently
        const updatedEdges: RoadEdge[] = [];
        const seenEdgeKeys = new Set<string>();

        for (const edge of edges) {
            let sourceId = edge.source;
            let targetId = edge.target;

            // Redirect merged nodes to target node
            if (mergeIdsSet.has(sourceId)) sourceId = targetNodeId;
            if (mergeIdsSet.has(targetId)) targetId = targetNodeId;

            // Skip self-loops
            if (sourceId === targetId) continue;

            // Create normalized edge key to avoid duplicates
            const edgeKey = sourceId < targetId 
                ? `${sourceId}-${targetId}` 
                : `${targetId}-${sourceId}`;

            if (!seenEdgeKeys.has(edgeKey)) {
                seenEdgeKeys.add(edgeKey);
                updatedEdges.push({
                    ...edge,
                    source: sourceId,
                    target: targetId
                });
            }
        }

        return { updatedNodes, updatedEdges };
    }
}

// Singleton instance for global use
export const geometryManager = new OptimizedGeometryManager(); 