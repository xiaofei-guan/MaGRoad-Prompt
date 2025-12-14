// Geometry utility functions for road network editing
import { RoadNode, RoadEdge } from '../types';
import { 
    OptimizedGeometryManager, 
    geometryManager,
    Point,
    pointDistance,
    pointToLineDistance,
    normalizeRectangle,
    isPointInRectangle
} from './optimizedGeometryUtils';

// Re-export types and utility functions from optimizedGeometryUtils
export type { Point };
export { 
    pointDistance, 
    pointToLineDistance, 
    normalizeRectangle, 
    isPointInRectangle,
    OptimizedGeometryManager
};

// Helper function to ensure optimized manager is initialized
function ensureOptimizedManager(nodes: RoadNode[], edges: RoadEdge[]): void {
    geometryManager.initialize(nodes, edges);
}

/**
 * Calculate midpoint of an edge (unique function)
 */
export function getEdgeMidpoint(
    edge: RoadEdge, 
    nodes: RoadNode[]
): Point | null {
    // Use more efficient lookup instead of find for better performance
    let startNode: RoadNode | undefined;
    let endNode: RoadNode | undefined;
    
    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        if (node.id === edge.source) {
            startNode = node;
            if (endNode) break; // Early break if both found
        } else if (node.id === edge.target) {
            endNode = node;
            if (startNode) break; // Early break if both found
        }
    }
    
    if (!startNode || !endNode) return null;
    
    return {
        x: (startNode.x + endNode.x) / 2,
        y: (startNode.y + endNode.y) / 2
    };
}

/**
 * Generate a unique ID for new nodes/edges (unique function)
 */
export function generateUniqueId(): number {
    return Date.now() + Math.floor(Math.random() * 1000);
}

/**
 * Convert polyline points to nodes and edges (unique function)
 */
export function polylineToNodesAndEdges(
    points: Point[]
): { nodes: RoadNode[]; edges: RoadEdge[] } {
    if (points.length < 2) {
        return { nodes: [], edges: [] };
    }
    
    const nodes: RoadNode[] = points.map((point) => ({
        id: generateUniqueId(),
        x: point.x,
        y: point.y
    }));
    
    const edges: RoadEdge[] = [];
    for (let i = 0; i < nodes.length - 1; i++) {
        edges.push({
            id: generateUniqueId(),
            source: nodes[i].id,
            target: nodes[i + 1].id
        });
    }
    
    return { nodes, edges };
}

/**
 * Apply snapping to a point based on existing nodes (unique function)
 */
export function applySnapping(
    point: Point, 
    nodes: RoadNode[], 
    snapDistance: number,
    enabled: boolean = true
): Point {
    if (!enabled || nodes.length === 0) {
        return point;
    }
    
    let closestNode: RoadNode | null = null;
    let minDistance = snapDistance;
    
    // Find the closest node within snap distance
    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const distance = pointDistance(point, { x: node.x, y: node.y });
        
        if (distance < minDistance) {
            minDistance = distance;
            closestNode = node;
        }
    }
    
    // If we found a node to snap to, return its position
    if (closestNode) {
        return { x: closestNode.x, y: closestNode.y };
    }
    
    return point;
}

// Direct exports from optimizedGeometryUtils with initialization
export function findNearestEdge(
    mouseX: number, 
    mouseY: number, 
    edges: RoadEdge[], 
    nodes: RoadNode[], 
    threshold: number = 10
): { edge: RoadEdge; distance: number } | null {
    ensureOptimizedManager(nodes, edges);
    return geometryManager.findNearestEdge({ x: mouseX, y: mouseY }, threshold);
}

export function findNearestNode(
    mouseX: number, 
    mouseY: number, 
    nodes: RoadNode[], 
    threshold: number = 15
): { node: RoadNode; distance: number } | null {
    ensureOptimizedManager(nodes, []);
    return geometryManager.findNearestNode({ x: mouseX, y: mouseY }, threshold);
}

export function findConnectedEdges(nodeId: number, edges: RoadEdge[], nodes?: RoadNode[]): RoadEdge[] {
    if (nodes) {
        ensureOptimizedManager(nodes, edges);
        const connectedEdgeIds = geometryManager.findConnectedEdges(nodeId);
        return edges.filter(edge => connectedEdgeIds.includes(edge.id));
    }
    
    // Fallback for cases without nodes array
    return edges.filter(edge => edge.source === nodeId || edge.target === nodeId);
}

export function getNeighborNodes(nodeId: number, edges: RoadEdge[], nodes?: RoadNode[]): number[] {
    if (nodes) {
        ensureOptimizedManager(nodes, edges);
        return geometryManager.getNeighborNodes(nodeId);
    }
    
    // Fallback for cases without nodes array
    const neighbors: number[] = [];
    for (let i = 0; i < edges.length; i++) {
        const edge = edges[i];
        if (edge.source === nodeId) {
            neighbors.push(edge.target);
        } else if (edge.target === nodeId) {
            neighbors.push(edge.source);
        }
    }
    return neighbors;
}

export function getNodeDegree(nodeId: number, edges: RoadEdge[], nodes?: RoadNode[]): number {
    if (nodes) {
        ensureOptimizedManager(nodes, edges);
        return geometryManager.getNodeDegree(nodeId);
    }
    
    // Fallback for cases without nodes array
    const connectedNodes = new Set<number>();
    for (let i = 0; i < edges.length; i++) {
        const edge = edges[i];
        if (edge.source === nodeId) {
            connectedNodes.add(edge.target);
        } else if (edge.target === nodeId) {
            connectedNodes.add(edge.source);
        }
    }
    return connectedNodes.size;
}

export function getNearestSnapNode(
    point: Point,
    nodes: RoadNode[],
    snapDistance: number,
    excludeNodeId?: number
): RoadNode | null {
    ensureOptimizedManager(nodes, []);
    return geometryManager.getNearestSnapNode(point, snapDistance, excludeNodeId);
}

export function mergeNodes(
    targetNodeId: number,
    nodesToMergeIds: number[],
    nodes: RoadNode[],
    edges: RoadEdge[]
): { updatedNodes: RoadNode[]; updatedEdges: RoadEdge[] } {
    ensureOptimizedManager(nodes, edges);
    return geometryManager.mergeNodes(targetNodeId, nodesToMergeIds, nodes, edges);
}

export function findNodesAtPosition(
    position: Point,
    nodes: RoadNode[],
    tolerance: number = 0.1
): RoadNode[] {
    ensureOptimizedManager(nodes, []);
    return geometryManager.findNodesAtPosition(position, tolerance);
}

export function getNodesInRectangle(
    nodes: RoadNode[],
    rectStart: Point,
    rectEnd: Point
): RoadNode[] {
    ensureOptimizedManager(nodes, []);
    return geometryManager.getNodesInRectangle(rectStart, rectEnd);
} 