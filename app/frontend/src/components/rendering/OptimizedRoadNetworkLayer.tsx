import React, { useMemo, useCallback, useRef, useEffect, useState } from 'react';
import { Line as KonvaLine, Circle } from 'react-konva';
import { RoadNetworkData, RoadNode, RoadEdge } from '../../types';
import { 
    calculateViewportBounds, 
    getDetailLevel, 
    getLODConfig,
    DetailLevel,
    ViewportBounds,
    LODConfig,
    debounce
} from '../../utils/renderingOptimization';
import { geometryManager } from '../../utils/optimizedGeometryUtils';

interface OptimizedRoadNetworkLayerProps {
    roadNetwork: RoadNetworkData | null;
    stageRef: React.RefObject<any>;
    stageScale: number;
    stagePos: { x: number; y: number };
    interactionMode: string;
    isInteracting: boolean;
    imageWidth: number;
    imageHeight: number;
    visible: boolean;
}

interface RenderCache {
    viewport: ViewportBounds | null;
    detailLevel: DetailLevel;
    filteredNodes: RoadNode[] | null;
    filteredEdges: RoadEdge[] | null;
    lodConfig: LODConfig;
    renderedElements: React.ReactElement[];
    nodeDataHash: string;
    edgeDataHash: string;
    timestamp: number;
}

// Helper function to generate data hash for cache invalidation
const generateDataHash = (nodes: RoadNode[], edges: RoadEdge[]): { nodeHash: string, edgeHash: string } => {
    const nodeHash = nodes.length > 0 ? `${nodes.length}-${nodes[0]?.id}-${nodes[nodes.length - 1]?.id}` : 'empty';
    const edgeHash = edges.length > 0 ? `${edges.length}-${edges[0]?.id}-${edges[edges.length - 1]?.id}` : 'empty';
    return { nodeHash, edgeHash };
};

// Helper function to filter nodes and edges based on viewport
const filterNodesAndEdges = (
    nodes: RoadNode[], 
    edges: RoadEdge[], 
    viewport: ViewportBounds | null
): { filteredNodes: RoadNode[], filteredEdges: RoadEdge[] } => {
    if (!viewport) {
        return { filteredNodes: nodes, filteredEdges: edges };
    }

    // Filter nodes within viewport with buffer
    const filteredNodes = nodes.filter(node => 
        node.x >= viewport.left && 
        node.x <= viewport.right &&
        node.y >= viewport.top && 
        node.y <= viewport.bottom
    );

    // Create a set of visible node IDs for efficient lookup
    const visibleNodeIds = new Set(filteredNodes.map(node => node.id));

    // Filter edges that have at least one visible node
    const filteredEdges = edges.filter(edge => 
        visibleNodeIds.has(edge.source) || visibleNodeIds.has(edge.target)
    );

    return { filteredNodes, filteredEdges };
};

// Deduplicate edges by their endpoints to avoid duplicate renders and key collisions
function deduplicateEdgesByEndpoints(edges: RoadEdge[]): RoadEdge[] {
    const seen = new Set<string>();
    const unique: RoadEdge[] = [];
    for (const edge of edges) {
        const a = Math.min(edge.source, edge.target);
        const b = Math.max(edge.source, edge.target);
        const key = `${a}-${b}`;
        if (!seen.has(key)) {
            seen.add(key);
            unique.push(edge);
        }
    }
    return unique;
}

const OptimizedRoadNetworkLayer: React.FC<OptimizedRoadNetworkLayerProps> = ({
    roadNetwork,
    stageRef,
    stageScale,
    interactionMode,
    isInteracting,
    imageWidth,
    imageHeight,
    visible
}) => {
    const renderCacheRef = useRef<RenderCache | null>(null);
    const [forceUpdate, setForceUpdate] = useState(0);

    // Force cache invalidation when roadNetwork changes
    useEffect(() => {
        if (roadNetwork) {
            // Clear cache when road network data changes
            renderCacheRef.current = null;
            setForceUpdate(prev => prev + 1);
            
            // Initialize geometry manager for all road networks
            if (roadNetwork.nodes && roadNetwork.edges) {
                geometryManager.initialize(roadNetwork.nodes, roadNetwork.edges);
            }
        }
    }, [roadNetwork?.nodes, roadNetwork?.edges]);

    // Debounced function to trigger re-render after interaction stops
    const debouncedRender = useCallback(
        debounce(() => {
            setForceUpdate(prev => prev + 1);
        }, 150),
        []
    );

    // Effect to trigger re-render when interaction stops
    useEffect(() => {
        if (!isInteracting) {
            debouncedRender();
        }
    }, [isInteracting, debouncedRender]);

    // Calculate current viewport and LOD settings
    const { viewport, detailLevel, lodConfig } = useMemo(() => {
        const viewport = calculateViewportBounds(stageRef, imageWidth, imageHeight, 0.2);
        const detailLevel = isInteracting ? DetailLevel.LOW : getDetailLevel(stageScale);
        const lodConfig = getLODConfig(detailLevel, stageScale);
        
        return { viewport, detailLevel, lodConfig };
    }, [stageRef, imageWidth, imageHeight, stageScale, isInteracting, forceUpdate]);

    // Filter road network data based on format and viewport
    const filteredData = useMemo(() => {
        if (!roadNetwork || !visible) {
            return null;
        }
        
        // Handle missing nodes/edges gracefully
        if (!roadNetwork.nodes || !roadNetwork.edges) {
            console.warn('Road network missing nodes or edges, skipping render');
            return null;
        }
        
        // Generate current data hash for cache validation
        const { nodeHash, edgeHash } = generateDataHash(roadNetwork.nodes, roadNetwork.edges);
        const currentTimestamp = Date.now();
        
        // Check if we can use cached result - now with data hash validation
        const cache = renderCacheRef.current;
        if (cache && 
            cache.nodeDataHash === nodeHash &&
            cache.edgeDataHash === edgeHash &&
            cache.viewport?.left === viewport?.left &&
            cache.viewport?.top === viewport?.top &&
            cache.viewport?.right === viewport?.right &&
            cache.viewport?.bottom === viewport?.bottom &&
            cache.detailLevel === detailLevel &&
            cache.filteredNodes && 
            cache.filteredEdges &&
            (currentTimestamp - cache.timestamp) < 5000) { // Cache expires after 5 seconds
            return {
                nodes: cache.filteredNodes,
                edges: cache.filteredEdges
            };
        }

        // Use node-edge format
    const filtered = filterNodesAndEdges(roadNetwork.nodes, roadNetwork.edges, viewport);
        // Remove potential duplicate edges that share the same endpoints
        const dedupedEdges = deduplicateEdgesByEndpoints(filtered.filteredEdges);
        
        // Update cache with data hashes
        renderCacheRef.current = {
            viewport,
            detailLevel,
            filteredNodes: filtered.filteredNodes,
            filteredEdges: dedupedEdges,
            lodConfig,
            renderedElements: [],
            nodeDataHash: nodeHash,
            edgeDataHash: edgeHash,
            timestamp: currentTimestamp
        };

        return {
            nodes: filtered.filteredNodes,
            edges: dedupedEdges
        };
    }, [roadNetwork?.nodes, roadNetwork?.edges, viewport, detailLevel, lodConfig, visible]);

    // Generate rendering elements with enhanced caching and validation
    const renderedElements = useMemo(() => {
        if (!filteredData || !visible || !filteredData.nodes || !filteredData.edges) {
            return [];
        }

        // Check if we can use cached rendered elements - with stricter validation
        const cache = renderCacheRef.current;
        if (cache && 
            cache.filteredNodes === filteredData.nodes && 
            cache.filteredEdges === filteredData.edges &&
            cache.lodConfig.lineWidth === lodConfig.lineWidth &&
            cache.lodConfig.pointRadius === lodConfig.pointRadius &&
            cache.lodConfig.showPoints === lodConfig.showPoints &&
            cache.renderedElements.length > 0 &&
            !isInteracting) { // Don't use cache during interaction for immediate feedback
            return cache.renderedElements;
        }

        const elements: React.ReactElement[] = [];
        
        try {
            // Render node-edge format
            const nodeMap = new Map(filteredData.nodes.map(node => [node.id, node]));
            
            // Render edges first for better layering
            if (filteredData.edges.length > 0) {
                filteredData.edges.forEach((edge, edgeIndex) => {
                    const sourceNode = nodeMap.get(edge.source);
                    const targetNode = nodeMap.get(edge.target);
                    
                    if (sourceNode && targetNode) {
                        elements.push(
                            <KonvaLine
                                key={`road-edge-${edge.id}-${edge.source}-${edge.target}-${edgeIndex}`}
                                points={[sourceNode.x, sourceNode.y, targetNode.x, targetNode.y]}
                                stroke="#4c88d6"
                                strokeWidth={lodConfig.lineWidth}
                                lineCap="round"
                                lineJoin="round"
                                opacity={isInteracting ? 0.7 : 0.95}
                                listening={interactionMode === 'edit' && !isInteracting}
                                perfectDrawEnabled={false}
                                shadowForStrokeEnabled={false}
                            />
                        );
                    }
                });
            }

            // Render nodes if enabled by LOD
            if (lodConfig.showPoints && lodConfig.pointRadius > 0 && filteredData.nodes.length > 0) {
                filteredData.nodes.forEach((node) => {
                    elements.push(
                        <Circle
                            key={`road-node-${node.id}`}
                            x={node.x}
                            y={node.y}
                            radius={lodConfig.pointRadius}
                            fill="#eedd5e"
                            stroke="black"
                            strokeWidth={Math.max(0.5, 0.5 / stageScale)}
                            opacity={isInteracting ? 0.5 : 1.0}
                            listening={interactionMode === 'edit' && !isInteracting}
                            perfectDrawEnabled={false}
                            shadowForStrokeEnabled={false}
                        />
                    );
                });
            }
        } catch (error) {
            console.error('Error rendering road network elements:', error);
            return [];
        }

        // Update cache with rendered elements only if not interacting
        if (cache && !isInteracting) {
            cache.renderedElements = elements;
        }

        return elements;
    }, [filteredData, lodConfig, isInteracting, interactionMode, stageScale, visible]);

    if (!visible || !renderedElements || renderedElements.length === 0) {
        return null;
    }

    return <>{renderedElements}</>;
};

export default React.memo(OptimizedRoadNetworkLayer); 