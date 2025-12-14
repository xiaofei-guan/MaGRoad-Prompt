import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { roadNetworkApi, saveRoadNetworkAPI } from '../services/api';
import { Prompt, RoadNetworkData, SaveRoadNetworkResponse, RoadNode, RoadEdge } from '../types';
import { usePromptStore } from './promptStore';
import { useImageStore } from './imageStore';
import { ensurePromptIds } from '../utils/promptUtils';
import { processMaskData, MaskData } from '../utils/maskUtils';
import { validateRoadNetwork } from '../utils/roadNetworkUtils';
import { 
    Point, 
    polylineToNodesAndEdges, 
    findConnectedEdges, 
    getNeighborNodes, 
    getNodeDegree, 
    generateUniqueId,
    getNearestSnapNode,
    mergeNodes,
    findNodesAtPosition,
    getNodesInRectangle,
} from '../utils/geometryUtils';

// Edit mode types
export type EditMode = 'addEdge' | 'modify' | null;
export type EditTool = 'addEdge' | 'modify' | null;

// Edit state interface
interface EditState {
    mode: EditMode;
    tool: EditTool;
    // AddEdge mode state
    isDrawing: boolean;
    currentPolyline: Point[];
    previewPoint: Point | null;
    // Modify mode state
    hoveredEdge: number | null;
    hoveredNode: number | null;
    selectedNode: number | null;
    isDraggingNode: boolean;
    // Rectangle selection state
    selectedNodes: number[];
    isRectangleSelecting: boolean;
    rectangleSelection: { start: Point; end: Point } | null;
}

// Define the structure of the roadNetwork state
interface RoadNetworkState {
    // The generated road network data (GeoJSON)
    roadNetwork: RoadNetworkData | null;
    // Prompts associated with the current network (loaded or generated)
    currentNetworkPrompts: Prompt[] | null;
    // Loading state for generation operations
    isLoading: boolean;
    // Error message if generation fails
    error: string | null;
    // Flag to indicate if network has unsaved changes
    isDirty: boolean;
    // Flag to track save success for notifications
    saveSuccess: boolean;
    // Last save response from the API
    lastSaveResponse: SaveRoadNetworkResponse | null;
    // Flag for feature computation during generation
    isComputingFeatures: boolean;
    // Road mask data - now supports both compressed and uncompressed
    roadMask: boolean[][] | null;
    // Keypoint mask data - now supports both compressed and uncompressed
    keypointMask: boolean[][] | null;
    // Flag to indicate saving in progress
    isSaving: boolean;
    // Flag to indicate mask processing in progress
    isMaskProcessing: boolean;
    // Raw mask data from API (before processing)
    rawMaskData: {
        roadMask?: MaskData;
        keypointMask?: MaskData;
    };
    
    // Edit state
    editState: EditState;
    
    // Snapping settings
    snappingEnabled: boolean;
    snappingDistance: number;
    
    // Undo/Redo system
    undoStack: RoadNetworkData[];
    redoStack: RoadNetworkData[];
    maxUndoSteps: number;
    
    // Action to set the road network data directly
    setRoadNetwork: (data: RoadNetworkData | null) => void;
    // Action to generate a road network with prompts
    generateRoadNetwork: (imageId: string, promptsForGeneration: Prompt[]) => Promise<void>;
    // Action to clear road network data
    clearRoadNetwork: () => void;
    // Action to save the current road network
    saveRoadNetwork: (imageId: string) => Promise<void>;
    // Action to load a saved road network
    loadSavedRoadNetwork: (imageId: string, abortSignal?: AbortSignal) => Promise<void>;
    // Action to clear current network and prompts
    clearCurrentRoadNetwork: () => void;
    // Function to reset save success state
    resetSaveSuccess: () => void;
    // Function to mark the network as having unsaved changes
    setNetworkAsDirty: () => void;
    // Function to process raw mask data asynchronously
    processMasks: () => Promise<void>;
    
    // Edit mode actions
    setEditMode: (mode: EditMode) => void;
    setEditTool: (tool: EditTool) => void;
    
    // AddEdge mode actions
    startDrawing: () => void;
    addPointToPolyline: (point: Point) => void;
    setPreviewPoint: (point: Point | null) => void;
    finishPolyline: () => void;
    cancelDrawing: () => void;
    
    // Modify mode actions
    setHoveredEdge: (edgeId: number | null) => void;
    setHoveredNode: (nodeId: number | null) => void;
    setSelectedNode: (nodeId: number | null) => void;
    addNodeAtEdgeMidpoint: (edgeId: number, x: number, y: number) => void;
    moveNode: (nodeId: number, x: number, y: number, recordUndo?: boolean) => void;
    moveNodeWithSnapping: (nodeId: number, x: number, y: number) => void;
    deleteNodeWithLogic: (nodeId: number) => void;
    setIsDraggingNode: (isDragging: boolean) => void;
    
    // Rectangle selection actions
    setSelectedNodes: (nodeIds: number[]) => void;
    addToSelectedNodes: (nodeId: number) => void;
    removeFromSelectedNodes: (nodeId: number) => void;
    clearSelectedNodes: () => void;
    setIsRectangleSelecting: (isSelecting: boolean) => void;
    setRectangleSelection: (selection: { start: Point; end: Point } | null) => void;
    updateRectangleSelection: (start: Point, current: Point) => void;
    selectNodesInRectangle: (start: Point, end: Point) => void;
    deleteSelectedNodes: () => void;
    
    // Snapping actions
    setSnappingEnabled: (enabled: boolean) => void;
    setSnappingDistance: (distance: number) => void;
    
    // Undo/Redo actions
    pushToUndoStack: () => void;
    undo: () => void;
    redo: () => void;
    canUndo: () => boolean;
    canRedo: () => boolean;
}

// Create the store with initial values and actions
export const useRoadNetworkStore = create<RoadNetworkState>()(
    subscribeWithSelector((set, get) => ({
    roadNetwork: null,
    currentNetworkPrompts: null,
    isLoading: false,
    error: null,
    isDirty: false,
    saveSuccess: false,
    lastSaveResponse: null,
    isComputingFeatures: false,
    roadMask: null,
    keypointMask: null,
    isSaving: false,
    isMaskProcessing: false,
    rawMaskData: {},
    
    // Initialize edit state
    editState: {
        mode: null,
        tool: null,
        isDrawing: false,
        currentPolyline: [],
        previewPoint: null,
        hoveredEdge: null,
        hoveredNode: null,
        selectedNode: null,
        isDraggingNode: false,
        selectedNodes: [],
        isRectangleSelecting: false,
        rectangleSelection: null,
    },
    
    // Initialize snapping settings
    snappingEnabled: false,
    snappingDistance: 10,
    
    // Initialize undo/redo system
    undoStack: [],
    redoStack: [],
    maxUndoSteps: 20,

    resetSaveSuccess: () => set({ saveSuccess: false }),

    setRoadNetwork: (data) => {
        if (data) {
            // Simplified validation for new node-edge format only
            const validation = validateRoadNetwork(data);
            if (!validation.isValid) {
                console.error('Invalid road network data:', validation.errors);
                set({ roadNetwork: null, error: `Invalid road network: ${validation.errors.join(', ')}` });
                return;
            }
            
            if (validation.warnings.length > 0) {
                console.warn('Road network warnings:', validation.warnings);
            }
        }
        
        set({ roadNetwork: data, error: null });
    },

    clearRoadNetwork: () => {
        console.log('Clearing road network data');
        set({ 
            roadNetwork: null, 
            isLoading: false, 
            error: null, 
            lastSaveResponse: null, 
            saveSuccess: false,
            roadMask: null,
            keypointMask: null,
            isMaskProcessing: false,
            rawMaskData: {},
            // Reset edit state
            editState: {
                mode: null,
                tool: null,
                isDrawing: false,
                currentPolyline: [],
                previewPoint: null,
                hoveredEdge: null,
                hoveredNode: null,
                selectedNode: null,
                isDraggingNode: false,
                selectedNodes: [],
                isRectangleSelecting: false,
                rectangleSelection: null,
            },
            // Clear undo/redo stacks
            undoStack: [],
            redoStack: []
        });
    },

    processMasks: async () => {
        const { rawMaskData } = get();
        
        if (!rawMaskData.roadMask && !rawMaskData.keypointMask) {
            return;
        }
        
        set({ isMaskProcessing: true });
        
        try {
            // Process masks asynchronously to avoid blocking UI
            const processedRoadMask = rawMaskData.roadMask 
                ? processMaskData(rawMaskData.roadMask)
                : null;
                
            const processedKeypointMask = rawMaskData.keypointMask 
                ? processMaskData(rawMaskData.keypointMask)
                : null;
            
            set({
                roadMask: processedRoadMask,
                keypointMask: processedKeypointMask,
                isMaskProcessing: false
            });
            
            console.log('Mask processing completed successfully');
            
        } catch (error) {
            console.error('Error processing masks:', error);
            set({
                roadMask: null,
                keypointMask: null,
                isMaskProcessing: false,
                error: 'Failed to process mask data'
            });
        }
    },

    // Edit mode actions
    setEditMode: (mode) => {
        console.log(`Setting edit mode to: ${mode}`);
        
        // Ensure we always have a valid mode (addEdge or modify), never null
        // This guarantees exclusive selection between the two tools
        const finalMode = mode || 'modify';
        
        // If entering edit mode for the first time, enable defaults
        const isEnteringEditMode = finalMode && get().editState.mode === null;
        
        set(state => ({
            editState: {
                ...state.editState,
                mode: finalMode,
                tool: finalMode, // Set tool to match mode
                // Reset drawing state when changing modes
                isDrawing: false,
                currentPolyline: [],
                previewPoint: null,
                hoveredEdge: null,
                hoveredNode: null,
                selectedNode: null,
                isDraggingNode: false,
            },
            // Default to enable snapping when entering edit mode for the first time
            ...(isEnteringEditMode && {
                snappingEnabled: true,
            })
        }));
    },

    setEditTool: (tool) => {
        set(state => ({
            editState: {
                ...state.editState,
                tool
            }
        }));
    },

    // AddEdge mode actions
    startDrawing: () => {
        set(state => ({
            editState: {
                ...state.editState,
                isDrawing: true,
                currentPolyline: [],
                previewPoint: null,
            }
        }));
    },

    addPointToPolyline: (point) => {
        const { roadNetwork, snappingEnabled, snappingDistance } = get();
        
        let finalPoint = point;
        
        // Apply snapping if enabled and we have a road network
        if (snappingEnabled && snappingDistance > 0 && roadNetwork) {
            const snapTarget = getNearestSnapNode(
                point,
                roadNetwork.nodes,
                snappingDistance
            );
            
            if (snapTarget) {
                finalPoint = { x: snapTarget.x, y: snapTarget.y };
                console.log(`Snapping AddEdge point to existing node ${snapTarget.id} at (${finalPoint.x}, ${finalPoint.y})`);
            }
        }
        
        console.log(`Adding point to polyline: ${finalPoint.x}, ${finalPoint.y}`);
        set(state => ({
            editState: {
                ...state.editState,
                currentPolyline: [...state.editState.currentPolyline, finalPoint],
                isDrawing: true,
            }
        }));
    },

    setPreviewPoint: (point) => {
        set(state => ({
            editState: {
                ...state.editState,
                previewPoint: point,
            }
        }));
    },

    finishPolyline: () => {
        const { editState, roadNetwork, pushToUndoStack } = get();
        
        if (editState.currentPolyline.length < 2) {
            console.warn('Need at least 2 points to create a polyline');
            return;
        }
        
        if (!roadNetwork) {
            console.warn('No road network available to add edges to');
            return;
        }
        
        // Push current state to undo stack
        pushToUndoStack();
        
        // Convert polyline to nodes and edges
        const { nodes: newNodes, edges: newEdges } = polylineToNodesAndEdges(editState.currentPolyline);
        
        // Process node merging for each polyline point
        let finalNodes = [...roadNetwork.nodes];
        let finalEdges = [...roadNetwork.edges];
        const nodeMapping = new Map<number, number>(); // Map from new node ID to existing node ID
        
        // First pass: identify nodes that need to be merged
        for (let i = 0; i < newNodes.length; i++) {
            const newNode = newNodes[i];
            const existingNodesAtPosition = findNodesAtPosition(
                { x: newNode.x, y: newNode.y },
                finalNodes,
                0.1
            );
            
            if (existingNodesAtPosition.length > 0) {
                // Use the first existing node as the merge target
                const targetNode = existingNodesAtPosition[0];
                nodeMapping.set(newNode.id, targetNode.id);
                console.log(`Merging new node ${newNode.id} with existing node ${targetNode.id}`);
            } else {
                // Add new node to the network
                finalNodes.push(newNode);
            }
        }
        
        // Second pass: create edges with proper node mapping
        for (const edge of newEdges) {
            const sourceId = nodeMapping.get(edge.source) || edge.source;
            const targetId = nodeMapping.get(edge.target) || edge.target;
            
            // Skip self-loops
            if (sourceId === targetId) {
                continue;
            }
            
            // Check if edge already exists
            const edgeExists = finalEdges.some(existingEdge => 
                (existingEdge.source === sourceId && existingEdge.target === targetId) ||
                (existingEdge.source === targetId && existingEdge.target === sourceId)
            );
            
            if (!edgeExists) {
                finalEdges.push({
                    ...edge,
                    source: sourceId,
                    target: targetId
                });
            }
        }
        
        const updatedNetwork = {
            ...roadNetwork,
            nodes: finalNodes,
            edges: finalEdges
        };
        
        console.log(`Created polyline with merging: ${finalNodes.length - roadNetwork.nodes.length} new nodes, ${finalEdges.length - roadNetwork.edges.length} new edges`);
        
        // Force immediate update to trigger re-render
        set({
            roadNetwork: updatedNetwork,
            isDirty: true,
            editState: {
                ...editState,
                isDrawing: false,
                currentPolyline: [],
                previewPoint: null,
            }
        });
    },

    cancelDrawing: () => {
        console.log('Cancelling drawing');
        set(state => ({
            editState: {
                ...state.editState,
                isDrawing: false,
                currentPolyline: [],
                previewPoint: null,
            }
        }));
    },

    // Modify mode actions
    setHoveredEdge: (edgeId: number | null) => {
        set(state => ({
            editState: {
                ...state.editState,
                hoveredEdge: edgeId,
            }
        }));
    },

    setHoveredNode: (nodeId: number | null) => {
        set(state => ({
            editState: {
                ...state.editState,
                hoveredNode: nodeId,
            }
        }));
    },

    setSelectedNode: (nodeId: number | null) => {
        console.log(`Selecting node: ${nodeId}`);
        set(state => ({
            editState: {
                ...state.editState,
                selectedNode: nodeId,
            }
        }));
    },

    addNodeAtEdgeMidpoint: (edgeId, x, y) => {
        const { roadNetwork, pushToUndoStack } = get();
        
        if (!roadNetwork) return;
        
        const edge = roadNetwork.edges.find(e => e.id === edgeId);
        if (!edge) return;
        
        // Push current state to undo stack
        pushToUndoStack();
        
        // Create new node at midpoint
        const newNode: RoadNode = {
            id: generateUniqueId(),
            x: x,
            y: y
        };
        
        // Create two new edges to replace the old one
        const newEdge1: RoadEdge = {
            id: generateUniqueId(),
            source: edge.source,
            target: newNode.id
        };
        
        const newEdge2: RoadEdge = {
            id: generateUniqueId(),
            source: newNode.id,
            target: edge.target
        };
        
        // Update road network
        const updatedNetwork = {
            ...roadNetwork,
            nodes: [...roadNetwork.nodes, newNode],
            edges: roadNetwork.edges
                .filter(e => e.id !== edgeId)
                .concat([newEdge1, newEdge2])
        };
        
        console.log(`Added node at edge midpoint: ${newNode.id}`);
        
        // Force immediate update to trigger re-render
        set({
            roadNetwork: updatedNetwork,
            isDirty: true,
        });
    },

    moveNode: (nodeId, x, y, recordUndo = true) => {
        const state = get();
        const { roadNetwork } = state;
        
        if (!roadNetwork) return;
        
        // Push current state to undo stack before making changes (if requested)
        if (recordUndo) {
            get().pushToUndoStack();
        }
        
        // For performance during dragging, only update the specific node position
        // without creating deep copies or triggering heavy operations
        const updatedNodes = roadNetwork.nodes.map(node =>
            node.id === nodeId
                ? { ...node, x: x, y: y }
                : node
        );
        
        const updatedNetwork = {
            ...roadNetwork,
            nodes: updatedNodes
        };
        
        // Use a lighter update during dragging
        set({
            roadNetwork: updatedNetwork,
            isDirty: true,
        });
    },

    moveNodeWithSnapping: (nodeId, x, y) => {
        const { 
            roadNetwork, 
            snappingEnabled, 
            snappingDistance 
        } = get();
        
        if (!roadNetwork) return;
        
        let finalPosition = { x, y };
        let nodesToMerge: number[] = [];
        
        // Apply snapping if enabled
        if (snappingEnabled && snappingDistance > 0) {
            const snapTarget = getNearestSnapNode(
                { x, y },
                roadNetwork.nodes,
                snappingDistance,
                nodeId // Exclude the node being moved
            );
            
            if (snapTarget) {
                finalPosition = { x: snapTarget.x, y: snapTarget.y };
                
                // Find all nodes at the snap position (including the snap target)
                const nodesAtPosition = findNodesAtPosition(
                    finalPosition,
                    roadNetwork.nodes,
                    0.1
                );
                
                // Collect nodes to merge (excluding the node being moved)
                nodesToMerge = nodesAtPosition
                    .filter(node => node.id !== nodeId)
                    .map(node => node.id);
                
                console.log(`Snapping node ${nodeId} to position (${finalPosition.x}, ${finalPosition.y}), merging with nodes: [${nodesToMerge.join(', ')}]`);
            }
        }
        
        // Update the moving node's position first
        const updatedNodesStep1 = roadNetwork.nodes.map(node =>
            node.id === nodeId
                ? { ...node, x: finalPosition.x, y: finalPosition.y }
                : node
        );
        
        // Apply node merging if we have nodes to merge
        let finalNodes = updatedNodesStep1;
        let finalEdges = roadNetwork.edges;
        
        if (nodesToMerge.length > 0) {
            const mergeResult = mergeNodes(
                nodeId, // Use the moved node as the target
                nodesToMerge,
                updatedNodesStep1,
                roadNetwork.edges
            );
            
            finalNodes = mergeResult.updatedNodes;
            finalEdges = mergeResult.updatedEdges;
            
            console.log(`Merged ${nodesToMerge.length} nodes into node ${nodeId}. Final network: ${finalNodes.length} nodes, ${finalEdges.length} edges`);
        }
        
        const updatedNetwork = {
            ...roadNetwork,
            nodes: finalNodes,
            edges: finalEdges
        };
        
        // Force immediate update to trigger re-render
        set({
            roadNetwork: updatedNetwork,
            isDirty: true,
        });
    },

    deleteNodeWithLogic: (nodeId) => {
        const { roadNetwork, pushToUndoStack } = get();
        
        if (!roadNetwork) return;
        
        console.log(`Starting deletion of node ${nodeId}`);
        
        // Push current state to undo stack BEFORE starting deletion
        pushToUndoStack();
        
        // Pre-calculate all needed data to minimize processing time
        const connectedEdges = findConnectedEdges(nodeId, roadNetwork.edges, roadNetwork.nodes);
        const nodeDegree = getNodeDegree(nodeId, roadNetwork.edges, roadNetwork.nodes);
        
        console.log(`Deleting node ${nodeId} with degree ${nodeDegree}`);
        
        // Filter out the node and its connected edges efficiently
        const updatedNodes = roadNetwork.nodes.filter(n => n.id !== nodeId);
        let updatedEdges = roadNetwork.edges.filter(e => 
            e.source !== nodeId && e.target !== nodeId
        );
        
        // If node degree is 2, connect the two neighbors
        if (nodeDegree === 2 && connectedEdges.length === 2) {
            const neighbors = getNeighborNodes(nodeId, roadNetwork.edges, roadNetwork.nodes);
            if (neighbors.length === 2) {
                const newEdge: RoadEdge = {
                    id: generateUniqueId(),
                    source: neighbors[0],
                    target: neighbors[1]
                };
                
                updatedEdges = [...updatedEdges, newEdge];
                console.log(`Connected neighbors ${neighbors[0]} and ${neighbors[1]}`);
            }
        }
        
        // Create updated network in one go
        const updatedNetwork = {
            ...roadNetwork,
            nodes: updatedNodes,
            edges: updatedEdges
        };
        
        // Update state atomically to prevent rendering issues
        set((state) => ({
            roadNetwork: updatedNetwork,
            isDirty: true,
            editState: {
                ...state.editState,
                selectedNode: null, // Clear selection
                hoveredNode: null,  // Clear hover state
                isDraggingNode: false, // Clear dragging state
            }
        }));
        
        console.log(`Node ${nodeId} deletion completed`);
    },

    setIsDraggingNode: (isDragging) => {
        set(state => ({
            editState: {
                ...state.editState,
                isDraggingNode: isDragging,
            }
        }));
    },

    // Rectangle selection actions
    setSelectedNodes: (nodeIds) => {
        set(state => ({
            editState: {
                ...state.editState,
                selectedNodes: nodeIds,
            }
        }));
    },
    addToSelectedNodes: (nodeId) => {
        set(state => ({
            editState: {
                ...state.editState,
                selectedNodes: [...state.editState.selectedNodes, nodeId],
            }
        }));
    },
    removeFromSelectedNodes: (nodeId) => {
        set(state => ({
            editState: {
                ...state.editState,
                selectedNodes: state.editState.selectedNodes.filter(id => id !== nodeId),
            }
        }));
    },
    clearSelectedNodes: () => {
        set(state => ({
            editState: {
                ...state.editState,
                selectedNodes: [],
            }
        }));
    },
    setIsRectangleSelecting: (isSelecting) => {
        set(state => ({
            editState: {
                ...state.editState,
                isRectangleSelecting: isSelecting,
            }
        }));
    },
    setRectangleSelection: (selection) => {
        set(state => ({
            editState: {
                ...state.editState,
                rectangleSelection: selection,
            }
        }));
    },
    updateRectangleSelection: (start, current) => {
        // Use efficient update without deep operations
        set(state => ({
            editState: {
                ...state.editState,
                rectangleSelection: { start, end: current },
            }
        }));
    },
    selectNodesInRectangle: (start, end) => {
        const { roadNetwork } = get();
        if (!roadNetwork) return;

        const selected = getNodesInRectangle(roadNetwork.nodes, start, end);
        set(state => ({
            editState: {
                ...state.editState,
                selectedNodes: selected.map(node => node.id),
            }
        }));
    },
    deleteSelectedNodes: () => {
        const { roadNetwork, pushToUndoStack, editState } = get();
        if (!roadNetwork || editState.selectedNodes.length === 0) return;

        const selectedNodeIds = editState.selectedNodes;
        console.log(`Starting deletion of ${selectedNodeIds.length} selected nodes`);

        // Push current state to undo stack BEFORE starting deletion
        pushToUndoStack();

        // Create sets for efficient lookup
        const selectedIdsSet = new Set(selectedNodeIds);

        // Filter out selected nodes
        const updatedNodes = roadNetwork.nodes.filter(node => !selectedIdsSet.has(node.id));

        // Filter out edges connected to selected nodes
        const updatedEdges = roadNetwork.edges.filter(edge => 
            !selectedIdsSet.has(edge.source) && !selectedIdsSet.has(edge.target)
        );

        const updatedNetwork = {
            ...roadNetwork,
            nodes: updatedNodes,
            edges: updatedEdges
        };

        // Update state atomically
        set(state => ({
            roadNetwork: updatedNetwork,
            isDirty: true,
            editState: {
                ...state.editState,
                selectedNodes: [],
                selectedNode: null, // Clear single selection too
                hoveredNode: null,  // Clear hover state
                isDraggingNode: false, // Clear dragging state
            }
        }));

        console.log(`Batch deletion completed: removed ${selectedNodeIds.length} nodes and ${roadNetwork.edges.length - updatedEdges.length} edges`);
    },

    // Undo/Redo system
    pushToUndoStack: () => {
        const { roadNetwork, undoStack, maxUndoSteps } = get();
        
        if (!roadNetwork) return;
        
        // Simple optimization: avoid pushing identical states
        if (undoStack.length > 0) {
            const lastState = undoStack[undoStack.length - 1];
            if (lastState.nodes.length === roadNetwork.nodes.length && 
                lastState.edges.length === roadNetwork.edges.length) {
                // Quick check if states might be identical
                const nodesEqual = lastState.nodes.every((node, i) => 
                    roadNetwork.nodes[i] && 
                    node.id === roadNetwork.nodes[i].id &&
                    node.x === roadNetwork.nodes[i].x &&
                    node.y === roadNetwork.nodes[i].y
                );
                const edgesEqual = lastState.edges.every((edge, i) =>
                    roadNetwork.edges[i] &&
                    edge.id === roadNetwork.edges[i].id &&
                    edge.source === roadNetwork.edges[i].source &&
                    edge.target === roadNetwork.edges[i].target
                );
                
                if (nodesEqual && edgesEqual) {
                    console.log('Skipping identical state push');
                    return;
                }
            }
        }
        
        try {
            // Use more efficient deep copy for smaller objects
            const currentStateCopy = {
                ...roadNetwork,
                nodes: roadNetwork.nodes.map(node => ({ ...node })),
                edges: roadNetwork.edges.map(edge => ({ ...edge })),
                features: roadNetwork.features || []
            };
            
            // Add to undo stack and maintain max size
            const newUndoStack = [...undoStack, currentStateCopy].slice(-maxUndoSteps);
            
            set({
                undoStack: newUndoStack,
                redoStack: [], // Clear redo stack when new action is performed
            });
            
            console.log(`Pushed state to undo stack. Stack size: ${newUndoStack.length}`);
        } catch (error) {
            console.error('Failed to push state to undo stack:', error);
        }
    },

    undo: () => {
        const { undoStack, roadNetwork } = get();
        
        if (undoStack.length === 0) {
            console.log('Nothing to undo');
            return;
        }
        
        const previousState = undoStack[undoStack.length - 1];
        const newUndoStack = undoStack.slice(0, -1);
        
        set(state => ({
            roadNetwork: previousState,
            undoStack: newUndoStack,
            redoStack: roadNetwork ? [...state.redoStack, roadNetwork] : state.redoStack,
            isDirty: true,
        }));
        
        console.log('Undo performed');
    },

    redo: () => {
        const { redoStack, roadNetwork } = get();
        
        if (redoStack.length === 0) {
            console.log('Nothing to redo');
            return;
        }
        
        const nextState = redoStack[redoStack.length - 1];
        const newRedoStack = redoStack.slice(0, -1);
        
        set(state => ({
            roadNetwork: nextState,
            redoStack: newRedoStack,
            undoStack: roadNetwork ? [...state.undoStack, roadNetwork] : state.undoStack,
            isDirty: true,
        }));
        
        console.log('Redo performed');
    },

    canUndo: () => {
        return get().undoStack.length > 0;
    },

    canRedo: () => {
        return get().redoStack.length > 0;
    },

    generateRoadNetwork: async (imageId: string, promptsForGeneration: Prompt[]) => {
        console.log(`Generating road network for image ${imageId} with ${promptsForGeneration.length} prompts`);
        set({ 
            isLoading: true, 
            error: null, 
            roadNetwork: null, 
            currentNetworkPrompts: null, 
            isDirty: false, 
            saveSuccess: false,
            isComputingFeatures: false,
            roadMask: null,
            keypointMask: null,
            isMaskProcessing: false,
            rawMaskData: {},
            // Reset edit state when generating new network
            editState: {
                mode: null,
                tool: null,
                isDrawing: false,
                currentPolyline: [],
                previewPoint: null,
                hoveredEdge: null,
                hoveredNode: null,
                selectedNode: null,
                isDraggingNode: false,
                selectedNodes: [],
                isRectangleSelecting: false,
                rectangleSelection: null,
            },
            // Clear undo/redo stacks
            undoStack: [],
            redoStack: []
        });
        
        const imageStore = useImageStore.getState();
        const currentImage = imageStore.availableImages.find(img => img.id === imageId);
        
        // Check if features are computed for this image
        if (currentImage && !currentImage.hasFeature) {
            console.log(`Image ${imageId} doesn't have precomputed features. This will be handled automatically by the backend.`);
            // Update UI to show computing features
            set({ isComputingFeatures: true });
            // Update image store feature status too (useful for UI components that observe it)
            imageStore.setFeatureStatus('computing');
        }
        
        try {
            // Transform prompts to backend format
            const promptsForBackend = promptsForGeneration.map(p => ({
                x: p.x,
                y: p.y,
                label: p.type === 'positive' ? 1 : 0,  // Convert type to label (1=positive, 0=negative)
                id: p.id
            }));
            
            const response = await roadNetworkApi.generateRoadNetwork(imageId, promptsForBackend);
            console.log(`Road network generated successfully with ${response.geojson_data.features.length} features`);
            
            // Make sure to validate the prompts coming back from the response
            const validatedPrompts = promptsForGeneration.map(p => ({
                ...p,
                id: p.id || `prompt-${Math.random().toString(36).substring(2, 11)}` // Ensure each prompt has an ID
            }));
            
            // Update feature status in image store if we were computing features
            if (get().isComputingFeatures) {
                // Update the feature status in the image store
                const updatedImageStore = useImageStore.getState();
                updatedImageStore.setFeatureStatus('ready');
                
                // Also update the hasFeature flag in the currentImage and availableImages
                const updatedAvailableImages = updatedImageStore.availableImages.map(img => 
                    img.id === imageId ? { ...img, hasFeature: true, features_computed: true } : img
                );
                
                updatedImageStore.setAvailableImages(updatedAvailableImages);
                if (updatedImageStore.currentImage && updatedImageStore.currentImage.id === imageId) {
                    updatedImageStore.setCurrentImageById(imageId);
                }
            }
            
            // Store raw mask data for later processing
            const rawMaskData: { roadMask?: MaskData; keypointMask?: MaskData } = {};
            
            if (response.road_mask) {
                rawMaskData.roadMask = {
                    data: response.road_mask,
                    metadata: response.road_mask_metadata
                };
            }
            
            if (response.kp_mask) {
                rawMaskData.keypointMask = {
                    data: response.kp_mask,
                    metadata: response.kp_mask_metadata
                };
            }
            
            set({ 
                roadNetwork: response.geojson_data, 
                currentNetworkPrompts: validatedPrompts, // Store validated prompts
                isLoading: false, 
                isDirty: true, // Newly generated network is dirty
                isComputingFeatures: false, // Reset computing features flag
                rawMaskData,
                // Initialize undo stack with the generated network
                undoStack: [],
                redoStack: []
            });
            
            // Process masks asynchronously if we have mask data
            if (Object.keys(rawMaskData).length > 0) {
                // Use setTimeout to ensure UI updates before starting mask processing
                setTimeout(() => {
                    get().processMasks();
                }, 100);
            }
            
        } catch (err) {
            console.error("Error in generateRoadNetwork:", err);
            
            // Reset computing features flag in both stores
            if (get().isComputingFeatures) {
                useImageStore.getState().setFeatureStatus('error');
            }
            
            set({ 
                isLoading: false, 
                error: 'Failed to generate road network', 
                roadNetwork: null, 
                currentNetworkPrompts: null, 
                isDirty: false,
                isComputingFeatures: false,
                roadMask: null,
                keypointMask: null,
                isMaskProcessing: false,
                rawMaskData: {}
            });
        }
    },

    saveRoadNetwork: async (imageId: string) => {
        const currentNetwork = get().roadNetwork;
        if (!currentNetwork) {
            console.warn("No road network to save.");
            set({ error: "No road network data to save." });
            return;
        }
        
        // Get current prompts from promptStore, as they are the source of truth for UI prompts
        const currentPromptsFromStore = usePromptStore.getState().prompts;
        const currentRoadMask = get().roadMask;
        const currentKeypointMask = get().keypointMask;
        
        console.log(`Saving road network for image ${imageId} with ${currentNetwork.features.length} features, ${currentPromptsFromStore.length} prompts, and mask data: ${!!currentRoadMask}, ${!!currentKeypointMask}`);

        set({ isLoading: true, error: null, saveSuccess: false, isSaving: true });
        try {
            // Use cached masks on backend - don't transmit mask data to reduce payload
            const saveResponse = await saveRoadNetworkAPI(
                imageId, 
                currentNetwork, 
                currentPromptsFromStore.map(p => ({
                    x: p.x,
                    y: p.y,
                    label: p.type === 'positive' ? 1 : 0,
                    id: p.id
                })),
                undefined, // No road mask transmission - use backend cache
                undefined  // No keypoint mask transmission - use backend cache
            );
            
            console.log(`Road network saved successfully: ${saveResponse?.message || 'Success'}`);
            set({ 
                isLoading: false, 
                isDirty: false, 
                lastSaveResponse: saveResponse, 
                saveSuccess: true,
                isSaving: false
            }); // Saved, so no longer dirty
            
            // Mark the image as annotated in the image store
            const imageStore = useImageStore.getState();
            const currentImages = imageStore.availableImages;
            const updatedImages = currentImages.map(img => 
                img.id === imageId 
                    ? { ...img, isAnnotated: true }
                    : img
            );
            
            // Update the image store with the updated image list
            imageStore.setAvailableImages(updatedImages);
            
            // If this is the current image, directly update its isAnnotated property
            if (imageStore.currentImage?.id === imageId) {
                imageStore.updateCurrentImage({ isAnnotated: true });
            }
            
            // Set a timeout to reset the success message
            setTimeout(() => {
                set({ saveSuccess: false });
            }, 2000);
            
        } catch (err) {
            console.error("Error in saveRoadNetwork:", err);
            set({ 
                isLoading: false, 
                error: 'Failed to save road network', 
                saveSuccess: false,
                isSaving: false
            });
        }
    },

    loadSavedRoadNetwork: async (imageId: string, abortSignal?: AbortSignal) => {
        console.log(`Loading saved road network for image ${imageId}`);
        // Start loading but don't clear network yet
        set({ isLoading: true, error: null, saveSuccess: false });
        try {
            const response = await roadNetworkApi.loadSavedRoadNetwork(imageId, abortSignal);
            // response can be RoadNetGenerationResponse or null
            
            if (response) {
                console.log(`Road network loaded with ${response.geojson_data.features.length} features and ${response.prompts?.length || 0} prompts`);
                
                // Store raw mask data for processing
                const rawMaskData: { roadMask?: MaskData; keypointMask?: MaskData } = {};
                
                if (response.road_mask) {
                    rawMaskData.roadMask = {
                        data: response.road_mask,
                        metadata: response.road_mask_metadata
                    };
                }
                
                if (response.kp_mask) {
                    rawMaskData.keypointMask = {
                        data: response.kp_mask,
                        metadata: response.kp_mask_metadata
                    };
                }
                
                set({ 
                    roadNetwork: response.geojson_data,
                    currentNetworkPrompts: response.prompts || null,
                    isLoading: false,
                    isDirty: false, // Loaded network is not dirty
                    rawMaskData,
                    // Reset edit state when loading
                    editState: {
                        mode: null,
                        tool: null,
                        isDrawing: false,
                        currentPolyline: [],
                        previewPoint: null,
                        hoveredEdge: null,
                        hoveredNode: null,
                        selectedNode: null,
                        isDraggingNode: false,
                        selectedNodes: [],
                        isRectangleSelecting: false,
                        rectangleSelection: null,
                    },
                    // Clear undo/redo stacks for loaded network
                    undoStack: [],
                    redoStack: []
                });
                
                // Update the image's annotation status in the imageStore since we successfully loaded a road network
                const imageStore = useImageStore.getState();
                if (imageStore.currentImage?.id === imageId) {
                    // Update the current image directly
                    imageStore.updateCurrentImage({ isAnnotated: true });
                }
                
                // Also update the image in the availableImages list
                const updatedImages = imageStore.availableImages.map(img => 
                    img.id === imageId ? { ...img, isAnnotated: true } : img
                );
                imageStore.setAvailableImages(updatedImages);
                
                // Update promptStore with the loaded prompts
                if (response.prompts && response.prompts.length > 0) {
                    console.log(`Setting ${response.prompts.length} loaded prompts:`, response.prompts);
                    // Convert backend prompts to frontend format
                    const promptsWithTypes = response.prompts.map((p: any) => ({
                        x: p.x,
                        y: p.y,
                        type: p.label === 1 ? 'positive' : 'negative',
                        id: p.id || undefined
                    }));
                    
                    // Ensure all prompts have valid IDs before setting them
                    const validatedPrompts = ensurePromptIds(promptsWithTypes);
                    usePromptStore.getState().setLoadedPrompts(validatedPrompts);
                } else {
                    console.log('No prompts found with network, cleared prompt store');
                    usePromptStore.getState().clearPrompts(); // Clear if no prompts came with network
                }
                
                // Process masks asynchronously if we have mask data
                if (Object.keys(rawMaskData).length > 0) {
                    setTimeout(() => {
                        get().processMasks();
                    }, 100);
                }
                
            } else {
                // This means a 404 was returned from the API (handled as null)
                console.log(`Image ${imageId} does not have any saved road network. Ready for new annotations.`);
                set({ 
                    isLoading: false,
                    roadNetwork: null,
                    roadMask: null,
                    keypointMask: null,
                    rawMaskData: {},
                    // Reset edit state
                    editState: {
                        mode: null,
                        tool: null,
                        isDrawing: false,
                        currentPolyline: [],
                        previewPoint: null,
                        hoveredEdge: null,
                        hoveredNode: null,
                        selectedNode: null,
                        isDraggingNode: false,
                        selectedNodes: [],
                        isRectangleSelecting: false,
                        rectangleSelection: null,
                    },
                    // Clear undo/redo stacks
                    undoStack: [],
                    redoStack: []
                });
                usePromptStore.getState().clearPrompts();
            }
        } catch (err: any) {
            // Handle AbortError specially - don't treat as an error state
            if (err.name === 'AbortError') {
                console.log(`Load operation for image ${imageId} was aborted`);
                // Don't update error state for aborted requests, just stop loading
                set({ isLoading: false });
                return;
            }
            
            // This catch block will now handle only unexpected errors, as 404s are handled above
            console.error("Error in loadSavedRoadNetwork:", err);
            
            // Provide a more user-friendly error message
            let errorMessage = 'Failed to load saved road network';
            
            // Handle different types of errors
            if (err.message?.includes('timeout')) {
                errorMessage = 'Loading timeout - the saved road network data is too large. The file has been optimized for future saves.';
            } else if (err.message?.includes('JSON')) {
                errorMessage = 'Data parsing error - the saved file may be corrupted or too large. Please regenerate the road network.';
            } else if (err.response?.data?.detail) {
                errorMessage += `: ${err.response.data.detail}`;
            } else if (err.message) {
                errorMessage += `: ${err.message}`;
            }
            
            set({ 
                isLoading: false, 
                error: errorMessage,
                roadNetwork: null,
                roadMask: null,
                keypointMask: null,
                rawMaskData: {},
                // Reset edit state on error
                editState: {
                    mode: null,
                    tool: null,
                    isDrawing: false,
                    currentPolyline: [],
                    previewPoint: null,
                    hoveredEdge: null,
                    hoveredNode: null,
                    selectedNode: null,
                    isDraggingNode: false,
                    selectedNodes: [],
                    isRectangleSelecting: false,
                    rectangleSelection: null,
                },
                // Clear undo/redo stacks on error
                undoStack: [],
                redoStack: []
            });
            usePromptStore.getState().clearPrompts();
        }
    },

    clearCurrentRoadNetwork: () => {
        console.log('Clearing current road network and prompts');
        set({ 
            roadNetwork: null, 
            currentNetworkPrompts: null, 
            isLoading: false, 
            error: null, 
            isDirty: false, 
            saveSuccess: false,
            roadMask: null,
            keypointMask: null,
            isMaskProcessing: false,
            rawMaskData: {},
            // Reset edit state
            editState: {
                mode: null,
                tool: null,
                isDrawing: false,
                currentPolyline: [],
                previewPoint: null,
                hoveredEdge: null,
                hoveredNode: null,
                selectedNode: null,
                isDraggingNode: false,
                selectedNodes: [],
                isRectangleSelecting: false,
                rectangleSelection: null,
            },
            // Clear undo/redo stacks
            undoStack: [],
            redoStack: []
        });
        usePromptStore.getState().clearPrompts(); // Also clear prompts in promptStore
    },

    setNetworkAsDirty: () => {
        set({ isDirty: true });
    },

    // Snapping actions
    setSnappingEnabled: (enabled) => {
        set({ snappingEnabled: enabled });
    },

    setSnappingDistance: (distance) => {
        // Ensure distance is within reasonable bounds (0-30 pixels)
        const clampedDistance = Math.max(0, Math.min(30, distance));
        set({ snappingDistance: clampedDistance });
    },
})));