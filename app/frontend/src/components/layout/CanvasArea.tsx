import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle, Line as KonvaLine, Rect} from 'react-konva';
import Konva from 'konva'; // Import Konva namespace for types
import { useImageStore } from '../../store/imageStore';
import { usePromptStore } from '../../store/promptStore';
import { useRoadNetworkStore } from '../../store/roadNetworkStore';
import { useUiStore } from '../../store/uiStore'; // Added uiStore
import { 
    findNearestEdge, 
    findNearestNode, 
    getEdgeMidpoint, 
    applySnapping,
    getNearestSnapNode,
    findNodesAtPosition 
} from '../../utils/geometryUtils';
import { KonvaEventObject } from 'konva/lib/Node';
import { Stage as StageType } from 'konva/lib/Stage';
import { globalEventBus } from '../../utils/eventBus'; // Import event bus
import Minimap from '../editor/Minimap'; // Try editor path
import MainToolBar from '../common/MainToolBar'; // Import unified toolbar
import StatusOverlay from '../common/StatusOverlay'; // Import status overlay
import DownloadProgressOverlay from '../common/DownloadProgressOverlay';
import SaveNotification from '../common/SaveNotification'; // Import SaveNotification component
import PatchGridOverlay from '../editor/PatchGridOverlay'; // Import PatchGridOverlay component
import { 
    createMaskCanvas, 
    processMaskWithWorker, 
    shouldUseWorkerProcessing, 
    estimateMaskProcessingTime 
} from '../../utils/maskUtils';
import OptimizedRoadNetworkLayer from '../rendering/OptimizedRoadNetworkLayer';
import { useProgressiveImage } from '../../hooks/useProgressiveImage';
import { getImageUrl } from '../../services/api';

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 10;
const ZOOM_STEP = 1.1;
// const BACKEND_URL = 'http://localhost:8000'; // add backend server URL (unused)
const INITIAL_PADDING_FACTOR = 0.8; // Leave 10% padding on each side (1 - 0.8 = 0.2)

// Main CanvasArea Component
const CanvasArea = () => {
    const currentImage = useImageStore((state) => state.currentImage);
    // Use selector function to minimize re-renders
    const prompts = usePromptStore((state) => state.prompts);
    const addPrompt = usePromptStore((state) => state.addPrompt);
    // Use selector function to minimize re-renders, only get the roadNetwork data
    const roadNetwork = useRoadNetworkStore((state) => state.roadNetwork);
    // const clearPrompts = usePromptStore((state) => state.clearPrompts);
    // const clearRoadNetwork = useRoadNetworkStore((state) => state.clearRoadNetwork);
    // Get saveSuccess and resetSaveSuccess from roadNetworkStore
    const saveSuccess = useRoadNetworkStore((state) => state.saveSuccess);
    const resetSaveSuccess = useRoadNetworkStore((state) => state.resetSaveSuccess);
    // Add mask data from roadNetworkStore (now boolean arrays)
    const roadMask = useRoadNetworkStore((state) => state.roadMask);
    const keypointMask = useRoadNetworkStore((state) => state.keypointMask);
    const isMaskProcessing = useRoadNetworkStore((state) => state.isMaskProcessing);
    // Select state pieces individually to prevent unnecessary re-renders
    const interactionMode = useUiStore((state) => state.interactionMode);
    const layerVisibility = useUiStore((state) => state.layerVisibility);
    // Get minimap visibility state
    const minimapVisible = useUiStore((state) => state.minimapVisible);
    // const toggleMinimapVisibility = useUiStore((state) => state.toggleMinimapVisibility);
    
    // Get edit state and actions
    const editState = useRoadNetworkStore((state) => state.editState);
    const setHoveredEdge = useRoadNetworkStore((state) => state.setHoveredEdge);
    const setHoveredNode = useRoadNetworkStore((state) => state.setHoveredNode);
    const setSelectedNode = useRoadNetworkStore((state) => state.setSelectedNode);
    const addPointToPolyline = useRoadNetworkStore((state) => state.addPointToPolyline);
    const setPreviewPoint = useRoadNetworkStore((state) => state.setPreviewPoint);
    const finishPolyline = useRoadNetworkStore((state) => state.finishPolyline);
    const cancelDrawing = useRoadNetworkStore((state) => state.cancelDrawing);
    const addNodeAtEdgeMidpoint = useRoadNetworkStore((state) => state.addNodeAtEdgeMidpoint);
    const moveNode = useRoadNetworkStore((state) => state.moveNode);
    const moveNodeWithSnapping = useRoadNetworkStore((state) => state.moveNodeWithSnapping);
    const deleteNodeWithLogic = useRoadNetworkStore((state) => state.deleteNodeWithLogic);
    const setIsDraggingNode = useRoadNetworkStore((state) => state.setIsDraggingNode);
    const pushToUndoStack = useRoadNetworkStore((state) => state.pushToUndoStack);
    const undo = useRoadNetworkStore((state) => state.undo);
    const redo = useRoadNetworkStore((state) => state.redo);
    
    // Rectangle selection actions
    const setSelectedNodes = useRoadNetworkStore((state) => state.setSelectedNodes);
    const clearSelectedNodes = useRoadNetworkStore((state) => state.clearSelectedNodes);
    const setIsRectangleSelecting = useRoadNetworkStore((state) => state.setIsRectangleSelecting);
    const setRectangleSelection = useRoadNetworkStore((state) => state.setRectangleSelection);
    const updateRectangleSelection = useRoadNetworkStore((state) => state.updateRectangleSelection);
    const selectNodesInRectangle = useRoadNetworkStore((state) => state.selectNodesInRectangle);
    const deleteSelectedNodes = useRoadNetworkStore((state) => state.deleteSelectedNodes);

    // Get snapping state
    const snappingEnabled = useRoadNetworkStore((state) => state.snappingEnabled);
    const snappingDistance = useRoadNetworkStore((state) => state.snappingDistance);

    // const backendUrl = import.meta.env.VITE_BACKEND_URL || BACKEND_URL;
    // const absoluteImageUrl = currentImage?.url ? `${backendUrl}${currentImage.url}` : null;
    const {
        loadingState: imageLoadingState,
        currentImageElement: image
    } = useProgressiveImage({
        imageUrl: currentImage ? getImageUrl(currentImage.id) : undefined,
        imageId: currentImage?.id, // Add imageId for chunked downloads
        preferWebP: false, // Do not prefer WebP to avoid conversion overhead
        timeout: 60000 // 60 seconds timeout for large images
    });
    const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
    const [stageScale, setStageScale] = useState(1);
    const stageRef = useRef<StageType>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const imageNodeRef = useRef<Konva.Image>(null);
    const [containerDims, setContainerDims] = useState<{width: number, height: number} | null>(null);
    // Add state to track if user is currently interacting with the canvas
    const [isInteracting, setIsInteracting] = useState(false);
    // Add a ref to store the interaction timeout
    const interactionTimeoutRef = useRef<number | null>(null);
    
    // Track dragging node position for snapping feedback
    const [draggingNodePosition, setDraggingNodePosition] = useState<{nodeId: number, x: number, y: number} | null>(null);

    // Track Ctrl key state for rectangle selection
    const [isCtrlPressed, setIsCtrlPressed] = useState(false);

    // Track the last known viewport state even when minimap is hidden
    const lastKnownViewportRef = useRef({
        position: { x: 0, y: 0 },
        scale: 1,
    });

    // Add mask rendering state
    const [maskRenderState, setMaskRenderState] = useState<{
        roadMaskImage: HTMLImageElement | null;
        keypointMaskImage: HTMLImageElement | null;
        isRenderingMasks: boolean;
        renderProgress: number;
    }>({
        roadMaskImage: null,
        keypointMaskImage: null,
        isRenderingMasks: false,
        renderProgress: 0
    });

    // Effect to keep track of the last known viewport state
    useEffect(() => {
        if (stageRef.current) {
            lastKnownViewportRef.current = {
                position: stageRef.current.position(),
                scale: stageRef.current.scaleX(),
            };
        }
    }, [stagePos, stageScale]);

    // Enhanced mask rendering function using worker for large images
    const renderMasks = useCallback(async () => {
        if (!image || imageLoadingState.isLoading || imageLoadingState.error) return;
        
        const width = image.width;
        const height = image.height;
        
        // Clear previous mask images
        setMaskRenderState(prev => ({
            ...prev,
            roadMaskImage: null,
            keypointMaskImage: null,
            isRenderingMasks: true,
            renderProgress: 0
        }));
        
        try {
            const useWorker = shouldUseWorkerProcessing(width, height);
            const estimatedTime = estimateMaskProcessingTime(width, height);
            
            if (useWorker && estimatedTime > 1000) {
                console.log(`Using Web Worker for mask processing (estimated ${estimatedTime}ms)`);
            }
            
            // Render road mask
            if (roadMask && layerVisibility.roadMask) {
                setMaskRenderState(prev => ({ ...prev, renderProgress: 25 }));
                
                let roadMaskImage: HTMLImageElement;
                
                if (useWorker) {
                    const imageData = await processMaskWithWorker(
                        roadMask, 
                        width, 
                        height, 
                        { r: 0, g: 255, b: 0, a: 200 }
                    );
                    
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        ctx.putImageData(imageData, 0, 0);
                        roadMaskImage = new Image();
                        roadMaskImage.src = canvas.toDataURL();
                    }
                } else {
                    const canvas = createMaskCanvas(
                        roadMask, 
                        width, 
                        height, 
                        { r: 0, g: 255, b: 0, a: 200 }
                    );
                    roadMaskImage = new Image();
                    roadMaskImage.src = canvas.toDataURL();
                }
                
                setMaskRenderState(prev => ({ 
                    ...prev, 
                    roadMaskImage: roadMaskImage,
                    renderProgress: 50 
                }));
            }
            
            // Render keypoint mask
            if (keypointMask && layerVisibility.keypointMask) {
                setMaskRenderState(prev => ({ ...prev, renderProgress: 75 }));
                
                let keypointMaskImage: HTMLImageElement;
                
                if (useWorker) {
                    const imageData = await processMaskWithWorker(
                        keypointMask, 
                        width, 
                        height, 
                        { r: 160, g: 0, b: 255, a: 200 }
                    );
                    
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        ctx.putImageData(imageData, 0, 0);
                        keypointMaskImage = new Image();
                        keypointMaskImage.src = canvas.toDataURL();
                    }
                } else {
                    const canvas = createMaskCanvas(
                        keypointMask, 
                        width, 
                        height, 
                        { r: 160, g: 0, b: 255, a: 200 }
                    );
                    keypointMaskImage = new Image();
                    keypointMaskImage.src = canvas.toDataURL();
                }
                
                setMaskRenderState(prev => ({ 
                    ...prev, 
                    keypointMaskImage: keypointMaskImage,
                    renderProgress: 100 
                }));
            }
            
        } catch (error) {
            console.error('Error rendering masks:', error);
        } finally {
            setMaskRenderState(prev => ({ 
                ...prev, 
                isRenderingMasks: false 
            }));
        }
    }, [roadMask, keypointMask, image, imageLoadingState, layerVisibility.roadMask, layerVisibility.keypointMask]);
    
    // Effect to trigger mask rendering when masks change
    useEffect(() => {
        if (roadMask || keypointMask) {
            renderMasks();
        } else {
            // Clear mask images when no masks
            setMaskRenderState({
                roadMaskImage: null,
                keypointMaskImage: null,
                isRenderingMasks: false,
                renderProgress: 0
            });
        }
    }, [renderMasks]);

    // Fit image to container initially or when image changes, adding padding
    useEffect(() => {
        if (image && !imageLoadingState.isLoading && !imageLoadingState.error && stageRef.current && containerRef.current) {
            const stage = stageRef.current;
            const container = containerRef.current;
            setContainerDims({width: container.offsetWidth, height: container.offsetHeight}); // Store dimensions

            const paddedContainerWidth = container.offsetWidth * INITIAL_PADDING_FACTOR;
            const paddedContainerHeight = container.offsetHeight * INITIAL_PADDING_FACTOR;
            const imageWidth = image.width;
            const imageHeight = image.height;

            const scale = Math.min(
                paddedContainerWidth / imageWidth,
                paddedContainerHeight / imageHeight,
                1
            );

            const newScale = Math.max(scale, MIN_ZOOM);
            const newX = (container.offsetWidth - imageWidth * newScale) / 2;
            const newY = (container.offsetHeight - imageHeight * newScale) / 2;

            if (stage.scaleX() !== newScale || stage.x() !== newX || stage.y() !== newY) {
                console.log("Fitting image to view with padding...");
                setStageScale(newScale);
                setStagePos({ x: newX, y: newY });
                stage.scale({ x: newScale, y: newScale });
                stage.position({ x: newX, y: newY });
                stage.batchDraw();
                // Fire custom event to notify other components of scale change
                stage.fire('scaleChange');
            }
        }
    }, [image?.src, imageLoadingState]);

    // Effect to handle image changes - don't clear automatically
    useEffect(() => {
        if (currentImage?.id) {
            console.log(`Image changed to ${currentImage.id}, waiting for saved data...`);
            // Don't automatically clear - let the road network store handle loading saved data
            // clearPrompts();
            // clearRoadNetwork();
        }
    }, [currentImage?.id]);

    // Enhanced zoom function that accepts target point coordinates
    const zoomToPoint = useCallback((scaleMultiplier: number, targetX?: number, targetY?: number) => {
        const stage = stageRef.current;
        if (!stage) return;
        
        // Set interaction flag
        setIsInteracting(true);
        
        // Clear any existing timeout
        if (interactionTimeoutRef.current !== null) {
            window.clearTimeout(interactionTimeoutRef.current);
        }
        
        const oldScale = stage.scaleX();
        let newScale = oldScale * scaleMultiplier;
        newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newScale));
        
        let mousePointTo = { x: 0, y: 0 };
        
        // If target point is provided, use it; otherwise use stage center
        if (targetX !== undefined && targetY !== undefined) {
            // We have a specific target point in image coordinates
            mousePointTo = {
                x: targetX,
                y: targetY
            };
        } else {
            // Use stage center as default
            const center = { x: stage.width() / 2, y: stage.height() / 2 };
            mousePointTo = {
                x: (center.x - stage.x()) / oldScale,
                y: (center.y - stage.y()) / oldScale,
            };
        }
        
        // Calculate new position to maintain the target point at the same screen position
        const newPos = {
            x: -(mousePointTo.x * newScale) + (stage.width() / 2),
            y: -(mousePointTo.y * newScale) + (stage.height() / 2)
        };
        
        // Update state and stage
        setStageScale(newScale);
        setStagePos(newPos);
        stage.scale({ x: newScale, y: newScale });
        stage.position(newPos);
        stage.batchDraw();
        
        // Fire custom event to notify minimap of scale change
        stage.fire('scaleChange');
        
        // Reset interaction flag after a delay
        interactionTimeoutRef.current = window.setTimeout(() => {
            setIsInteracting(false);
            interactionTimeoutRef.current = null;
        }, 300);
    }, []);

    // Zoom in/out without specific target point (for button clicks)
    const zoomIn = useCallback(() => zoomToPoint(ZOOM_STEP), [zoomToPoint]);
    const zoomOut = useCallback(() => zoomToPoint(1 / ZOOM_STEP), [zoomToPoint]);

    // Handle zoom from minimap
    const handleMinimapZoom = useCallback((direction: 'in' | 'out', x: number, y: number) => {
        if (direction === 'in') {
            zoomToPoint(ZOOM_STEP, x, y);
        } else {
            zoomToPoint(1 / ZOOM_STEP, x, y);
        }
    }, [zoomToPoint]);

    // Zoom handler for mouse wheel on main canvas
    const handleWheel = useCallback((e: KonvaEventObject<WheelEvent>) => {
        e.evt.preventDefault();
        
        // Set interaction flag
        setIsInteracting(true);
        
        // Clear any existing timeout
        if (interactionTimeoutRef.current !== null) {
            window.clearTimeout(interactionTimeoutRef.current);
        }
        
        const scaleBy = ZOOM_STEP;
        const stage = e.target.getStage();
        if (!stage) return;

        const oldScale = stage.scaleX();
        const pointer = stage.getPointerPosition();
        if (!pointer) return;

        const mousePointTo = {
            x: (pointer.x - stage.x()) / oldScale,
            y: (pointer.y - stage.y()) / oldScale,
        };

        const direction = e.evt.deltaY > 0 ? -1 : 1;
        let newScale = oldScale * (scaleBy ** direction);
        newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newScale));

        setStageScale(newScale);
        setStagePos({
            x: pointer.x - mousePointTo.x * newScale,
            y: pointer.y - mousePointTo.y * newScale,
        });
        
        // Fire custom event to notify minimap of scale change
        stage.fire('scaleChange');
        
        // Reset interaction flag after a delay
        interactionTimeoutRef.current = window.setTimeout(() => {
            setIsInteracting(false);
            interactionTimeoutRef.current = null;
        }, 300);
    }, []);

    // Handle stage drag
    const handleDragStart = useCallback(() => {
        setIsInteracting(true);
        
        // Clear any existing timeout
        if (interactionTimeoutRef.current !== null) {
            window.clearTimeout(interactionTimeoutRef.current);
        }
    }, []);
    
    const handleDragEnd = useCallback((e: KonvaEventObject<DragEvent>) => {
        const stage = e.target;
        setStagePos(stage.position());
        
        // Fire custom event to ensure minimap updates
        stage.fire('scaleChange');
        
        // Reset interaction flag after a delay
        if (interactionTimeoutRef.current !== null) {
            window.clearTimeout(interactionTimeoutRef.current);
        }
        
        interactionTimeoutRef.current = window.setTimeout(() => {
            setIsInteracting(false);
            interactionTimeoutRef.current = null;
        }, 300);
    }, []);

    const handleClick = useCallback((e: KonvaEventObject<MouseEvent>) => {
        // Handle Label mode with both left and right click for different prompt types
        if (interactionMode === 'label') {
            // Only handle left (button 0) and right (button 2) mouse button clicks
            if (e.evt.button !== 0 && e.evt.button !== 2) {
                return;
            }

            if (e.target !== stageRef.current && e.target !== imageNodeRef.current) {
                console.log('Clicked on a shape, not stage or image:', e.target.getClassName());
                return;
            }

            const stage = e.target.getStage();
            const imageNode = imageNodeRef.current;

            if (!stage || !imageNode || !image) return;

            const pointerPosition = stage.getPointerPosition();
            if (!pointerPosition) return;

            const transform = imageNode.getAbsoluteTransform().copy().invert();
            const point = transform.point(pointerPosition);

            if (point.x >= 0 && point.x <= image.width && point.y >= 0 && point.y <= image.height) {
                // Determine prompt type based on mouse button
                const promptType = e.evt.button === 0 ? 'positive' : 'negative'; // Left = positive, Right = negative
                console.log(`Adding ${promptType} prompt at image coordinates:`, point);
                addPrompt({ x: point.x, y: point.y }, promptType);
            } else {
                console.log("Clicked outside image bounds.");
            }
            return;
        }

        // For edit mode, only handle left mouse button clicks (button 0)
        if (e.evt.button !== 0) {
            return;
        }

        // Handle Edit mode
        if (interactionMode === 'edit' && roadNetwork) {
            const stage = e.target.getStage();
            const imageNode = imageNodeRef.current;

            if (!stage || !imageNode || !image) return;

            const pointerPosition = stage.getPointerPosition();
            if (!pointerPosition) return;

            const transform = imageNode.getAbsoluteTransform().copy().invert();
            const point = transform.point(pointerPosition);

            // Only handle clicks within image bounds
            if (point.x < 0 || point.x > image.width || point.y < 0 || point.y > image.height) {
                return;
            }

            // Handle AddEdge mode
            if (editState.mode === 'addEdge') {
                // Apply snapping if enabled
                const snappedPoint = applySnapping(
                    { x: point.x, y: point.y },
                    roadNetwork.nodes,
                    snappingDistance,
                    snappingEnabled
                );
                addPointToPolyline(snappedPoint);
                return;
            }

            // Handle Modify mode
            if (editState.mode === 'modify') {
                // If Ctrl is pressed and we're not already rectangle selecting, don't handle regular clicks
                if (isCtrlPressed && !editState.isRectangleSelecting) {
                    return;
                }

                // Check if clicked on a node first
                const nearestNode = findNearestNode(point.x, point.y, roadNetwork.nodes, 15);
                if (nearestNode) {
                    // Handle multi-selection with Ctrl key
                    if (isCtrlPressed) {
                        if (editState.selectedNodes.includes(nearestNode.node.id)) {
                            // Remove from selection
                            setSelectedNodes(editState.selectedNodes.filter(id => id !== nearestNode.node.id));
                        } else {
                            // Add to selection
                            setSelectedNodes([...editState.selectedNodes, nearestNode.node.id]);
                        }
                    } else {
                        // Single selection
                        setSelectedNode(editState.selectedNode === nearestNode.node.id ? null : nearestNode.node.id);
                        clearSelectedNodes(); // Clear multi-selection when single selecting
                    }
                    return;
                }

                // Check if clicked on edge midpoint to add node
                if (editState.hoveredEdge) {
                    const edge = roadNetwork.edges.find(e => e.id === editState.hoveredEdge);
                    if (edge) {
                        const midpoint = getEdgeMidpoint(edge, roadNetwork.nodes);
                        if (midpoint) {
                            addNodeAtEdgeMidpoint(editState.hoveredEdge, midpoint.x, midpoint.y);
                        }
                    }
                    return;
                }

                // Clear selection if clicked on empty space
                if (!isCtrlPressed) {
                    setSelectedNode(null);
                    clearSelectedNodes();
                }
            }
        }
    }, [interactionMode, image, addPrompt, roadNetwork, editState, addPointToPolyline, setSelectedNode, addNodeAtEdgeMidpoint, isCtrlPressed, setSelectedNodes, clearSelectedNodes]);

    // Handle mouse down for rectangle selection
    const handleMouseDown = useCallback((e: KonvaEventObject<MouseEvent>) => {
        // Only handle left mouse button
        if (e.evt.button !== 0) return;

        // Only in modify mode with Ctrl pressed
        if (interactionMode === 'edit' && editState.mode === 'modify' && isCtrlPressed && roadNetwork) {
            const stage = e.target.getStage();
            const imageNode = imageNodeRef.current;

            if (!stage || !imageNode || !image) return;

            const pointerPosition = stage.getPointerPosition();
            if (!pointerPosition) return;

            const transform = imageNode.getAbsoluteTransform().copy().invert();
            const point = transform.point(pointerPosition);

            // Only start rectangle selection within image bounds
            if (point.x >= 0 && point.x <= image.width && point.y >= 0 && point.y <= image.height) {
                setIsRectangleSelecting(true);
                setRectangleSelection({ start: point, end: point });
                console.log('Started rectangle selection at:', point);
            }
        }
    }, [interactionMode, editState.mode, isCtrlPressed, roadNetwork, image, setIsRectangleSelecting, setRectangleSelection]);

    // Optimized mouse move handler for better performance
    const optimizedMouseMoveHandler = useMemo(() => {
        let lastCallTime = 0;
        const minInterval = 8; // ~120fps for smooth interaction
        
        return (e: KonvaEventObject<MouseEvent>) => {
            const now = performance.now();
            if (now - lastCallTime < minInterval) {
                return;
            }
            lastCallTime = now;
            
            // Handle rectangle selection
            if (editState.isRectangleSelecting && editState.rectangleSelection) {
                const stage = e.target.getStage();
                const imageNode = imageNodeRef.current;

                if (!stage || !imageNode || !image) return;

                const pointerPosition = stage.getPointerPosition();
                if (!pointerPosition) return;

                const transform = imageNode.getAbsoluteTransform().copy().invert();
                const point = transform.point(pointerPosition);

                // Update rectangle selection using requestAnimationFrame
                requestAnimationFrame(() => {
                    updateRectangleSelection(editState.rectangleSelection!.start, point);
                });
                return;
            }

            // Continue with existing mouse move logic for other modes
            if (interactionMode !== 'edit' || !roadNetwork || !image) return;

            const stage = e.target.getStage();
            const imageNode = imageNodeRef.current;

            if (!stage || !imageNode) return;

            const pointerPosition = stage.getPointerPosition();
            if (!pointerPosition) return;

            const transform = imageNode.getAbsoluteTransform().copy().invert();
            const point = transform.point(pointerPosition);

            // Only handle mouse move within image bounds
            if (point.x < 0 || point.x > image.width || point.y < 0 || point.y > image.height) {
                setPreviewPoint(null);
                setHoveredEdge(null);
                setHoveredNode(null);
                return;
            }

            // Handle AddEdge mode preview
            if (editState.mode === 'addEdge' && editState.isDrawing) {
                // Apply snapping to preview point if enabled
                const snappedPoint = applySnapping(
                    { x: point.x, y: point.y },
                    roadNetwork.nodes,
                    snappingDistance,
                    snappingEnabled
                );
                setPreviewPoint(snappedPoint);
                return;
            }

            // Handle Modify mode hover detection (skip if rectangle selecting)
            if (editState.mode === 'modify' && !editState.isRectangleSelecting) {
                // Clear previous hover states
                setHoveredEdge(null);
                setHoveredNode(null);

                // Check for node hover first (higher priority)
                const nearestNode = findNearestNode(point.x, point.y, roadNetwork.nodes, 15);
                if (nearestNode) {
                    setHoveredNode(nearestNode.node.id);
                    return;
                }

                // Check for edge hover
                const nearestEdge = findNearestEdge(point.x, point.y, roadNetwork.edges, roadNetwork.nodes, 10);
                if (nearestEdge) {
                    setHoveredEdge(nearestEdge.edge.id);
                }
            }
        };
    }, [interactionMode, roadNetwork, image, editState, setPreviewPoint, setHoveredEdge, setHoveredNode, updateRectangleSelection]);

    // Handle mouse move for rectangle selection
    const handleMouseMoveForSelection = useCallback((e: KonvaEventObject<MouseEvent>) => {
        optimizedMouseMoveHandler(e);
    }, [optimizedMouseMoveHandler]);

    // Handle mouse up for rectangle selection
    const handleMouseUp = useCallback(() => {
        if (editState.isRectangleSelecting && editState.rectangleSelection && roadNetwork) {
            // Complete rectangle selection
            const { start, end } = editState.rectangleSelection;
            selectNodesInRectangle(start, end);
            setIsRectangleSelecting(false);
            setRectangleSelection(null);
            console.log('Completed rectangle selection from', start, 'to', end);
        }
    }, [editState.isRectangleSelecting, editState.rectangleSelection, roadNetwork, selectNodesInRectangle, setIsRectangleSelecting, setRectangleSelection]);

    // Handle right-click for different modes
    const handleRightClick = useCallback((e: KonvaEventObject<MouseEvent>) => {
        // In label mode, allow right-click to add negative prompts via handleClick
        if (interactionMode === 'label') {
            e.evt.preventDefault(); // Still prevent context menu
            return; // Let handleClick handle the prompt addition
        }
        
        // In edit mode, prevent context menu and handle polyline completion
        e.evt.preventDefault(); // Prevent context menu
        
        if (interactionMode === 'edit' && editState.mode === 'addEdge' && editState.isDrawing) {
            finishPolyline();
        }
    }, [interactionMode, editState, finishPolyline]);

    useEffect(() => {
        const stage = stageRef.current;
        if (!stage) return;
        const container = stage.container();

        const setCursor = (cursorStyle: string) => {
            if (container?.style) {
                container.style.cursor = cursorStyle;
            }
        };

        // Set cursor based on interaction mode and edit state
        if (interactionMode === 'label') {
            setCursor('crosshair');
        } else if (interactionMode === 'view') {
            setCursor('grab');
        } else if (interactionMode === 'edit') {
            // Handle edit mode cursors
            if (editState.mode === 'addEdge') {
                // Precision pointer for adding edges (crosshair with better precision)
                setCursor('crosshair');
            } else if (editState.mode === 'modify') {
                // Hand cursor for modify mode
                setCursor('pointer');
            } else {
                setCursor('default');
            }
        } else {
            setCursor('default');
        }

        const handleDragStart = () => {
            if (interactionMode === 'view') {
                setCursor('grabbing');
            }
        };

        const handleDragEnd = () => {
            if (interactionMode === 'view') {
                setCursor('grab');
            } else if (interactionMode === 'edit') {
                // Restore edit mode cursor after drag
                if (editState.mode === 'addEdge') {
                    setCursor('crosshair');
                } else if (editState.mode === 'modify') {
                    setCursor('pointer');
                }
            }
        };

        stage.on('dragstart', handleDragStart);
        stage.on('dragend', handleDragEnd);

        return () => {
            if (stage) {
                stage.off('dragstart', handleDragStart);
                stage.off('dragend', handleDragEnd);
            }
            setCursor('default');
            
            // Clear any pending timeout on unmount
            if (interactionTimeoutRef.current !== null) {
                window.clearTimeout(interactionTimeoutRef.current);
            }
        };
    }, [interactionMode, editState.mode]);

    // Effect to handle global zoom events from hotkeys
    useEffect(() => {
        globalEventBus.on('zoom-in', zoomIn);
        globalEventBus.on('zoom-out', zoomOut);

        // Cleanup listeners on unmount
        return () => {
            globalEventBus.off('zoom-in', zoomIn);
            globalEventBus.off('zoom-out', zoomOut);
            
            // Clear any pending timeout on unmount
            if (interactionTimeoutRef.current !== null) {
                window.clearTimeout(interactionTimeoutRef.current);
            }
        };
    }, [zoomIn, zoomOut]); // Depend on the memoized zoom functions

    // Enhanced keyboard event handler
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Track Ctrl key state
            if (e.key === 'Control') {
                setIsCtrlPressed(true);
            }

            if (interactionMode !== 'edit') return;

            // Handle Delete key for both single and multiple selection
            if ((e.key === 'Delete' || e.key === 'Backspace')) {
                e.preventDefault();
                e.stopPropagation();
                
                // Prevent rapid multiple deletions
                if (editState.isDraggingNode) {
                    console.log('Ignoring delete while dragging');
                    return;
                }
                
                // Check for multiple selection first
                if (editState.selectedNodes.length > 0) {
                    console.log(`Delete key pressed for ${editState.selectedNodes.length} selected nodes`);
                    try {
                        deleteSelectedNodes();
                    } catch (error) {
                        console.error('Error during batch node deletion:', error);
                    }
                    return;
                }
                
                // Handle single node deletion
                if (editState.selectedNode !== null) {
                    const nodeToDelete = editState.selectedNode;
                    console.log(`Delete key pressed for node ${nodeToDelete}`);
                    
                    try {
                        deleteNodeWithLogic(nodeToDelete);
                    } catch (error) {
                        console.error('Error during node deletion:', error);
                    }
                    return;
                }
            }

            // Handle Escape key (cancel current operation)
            if (e.key === 'Escape') {
                e.preventDefault();
                if (editState.isDrawing) {
                    cancelDrawing();
                } else if (editState.isRectangleSelecting) {
                    setIsRectangleSelecting(false);
                    setRectangleSelection(null);
                } else {
                    setSelectedNode(null);
                    clearSelectedNodes();
                }
                return;
            }

            // Handle Ctrl+Z (undo)
            if (e.ctrlKey && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                try {
                    undo();
                } catch (error) {
                    console.error('Error during undo:', error);
                }
                return;
            }

            // Handle Ctrl+Y or Ctrl+Shift+Z (redo)
            if ((e.ctrlKey && e.key === 'y') || (e.ctrlKey && e.shiftKey && e.key === 'z')) {
                e.preventDefault();
                try {
                    redo();
                } catch (error) {
                    console.error('Error during redo:', error);
                }
                return;
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            // Track Ctrl key state
            if (e.key === 'Control') {
                setIsCtrlPressed(false);
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keyup', handleKeyUp);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.removeEventListener('keyup', handleKeyUp);
        };
    }, [interactionMode, editState.selectedNode, editState.selectedNodes, editState.isDrawing, editState.isDraggingNode, editState.isRectangleSelecting, 
        deleteNodeWithLogic, deleteSelectedNodes, cancelDrawing, setSelectedNode, clearSelectedNodes, undo, redo, setIsRectangleSelecting, setRectangleSelection]);

    // Optimized road network rendering
    const optimizedRoadNetwork = useMemo(() => {
        if (!image || !roadNetwork?.features || !layerVisibility.network) return null;
        
        return (
            <OptimizedRoadNetworkLayer
                roadNetwork={roadNetwork}
                stageRef={stageRef}
                stageScale={stageScale}
                stagePos={stagePos}
                interactionMode={interactionMode}
                isInteracting={isInteracting}
                imageWidth={image.width}
                imageHeight={image.height}
                visible={layerVisibility.network}
            />
        );
    }, [roadNetwork, stageScale, stagePos, interactionMode, isInteracting, image, layerVisibility.network]);

    // Memoize the prompts rendering
    const memoizedPrompts = useMemo(() => {
        if (!prompts.length || !layerVisibility.prompts) return null;
        
        return (
            <>
                {prompts.map((prompt) => (
                    <Circle
                        key={`prompt-${prompt.id}`}
                        x={prompt.x}
                        y={prompt.y}
                        radius={5 / stageScale}
                        fill={prompt.type === 'positive' ? '#4ade80' : '#f87171'}
                        stroke="black"
                        strokeWidth={1 / stageScale}
                        opacity={0.85}
                        listening={interactionMode !== 'label'}
                    />
                ))}
            </>
        );
    }, [prompts, stageScale, interactionMode, layerVisibility.prompts]);

    // Optimized mask rendering with pre-rendered images
    const memoizedRoadMask = useMemo(() => {
        if (!layerVisibility.roadMask || !maskRenderState.roadMaskImage || !image) return null;
        
        return (
            <KonvaImage
                image={maskRenderState.roadMaskImage}
                width={image.width}
                height={image.height}
                opacity={0.7}
                listening={false}
            />
        );
    }, [maskRenderState.roadMaskImage, image, layerVisibility.roadMask]);

    const memoizedKeypointMask = useMemo(() => {
        if (!layerVisibility.keypointMask || !maskRenderState.keypointMaskImage || !image) return null;
        
        return (
            <KonvaImage
                image={maskRenderState.keypointMaskImage}
                width={image.width}
                height={image.height}
                opacity={0.7}
                listening={false}
            />
        );
    }, [maskRenderState.keypointMaskImage, image, layerVisibility.keypointMask]);

    // Memoized minimap component to reduce unnecessary re-renders and calculations
    const MinimapComponent = useMemo(() => {
        // If minimap is not visible, or required data is missing, return null
        // This ensures we don't waste resources on calculations when minimap is hidden
        if (!minimapVisible || !image || imageLoadingState.isLoading || imageLoadingState.error || !containerDims) {
            return null;
        }

        return (
            <Minimap
                mainStageRef={stageRef}
                mainImage={image}
                mainContainerWidth={containerDims.width}
                mainContainerHeight={containerDims.height}
                onZoom={handleMinimapZoom}
                prompts={prompts}
                roadNetwork={roadNetwork || undefined}
                layerVisibility={layerVisibility}
                isVisible={minimapVisible}
                // Optional: override default minimap size
                // minimapWidth={200} 
                // minimapHeight={150}
            />
        );
    }, [
        minimapVisible,
        image, 
        imageLoadingState, 
        containerDims, 
        prompts, 
        roadNetwork, 
        layerVisibility, 
        handleMinimapZoom
    ]);

    return (
        <div ref={containerRef} className="flex-grow bg-gray-100 relative overflow-hidden">
            {imageLoadingState.isLoading && (
                <div className="absolute inset-0 flex items-center justify-center text-gray-600">
                    Loading image...
                </div>
            )}
            
            {imageLoadingState.error && (
                <div className="absolute inset-0 flex items-center justify-center text-red-500">
                    Failed to load image: {imageLoadingState.error}
                </div>
            )}

            <DownloadProgressOverlay
                visible={imageLoadingState.isChunkedLoading}
                percentage={imageLoadingState.chunkedProgress || 0}
                downloadSpeedBytesPerSec={imageLoadingState.downloadSpeed || 0}
                estimatedTimeRemainingSec={imageLoadingState.estimatedTimeRemaining || 0}
                message="Loading Image"
            />

            {/* Save Success Notification */}
            <SaveNotification 
                show={saveSuccess} 
                onDismiss={resetSaveSuccess}
            />

            {/* Mask Processing Notification */}
            {(isMaskProcessing || maskRenderState.isRenderingMasks) && (
                <div className="absolute top-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg z-50">
                    <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        <span>
                            {isMaskProcessing ? 'Processing masks...' : 'Rendering masks...'}
                            {maskRenderState.renderProgress > 0 && ` ${maskRenderState.renderProgress}%`}
                        </span>
                    </div>
                </div>
            )}

            <Stage
                width={containerRef.current?.clientWidth ?? window.innerWidth}
                height={containerRef.current?.clientHeight ?? window.innerHeight}
                draggable={interactionMode === 'view' && !editState.isDraggingNode}
                onDragStart={handleDragStart}
                onDragEnd={handleDragEnd}
                onWheel={handleWheel}
                onClick={handleClick}
                onContextMenu={handleRightClick}
                onMouseMove={handleMouseMoveForSelection}
                onMouseDown={handleMouseDown}
                onMouseUp={handleMouseUp}
                scaleX={stageScale}
                scaleY={stageScale}
                x={stagePos.x}
                y={stagePos.y}
                ref={stageRef}
            >
                <Layer name="image-layer" visible={layerVisibility.image}>
                    {image && !imageLoadingState.isLoading && !imageLoadingState.error && (
                        <KonvaImage
                            ref={imageNodeRef}
                            name="main-image"
                            image={image}
                            width={image.width}
                            height={image.height}
                        />
                    )}
                </Layer>

                <Layer name="prompt-layer" visible={layerVisibility.prompts}>
                    {memoizedPrompts}
                </Layer>

                <Layer name="road-network-layer" visible={layerVisibility.network}>
                    {optimizedRoadNetwork}
                </Layer>

                <Layer name="road-mask-layer" visible={layerVisibility.roadMask}>
                    {memoizedRoadMask}
                </Layer>

                <Layer name="keypoint-mask-layer" visible={layerVisibility.keypointMask}>
                    {memoizedKeypointMask}
                </Layer>

                <Layer name="patch-grid-layer" visible={layerVisibility.patchGrid}>
                    {image && !imageLoadingState.isLoading && !imageLoadingState.error && (
                        <PatchGridOverlay
                            imageWidth={image.width}
                            imageHeight={image.height}
                            visible={layerVisibility.patchGrid}
                        />
                    )}
                </Layer>

                {/* Edit Mode Visual Feedback Layer */}
                <Layer name="edit-feedback-layer" visible={interactionMode === 'edit'}>
                    {interactionMode === 'edit' && roadNetwork && (
                        <>
                            {/* Rectangle Selection Visual Feedback */}
                            {editState.isRectangleSelecting && editState.rectangleSelection && (() => {
                                const rect = editState.rectangleSelection;
                                const x = Math.min(rect.start.x, rect.end.x);
                                const y = Math.min(rect.start.y, rect.end.y);
                                const width = Math.abs(rect.end.x - rect.start.x);
                                const height = Math.abs(rect.end.y - rect.start.y);
                                
                                // Skip rendering very small rectangles for performance
                                if (width < 5 || height < 5) return null;
                                
                                return (
                                    <Rect
                                        x={x}
                                        y={y}
                                        width={width}
                                        height={height}
                                        stroke="#2563eb"
                                        strokeWidth={2 / stageScale}
                                        fill="rgba(37, 99, 235, 0.1)"
                                        dash={[5 / stageScale, 5 / stageScale]}
                                        listening={false}
                                        perfectDrawEnabled={false}
                                        shadowForStrokeEnabled={false}
                                    />
                                );
                            })()}

                            {/* Multi-Selected Nodes Highlight */}
                            {editState.selectedNodes.length > 0 && editState.selectedNodes.map(nodeId => {
                                const node = roadNetwork.nodes.find(n => n.id === nodeId);
                                if (!node) return null;
                                
                                return (
                                    <React.Fragment key={`multi-selected-${nodeId}`}>
                                        {/* Outer ring for multi-selection */}
                                        <Circle
                                            x={node.x}
                                            y={node.y}
                                            radius={12 / stageScale}
                                            stroke="#dc2626"
                                            strokeWidth={3 / stageScale}
                                            fill="rgba(220, 38, 38, 0.2)"
                                            listening={false}
                                        />
                                        {/* Inner selection indicator */}
                                        <Circle
                                            x={node.x}
                                            y={node.y}
                                            radius={6 / stageScale}
                                            fill="#dc2626"
                                            stroke="white"
                                            strokeWidth={1 / stageScale}
                                            listening={false}
                                        />
                                    </React.Fragment>
                                );
                            })}

                            {/* Snapping Visual Feedback for AddEdge mode */}
                            {snappingEnabled && editState.mode === 'addEdge' && editState.previewPoint && (() => {
                                // Find if preview point is snapped to an existing node
                                const snapNode = getNearestSnapNode(
                                    editState.previewPoint,
                                    roadNetwork.nodes,
                                    snappingDistance
                                );
                                
                                if (snapNode) {
                                    return (
                                        <Circle
                                            key={`snap-indicator-addedge-${snapNode.id}`}
                                            x={snapNode.x}
                                            y={snapNode.y}
                                            radius={snappingDistance / stageScale}
                                            stroke="#ff6b6b"
                                            strokeWidth={2 / stageScale}
                                            fill="transparent"
                                            dash={[3 / stageScale, 3 / stageScale]}
                                            listening={false}
                                            opacity={0.7}
                                        />
                                    );
                                }
                                return null;
                            })()}

                            {/* Snapping Visual Feedback for Node Movement */}
                            {snappingEnabled && editState.mode === 'modify' && draggingNodePosition && (() => {
                                // Find if dragging node can snap to an existing node
                                const snapNode = getNearestSnapNode(
                                    { x: draggingNodePosition.x, y: draggingNodePosition.y },
                                    roadNetwork.nodes,
                                    snappingDistance,
                                    draggingNodePosition.nodeId // Exclude the node being dragged
                                );
                                
                                if (snapNode) {
                                    // Find all nodes that would be merged
                                    const nodesAtPosition = findNodesAtPosition(
                                        { x: snapNode.x, y: snapNode.y },
                                        roadNetwork.nodes,
                                        0.1
                                    );
                                    
                                    return (
                                        <React.Fragment key={`snap-indicator-move-${snapNode.id}`}>
                                            {/* Main snap square */}
                                            <Rect
                                                x={snapNode.x - 10 / stageScale}
                                                y={snapNode.y - 10 / stageScale}
                                                width={15 / stageScale}
                                                height={15 / stageScale}
                                                stroke="#df15b6"
                                                strokeWidth={2.0 / stageScale}
                                                fill="transparent"
                                                dash={[3 / stageScale, 3 / stageScale]}
                                                listening={false}
                                                opacity={1.0}
                                            />
                                            {/* Merge indicator for multiple nodes */}
                                            {nodesAtPosition.length > 1 && (
                                                <Rect
                                                    x={snapNode.x - (snappingDistance * 0.6) / stageScale}
                                                    y={snapNode.y - (snappingDistance * 0.6) / stageScale}
                                                    width={(snappingDistance * 1.2) / stageScale}
                                                    height={(snappingDistance * 1.2) / stageScale}
                                                    stroke="#df15b6"
                                                    strokeWidth={1.5 / stageScale}
                                                    fill="rgba(223, 21, 182, 0.80)"
                                                    listening={false}
                                                    opacity={0.9}
                                                />
                                            )}
                                        </React.Fragment>
                                    );
                                }
                                return null;
                            })()}

                            {/* AddEdge Mode: Preview Line */}
                            {editState.mode === 'addEdge' && editState.isDrawing && editState.currentPolyline.length > 0 && editState.previewPoint && (
                                <KonvaLine
                                    points={[
                                        editState.currentPolyline[editState.currentPolyline.length - 1].x,
                                        editState.currentPolyline[editState.currentPolyline.length - 1].y,
                                        editState.previewPoint.x,
                                        editState.previewPoint.y
                                    ]}
                                    stroke="#3b82f6"
                                    strokeWidth={2 / stageScale}
                                    dash={[5 / stageScale, 5 / stageScale]}
                                    listening={false}
                                />
                            )}

                            {/* AddEdge Mode: Current Polyline */}
                            {editState.mode === 'addEdge' && editState.currentPolyline.length > 1 && (
                                <KonvaLine
                                    points={editState.currentPolyline.flatMap(point => [point.x, point.y])}
                                    stroke="#3b82f6"
                                    strokeWidth={3 / stageScale}
                                    listening={false}
                                />
                            )}

                            {/* AddEdge Mode: Polyline Points */}
                            {editState.mode === 'addEdge' && editState.currentPolyline.map((point, index) => (
                                <Circle
                                    key={`polyline-point-${index}`}
                                    x={point.x}
                                    y={point.y}
                                    radius={4 / stageScale}
                                    fill="#3b82f6"
                                    stroke="white"
                                    strokeWidth={1 / stageScale}
                                    listening={false}
                                />
                            ))}

                            {/* Modify Mode: Hovered Edge Highlight */}
                            {editState.mode === 'modify' && editState.hoveredEdge !== null && (() => {
                                const edge = roadNetwork.edges.find(e => e.id === editState.hoveredEdge);
                                if (!edge) return null;
                                const startNode = roadNetwork.nodes.find(n => n.id === edge.source);
                                const endNode = roadNetwork.nodes.find(n => n.id === edge.target);
                                if (!startNode || !endNode) return null;
                                const midpoint = getEdgeMidpoint(edge, roadNetwork.nodes);
                                if (!midpoint) return null;
                                
                                return (
                                    <React.Fragment key={`edge-highlight-${edge.id}`}>
                                        {/* Highlight the edge */}
                                        <KonvaLine
                                            points={[startNode.x, startNode.y, endNode.x, endNode.y]}
                                            stroke="#f59e0b"
                                            strokeWidth={6 / stageScale}
                                            listening={false}
                                        />
                                        {/* Show add point indicator at midpoint */}
                                        <Circle
                                            x={midpoint.x}
                                            y={midpoint.y}
                                            radius={8 / stageScale}
                                            fill="#f59e0b"
                                            stroke="white"
                                            strokeWidth={2 / stageScale}
                                            listening={false}
                                        />
                                        {/* Plus sign indicator */}
                                        <KonvaLine
                                            points={[midpoint.x - 4/stageScale, midpoint.y, midpoint.x + 4/stageScale, midpoint.y]}
                                            stroke="white"
                                            strokeWidth={2 / stageScale}
                                            listening={false}
                                        />
                                        <KonvaLine
                                            points={[midpoint.x, midpoint.y - 4/stageScale, midpoint.x, midpoint.y + 4/stageScale]}
                                            stroke="white"
                                            strokeWidth={2 / stageScale}
                                            listening={false}
                                        />
                                    </React.Fragment>
                                );
                            })()}

                            {/* Modify Mode: Hovered Node Highlight */}
                            {editState.mode === 'modify' && editState.hoveredNode !== null && (() => {
                                const node = roadNetwork.nodes.find(n => n.id === editState.hoveredNode);
                                if (!node) return null;
                                
                                return (
                                    <Circle
                                        key={`node-hover-${node.id}`}
                                        x={node.x}
                                        y={node.y}
                                        radius={8 / stageScale}
                                        stroke="#ef4444"
                                        strokeWidth={1.5 / stageScale}
                                        fill="transparent"
                                        listening={false}
                                    />
                                );
                            })()}

                            {/* Modify Mode: Selected Node Highlight with optimized dragging */}
                            {editState.mode === 'modify' && editState.selectedNode !== null && (() => {
                                const node = roadNetwork.nodes.find(n => n.id === editState.selectedNode);
                                if (!node) return null;
                                
                                return (
                                    <Circle
                                        key={`node-selected-${node.id}`}
                                        x={node.x}
                                        y={node.y}
                                        radius={8 / stageScale}
                                        stroke="#10b981"
                                        strokeWidth={2 / stageScale}
                                        fill="rgba(16, 185, 129, 0.2)"
                                        draggable={true}
                                        onDragStart={(e) => {
                                            e.cancelBubble = true;
                                            
                                            // Record undo state before starting drag
                                            pushToUndoStack();
                                            
                                            setIsDraggingNode(true);
                                            const startPos = e.target.position();
                                            setDraggingNodePosition({
                                                nodeId: editState.selectedNode!,
                                                x: startPos.x,
                                                y: startPos.y
                                            });
                                        }}
                                        onDragMove={(e) => {
                                            e.cancelBubble = true;
                                            const newPos = e.target.position();
                                            
                                            // Update dragging position for snapping feedback
                                            setDraggingNodePosition({
                                                nodeId: editState.selectedNode!,
                                                x: newPos.x,
                                                y: newPos.y
                                            });
                                            
                                            // Use regular moveNode without recording undo during drag
                                            // Only update position, don't trigger heavy operations
                                            moveNode(editState.selectedNode!, newPos.x, newPos.y, false);
                                        }}
                                        onDragEnd={(e) => {
                                            e.cancelBubble = true;
                                            const newPos = e.target.position();
                                            
                                            // Clear dragging position
                                            setDraggingNodePosition(null);
                                            
                                            // Apply snapping and merging when drag ends
                                            moveNodeWithSnapping(editState.selectedNode!, newPos.x, newPos.y);
                                            setIsDraggingNode(false);
                                        }}
                                        listening={true}
                                        perfectDrawEnabled={false}
                                        shadowForStrokeEnabled={false}
                                    />
                                );
                            })()}
                        </>
                    )}
                </Layer>

            </Stage>

            {/* Main unified toolbar */}
            <MainToolBar
                zoomIn={zoomIn}
                zoomOut={zoomOut}
            />

            {/* Status overlay for coordinates and legend */}
            <StatusOverlay
                stageRef={stageRef}
                imageDimensions={image ? { width: image.width, height: image.height } : null}
            />

            {/* Minimap component - render through memoized component */}
            {MinimapComponent}
        </div>
    );
};

export default React.memo(CanvasArea);