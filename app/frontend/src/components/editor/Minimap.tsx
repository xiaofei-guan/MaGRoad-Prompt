import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Circle, Line as KonvaLine } from 'react-konva';
import Konva from 'konva';
import { Stage as StageType } from 'konva/lib/Stage';
import { Prompt, RoadNetworkData } from '../../types';
import { LayerVisibility } from '../../store/uiStore';
import PatchGridOverlay from './PatchGridOverlay';

interface MinimapProps {
    mainStageRef: React.RefObject<StageType | null>;
    mainImage: HTMLImageElement | undefined;
    mainContainerWidth: number | undefined;
    mainContainerHeight: number | undefined;
    minimapWidth?: number;
    minimapHeight?: number;
    onZoom?: (direction: 'in' | 'out', x: number, y: number) => void;
    prompts?: Prompt[]; // Prompts to display on the minimap
    roadNetwork?: RoadNetworkData; // Road network to display on the minimap
    layerVisibility: LayerVisibility; // Layer visibility settings
    isVisible: boolean; // Whether the minimap is visible
}

const MINIMAP_DEFAULT_WIDTH = 125;
const MINIMAP_DEFAULT_HEIGHT = 125;
const MINIMAP_PADDING = 5; // 5px padding on all sides

function Minimap({
    mainStageRef,
    mainImage,
    mainContainerWidth,
    mainContainerHeight,
    minimapWidth = MINIMAP_DEFAULT_WIDTH,
    minimapHeight = MINIMAP_DEFAULT_HEIGHT,
    onZoom,
    prompts = [],
    roadNetwork,
    layerVisibility,
    isVisible,
}: MinimapProps) {

    const minimapStageRef = useRef<StageType>(null);
    const [viewRect, setViewRect] = useState({ x: 0, y: 0, width: 0, height: 0 });
    const [scale, setScale] = useState(1);
    const [imagePos, setImagePos] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    // Add state to track if the main stage is being interacted with
    const [isMainStageInteracting, setIsMainStageInteracting] = useState(false);
    // Add state to force re-render when interaction ends
    const [interactionEndCounter, setInteractionEndCounter] = useState(0);
    // Add a timeout ref to manage interaction state
    const interactionTimeoutRef = useRef<number | null>(null);
    // Add a ref to store the mutation observer
    const observerRef = useRef<MutationObserver | null>(null);

    // Calculate minimap scale and image position with padding
    useEffect(() => {
        if (!mainImage) return;

        // Calculate available space after applying padding
        const availableWidth = minimapWidth - (MINIMAP_PADDING * 2);
        const availableHeight = minimapHeight - (MINIMAP_PADDING * 2);
        
        // Calculate scale to fit image within the available space
        const scaleX = availableWidth / mainImage.width;
        const scaleY = availableHeight / mainImage.height;
        const newScale = Math.min(scaleX, scaleY);
        setScale(newScale);

        // Center the scaled image within the minimap with padding
        const imgWidth = mainImage.width * newScale;
        const imgHeight = mainImage.height * newScale;
        setImagePos({
            x: MINIMAP_PADDING + (availableWidth - imgWidth) / 2,
            y: MINIMAP_PADDING + (availableHeight - imgHeight) / 2,
        });

    }, [mainImage, minimapWidth, minimapHeight]);

    // Update viewport rectangle on main stage changes
    useEffect(() => {
        // Skip all event handling if minimap is not visible
        if (!isVisible) return;

        const mainStage = mainStageRef.current;
        if (!mainStage || !mainContainerWidth || !mainContainerHeight || !mainImage || scale === 0) {
            return;
        }

        const updateViewRect = () => {
            const mainScale = mainStage.scaleX(); // Assuming uniform scale
            const mainPos = mainStage.position();

            // Calculate the viewport boundaries relative to the main image (top-left corner is 0,0)
            const viewX1 = -mainPos.x / mainScale;
            const viewY1 = -mainPos.y / mainScale;
            const viewWidth = mainContainerWidth / mainScale;
            const viewHeight = mainContainerHeight / mainScale;

            // Convert main image coordinates to minimap coordinates
            setViewRect({
                x: viewX1 * scale + imagePos.x, // Apply minimap scale and image offset
                y: viewY1 * scale + imagePos.y,
                width: viewWidth * scale,
                height: viewHeight * scale,
            });
        };

        // Set interaction flag when dragging or zooming starts
        const handleInteractionStart = () => {
            setIsMainStageInteracting(true);
            
            // Clear any existing timeout
            if (interactionTimeoutRef.current !== null) {
                window.clearTimeout(interactionTimeoutRef.current);
                interactionTimeoutRef.current = null;
            }
        };

        // Reset interaction flag after a delay when interaction ends
        const handleInteractionEnd = () => {
            if (interactionTimeoutRef.current !== null) {
                window.clearTimeout(interactionTimeoutRef.current);
            }
            
            interactionTimeoutRef.current = window.setTimeout(() => {
                setIsMainStageInteracting(false);
                setInteractionEndCounter(prev => prev + 1); // Force re-render when interaction ends
                interactionTimeoutRef.current = null;
            }, 300); // 300ms delay before rendering full detail again
        };

        // Initial update
        updateViewRect();

        // Listen for changes on the main stage
        mainStage.on('dragstart', handleInteractionStart);
        mainStage.on('dragmove', updateViewRect);
        mainStage.on('wheel', handleInteractionStart);
        mainStage.on('wheel', updateViewRect);
        mainStage.on('dragend', updateViewRect);
        mainStage.on('dragend', handleInteractionEnd);
        mainStage.on('scaleChange', updateViewRect); // Listen for custom scaleChange event

        return () => {
            mainStage.off('dragstart', handleInteractionStart);
            mainStage.off('dragmove', updateViewRect);
            mainStage.off('wheel', handleInteractionStart);
            mainStage.off('wheel', updateViewRect);
            mainStage.off('dragend', updateViewRect);
            mainStage.off('dragend', handleInteractionEnd);
            mainStage.off('scaleChange', updateViewRect);
            
            // Clear any pending timeout on unmount
            if (interactionTimeoutRef.current !== null) {
                window.clearTimeout(interactionTimeoutRef.current);
                interactionTimeoutRef.current = null;
            }
        };

    }, [
        mainStageRef,
        mainContainerWidth,
        mainContainerHeight,
        mainImage, // Re-run if image changes
        scale,     // Re-run if minimap scale changes
        imagePos,  // Re-run if image position on minimap changes
        isVisible  // Re-run if visibility changes
    ]);

    // Force update viewRect when main stage scale changes (for zoom buttons)
    useEffect(() => {
        // Skip if minimap is not visible
        if (!isVisible) {
            // Clean up any existing observer
            if (observerRef.current) {
                observerRef.current.disconnect();
                observerRef.current = null;
            }
            return;
        }

        const mainStage = mainStageRef.current;
        if (!mainStage || !mainContainerWidth || !mainContainerHeight || !mainImage || scale === 0) {
            return;
        }

        // Create a mutation observer to watch for style changes on the stage
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.attributeName === 'style') {
                    // A style attribute changed - likely a scale change
                    const mainScale = mainStage.scaleX(); // Assuming uniform scale
                    const mainPos = mainStage.position();

                    // Calculate the viewport boundaries
                    const viewX1 = -mainPos.x / mainScale;
                    const viewY1 = -mainPos.y / mainScale;
                    const viewWidth = mainContainerWidth / mainScale;
                    const viewHeight = mainContainerHeight / mainScale;

                    // Update viewRect
                    setViewRect({
                        x: viewX1 * scale + imagePos.x,
                        y: viewY1 * scale + imagePos.y,
                        width: viewWidth * scale,
                        height: viewHeight * scale,
                    });
                    
                    // Set interaction flag when scale changes
                    setIsMainStageInteracting(true);
                    
                    // Clear any existing timeout
                    if (interactionTimeoutRef.current !== null) {
                        window.clearTimeout(interactionTimeoutRef.current);
                    }
                    
                    // Reset interaction flag after a delay
                    interactionTimeoutRef.current = window.setTimeout(() => {
                        setIsMainStageInteracting(false);
                        setInteractionEndCounter(prev => prev + 1); // Force re-render when interaction ends
                        interactionTimeoutRef.current = null;
                    }, 300);
                }
            }
        });

        // Store the observer in the ref
        observerRef.current = observer;

        // Start observing
        observer.observe(mainStage.container(), { 
            attributes: true, 
            attributeFilter: ['style'] 
        });

        return () => {
            // Disconnect the observer when cleaning up
            observer.disconnect();
            observerRef.current = null;
            
            // Clear any pending timeout on unmount
            if (interactionTimeoutRef.current !== null) {
                window.clearTimeout(interactionTimeoutRef.current);
                interactionTimeoutRef.current = null;
            }
        };
    }, [
        mainStageRef,
        mainContainerWidth,
        mainContainerHeight,
        mainImage,
        scale,
        imagePos,
        isVisible // Re-run if visibility changes
    ]);

    // Handle dragging the viewport rectangle
    const handleRectDrag = (e: Konva.KonvaEventObject<DragEvent>) => {
        const mainStage = mainStageRef.current;
        if (!mainStage || !mainImage) return;

        const rect = e.target;
        
        // Convert minimap rectangle position to main image coordinate space
        // First, calculate position relative to the minimap image origin
        const rectX = rect.x();
        const rectY = rect.y();
        const relX = (rectX - imagePos.x) / scale;
        const relY = (rectY - imagePos.y) / scale;
        
        // Set main stage position to show this area
        // Note: we negate because dragging right should move the view left
        const newMainX = -relX * mainStage.scaleX();
        const newMainY = -relY * mainStage.scaleY();
        
        mainStage.position({ x: newMainX, y: newMainY });
        mainStage.fire('dragend'); // Fire event to update other components
    };

    // Handle wheel event on minimap for zooming
    const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
        e.evt.preventDefault();
        
        if (!onZoom || !minimapStageRef.current) return;
        
        // Get pointer position relative to minimap
        const pointer = minimapStageRef.current.getPointerPosition();
        if (!pointer) return;
        
        // Determine zoom direction
        const direction = e.evt.deltaY < 0 ? 'in' : 'out';
        
        // Convert minimap coordinates to main canvas coordinates
        const mainStage = mainStageRef.current;
        if (!mainStage || !mainImage) return;
        
        // Calculate the position in the main image coordinates
        const relX = (pointer.x - imagePos.x) / scale;
        const relY = (pointer.y - imagePos.y) / scale;
        
        // Call the onZoom callback with the zoom direction and coordinates
        onZoom(direction, relX, relY);
    };

    // Memoize the road network rendering to prevent unnecessary recalculations
    const memoizedRoadNetwork = useMemo(() => {
        if (!roadNetwork || !layerVisibility.network) return null;
        
        // During interaction, render a simplified version (just lines, no points)
        const shouldSimplify = isMainStageInteracting;
        
        // Log for debugging
        console.log('Minimap: Rendering road network, shouldSimplify =', shouldSimplify);
        
        return (
            <>
                {roadNetwork.features.map((feature, featureIndex) => {
                    if (feature.geometry.type === 'LineString') {
                        const points = feature.geometry.coordinates as [number, number][];
                        const konvaPoints = points.map(point => [
                            point[0] * scale + imagePos.x, 
                            point[1] * scale + imagePos.y
                        ]).flat();

                        return (
                            <React.Fragment key={`minimap-road-feature-${featureIndex}`}>
                                <KonvaLine
                                    points={konvaPoints}
                                    stroke="#4c88d6" // Same color as in CanvasArea
                                    strokeWidth={1}
                                    lineCap="round"
                                    lineJoin="round"
                                    opacity={0.95}
                                    listening={false}
                                />
                                {/* Only render points when not interacting */}
                                {!shouldSimplify && points.map((point, pointIndex) => (
                                    <Circle
                                        key={`minimap-road-point-${featureIndex}-${pointIndex}`}
                                        x={point[0] * scale + imagePos.x}
                                        y={point[1] * scale + imagePos.y}
                                        radius={1}
                                        fill="#eedd5e" // Same color as in CanvasArea
                                        stroke="black"
                                        strokeWidth={0.5}
                                        opacity={1.0}
                                        listening={false}
                                    />
                                ))}
                            </React.Fragment>
                        );
                    }
                    return null;
                })}
            </>
        );
    }, [roadNetwork, scale, imagePos, layerVisibility.network, isMainStageInteracting, interactionEndCounter]);

    // Memoize the prompts rendering
    const memoizedPrompts = useMemo(() => {
        if (!prompts.length || !layerVisibility.prompts) return null;
        
        return prompts.map((prompt) => (
            <Circle
                key={`minimap-prompt-${prompt.id}`}
                x={prompt.x * scale + imagePos.x}
                y={prompt.y * scale + imagePos.y}
                radius={2}
                fill={prompt.type === 'positive' ? '#4ade80' : '#f87171'} // Same colors as main canvas
                stroke="black"
                strokeWidth={0.5}
                opacity={0.85}
                listening={false} // No interaction needed on minimap prompts
            />
        ));
    }, [prompts, scale, imagePos, layerVisibility.prompts]);

    // Don't render anything if not visible or no image
    if (!isVisible || !mainImage) {
        return null; 
    }

    return (
        <div
            className="absolute top-2 right-2 bg-white border border-gray-400 shadow-md overflow-hidden"
            style={{ width: minimapWidth, height: minimapHeight }}
        >
            <Stage
                ref={minimapStageRef}
                width={minimapWidth}
                height={minimapHeight}
                onWheel={handleWheel}
            >
                {/* Image Layer */}
                <Layer visible={layerVisibility.image}>
                    <KonvaImage
                        image={mainImage}
                        x={imagePos.x}
                        y={imagePos.y}
                        width={mainImage.width * scale}
                        height={mainImage.height * scale}
                        listening={false} // Background image doesn't need events
                    />
                </Layer>

                {/* Road Network Layer */}
                <Layer visible={layerVisibility.network}>
                    {memoizedRoadNetwork}
                </Layer>

                {/* Prompts Layer */}
                <Layer visible={layerVisibility.prompts}>
                    {memoizedPrompts}
                </Layer>

                {/* Patch Grid Layer */}
                <Layer visible={layerVisibility.patchGrid}>
                    <PatchGridOverlay
                        imageWidth={mainImage.width}
                        imageHeight={mainImage.height}
                        visible={layerVisibility.patchGrid}
                        patchSize={1024}
                        overlap={128}
                        offsetX={imagePos.x}
                        offsetY={imagePos.y}
                        scaleX={scale}
                        scaleY={scale}
                    />
                </Layer>

                {/* Viewport Rectangle Layer - Always on top and visible */}
                <Layer>
                    <Rect
                        x={viewRect.x}
                        y={viewRect.y}
                        width={viewRect.width}
                        height={viewRect.height}
                        stroke="#1bff17"
                        strokeWidth={2}
                        fill="rgba(12, 201, 78, 0.2)"
                        draggable // Allow dragging the viewport
                        onDragMove={handleRectDrag}
                        onDragStart={() => setIsDragging(true)}
                        onDragEnd={() => setIsDragging(false)}
                        onMouseEnter={(e) => {
                            const container = e.target.getStage()?.container();
                            if (container) {
                                container.style.cursor = 'grab';
                            }
                        }}
                        onMouseLeave={(e) => {
                            const container = e.target.getStage()?.container();
                            if (container) {
                                container.style.cursor = 'default';
                            }
                        }}
                        onMouseDown={(e) => {
                            const container = e.target.getStage()?.container();
                            if (container) {
                                container.style.cursor = 'grabbing';
                            }
                        }}
                        onMouseUp={(e) => {
                            const container = e.target.getStage()?.container();
                            if (container) {
                                container.style.cursor = 'grab';
                            }
                        }}
                        // Allow dragging across entire minimap container, not just image area
                        dragBoundFunc={(pos) => {
                            // Constrain to minimap container boundaries, not image
                            return {
                                x: Math.max(0, Math.min(pos.x, minimapWidth - viewRect.width)),
                                y: Math.max(0, Math.min(pos.y, minimapHeight - viewRect.height)),
                            };
                        }}
                    />
                </Layer>
            </Stage>
        </div>
    );
}

// Use React.memo to prevent unnecessary re-renders of the entire component
export default React.memo(Minimap);
