import React from 'react';
import { useUiStore, InteractionMode } from '../../store/uiStore';
import { useRoadNetworkStore, EditMode } from '../../store/roadNetworkStore';
import ActionButton from './ActionButton';
import { ZoomInIcon, ZoomOutIcon } from './ZoomIcons';

// Props definition for MainToolBar
interface MainToolBarProps {
    zoomIn: () => void;
    zoomOut: () => void;
}

// Main unified toolbar component
const MainToolBar = React.memo(({ 
    zoomIn, 
    zoomOut 
}: MainToolBarProps) => {
    // Select UI state and actions individually
    const interactionMode = useUiStore((state) => state.interactionMode);
    const setInteractionMode = useUiStore((state) => state.setInteractionMode);
    const minimapVisible = useUiStore((state) => state.minimapVisible);
    const toggleMinimapVisibility = useUiStore((state) => state.toggleMinimapVisibility);
    
    // Select edit state and actions
    const editState = useRoadNetworkStore((state) => state.editState);
    const setEditMode = useRoadNetworkStore((state) => state.setEditMode);
    const roadNetwork = useRoadNetworkStore((state) => state.roadNetwork);
    const canUndo = useRoadNetworkStore((state) => state.canUndo);
    const canRedo = useRoadNetworkStore((state) => state.canRedo);
    const undo = useRoadNetworkStore((state) => state.undo);
    const redo = useRoadNetworkStore((state) => state.redo);
    
    // Select snapping state and actions
    const snappingEnabled = useRoadNetworkStore((state) => state.snappingEnabled);
    const snappingDistance = useRoadNetworkStore((state) => state.snappingDistance);
    const setSnappingEnabled = useRoadNetworkStore((state) => state.setSnappingEnabled);
    const setSnappingDistance = useRoadNetworkStore((state) => state.setSnappingDistance);

    // Define colors for different button types
    const modeButtonColor = "bg-[#6d97c4]";
    const modeButtonTextColor = "text-gray-800";
    const modeButtonHoverColor = "hover:bg-gray-400";
    const modeButtonActiveBgColor = "bg-blue-400";
    const modeButtonActiveTextColor = "text-white";

    const zoomButtonColor = "bg-[#82bab1]";
    const zoomButtonTextColor = "text-white";
    const zoomButtonHoverColor = "hover:bg-gray-400";

    const utilityButtonColor = "bg-[#9b6db4]";
    const utilityButtonTextColor = "text-white";
    const utilityButtonHoverColor = "hover:bg-gray-400";
    const utilityButtonActiveBgColor = "bg-purple-500";
    const utilityButtonActiveTextColor = "text-white";

    const editButtonColor = "bg-[#95be76]";
    const editButtonTextColor = "text-black";
    const editButtonHoverColor = "hover:bg-yellow-500";
    const editButtonActiveBgColor = "bg-[#27b692]";
    const editButtonActiveTextColor = "text-white";

    const undoButtonColor = "bg-[#6b7280]";
    const undoButtonTextColor = "text-black";
    const undoButtonHoverColor = "hover:bg-gray-600";

    return (
        <div className="absolute top-2 left-2 pointer-events-auto">
            <div className="flex items-center bg-white bg-opacity-90 rounded-lg shadow-lg p-1 space-x-1">
                {/* Interaction Mode Buttons */}
                <div className="flex items-center border-r border-gray-300 pr-2">
                    <ActionButton
                        title="View Mode (Pan/Zoom) - Ctrl+V to toggle with Edit"
                        bgColor={modeButtonColor}
                        textColor={modeButtonTextColor}
                        hoverBgColor={modeButtonHoverColor}
                        activeBgColor={modeButtonActiveBgColor}
                        activeTextColor={modeButtonActiveTextColor}
                        isActive={interactionMode === 'view'}
                        onClick={() => setInteractionMode('view')}
                    >
                        View
                    </ActionButton>
                    <ActionButton
                        title="Label Mode (Add Points) - Ctrl+L to activate"
                        bgColor={modeButtonColor}
                        textColor={modeButtonTextColor}
                        hoverBgColor={modeButtonHoverColor}
                        activeBgColor={modeButtonActiveBgColor}
                        activeTextColor={modeButtonActiveTextColor}
                        isActive={interactionMode === 'label'}
                        onClick={() => setInteractionMode('label')}
                    >
                        Label
                    </ActionButton>
                    <ActionButton
                        title="Edit Mode (Edit Road Network) - Ctrl+V to toggle with View"
                        bgColor={modeButtonColor}
                        textColor={modeButtonTextColor}
                        hoverBgColor={modeButtonHoverColor}
                        activeBgColor={modeButtonActiveBgColor}
                        activeTextColor={modeButtonActiveTextColor}
                        isActive={interactionMode === 'edit'}
                        onClick={() => setInteractionMode('edit')}
                    >
                        Edit
                    </ActionButton>
                </div>

                {/* Edit Tools (Always visible, but disabled when not in edit mode or no road network) */}
                <div className="flex items-center border-r border-gray-300 pr-2">
                    <ActionButton
                        title="Add Edge - Click to add points and create road connections (Ctrl+E to toggle)"
                        bgColor={editButtonColor}
                        textColor={editButtonTextColor}
                        hoverBgColor={editButtonHoverColor}
                        activeBgColor={editButtonActiveBgColor}
                        activeTextColor={editButtonActiveTextColor}
                        isActive={interactionMode === 'edit' && editState.mode === 'addEdge'}
                        disabled={interactionMode !== 'edit'}
                        onClick={() => {
                            if (interactionMode === 'edit') {
                                // Exclusive behavior: switch to addEdge mode
                                setEditMode('addEdge');
                            }
                        }}
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path 
                                strokeLinecap="round" 
                                strokeLinejoin="round" 
                                strokeWidth={2.0} 
                                d="M4 4l6 6m0 0l6 6m0 0l4 4" 
                            />
                            <circle cx="3" cy="3" r="2.0" fill="currentColor" />
                            <circle cx="12" cy="12" r="2.0" fill="currentColor" />
                            <circle cx="21" cy="21" r="2.0" fill="currentColor" />
                        </svg>
                    </ActionButton>
                    <ActionButton
                        title="Modify Network - Hover edges to add nodes, select nodes to move/delete. Hold Ctrl+Drag for rectangle selection (Ctrl+E to toggle)"
                        bgColor={editButtonColor}
                        textColor={editButtonTextColor}
                        hoverBgColor={editButtonHoverColor}
                        activeBgColor={editButtonActiveBgColor}
                        activeTextColor={editButtonActiveTextColor}
                        isActive={interactionMode === 'edit' && editState.mode === 'modify'}
                        disabled={interactionMode !== 'edit'}
                        onClick={() => {
                            if (interactionMode === 'edit') {
                                // Exclusive behavior: switch to modify mode
                                setEditMode('modify');
                            }
                        }}
                    >
                        🔨
                    </ActionButton>
                    {/* Snapping Controls */}
                    <div className="flex items-center ml-1 space-x-1">
                        {/* Snapping Toggle Button */}
                        <ActionButton
                            title={snappingEnabled ? "Disable Node Snapping" : "Enable Node Snapping"}
                            bgColor={editButtonColor}
                            textColor={editButtonTextColor}
                            hoverBgColor={editButtonHoverColor}
                            activeBgColor={editButtonActiveBgColor}
                            activeTextColor={editButtonActiveTextColor}
                            isActive={interactionMode === 'edit' && snappingEnabled}
                            disabled={interactionMode !== 'edit'}
                            onClick={() => {
                                if (interactionMode === 'edit') {
                                    setSnappingEnabled(!snappingEnabled);
                                }
                            }}
                        >
                            🧲
                        </ActionButton>
                        {/* Snapping Distance Input */}
                        <div className="flex items-center bg-white rounded border border-gray-300 px-1">
                            <input
                                type="number"
                                value={snappingDistance}
                                onChange={(e) => {
                                    if (interactionMode === 'edit') {
                                        const value = parseInt(e.target.value) || 0;
                                        setSnappingDistance(value);
                                    }
                                }}
                                min="0"
                                max="30"
                                className="w-8 text-xs text-center border-0 outline-none bg-transparent text-gray-900"
                                title="Snap distance in pixels"
                                disabled={interactionMode !== 'edit'}
                            />
                        </div>
                    </div>
                </div>

                {/* Undo/Redo Buttons (Only visible in edit mode) */}
                {interactionMode === 'edit' && roadNetwork && (
                    <div className="flex items-center border-r border-gray-300 pr-2">
                        <ActionButton
                            title="Undo (Ctrl+Z)"
                            bgColor={undoButtonColor}
                            textColor={undoButtonTextColor}
                            hoverBgColor={undoButtonHoverColor}
                            onClick={undo}
                            disabled={!canUndo()}
                        >
                            ↶
                        </ActionButton>
                        <ActionButton
                            title="Redo (Ctrl+Y)"
                            bgColor={undoButtonColor}
                            textColor={undoButtonTextColor}
                            hoverBgColor={undoButtonHoverColor}
                            onClick={redo}
                            disabled={!canRedo()}
                        >
                            ↷
                        </ActionButton>
                    </div>
                )}

                {/* Zoom Buttons */}
                <div className="flex items-center border-r border-gray-300 pr-2">
                    <ActionButton
                        title="Zoom In"
                        bgColor={zoomButtonColor}
                        textColor={zoomButtonTextColor}
                        hoverBgColor={zoomButtonHoverColor}
                        onClick={zoomIn}
                    >
                        <ZoomInIcon />
                    </ActionButton>
                    <ActionButton
                        title="Zoom Out"
                        bgColor={zoomButtonColor}
                        textColor={zoomButtonTextColor}
                        hoverBgColor={zoomButtonHoverColor}
                        onClick={zoomOut}
                    >
                        <ZoomOutIcon />
                    </ActionButton>
                </div>

                {/* Utility Buttons */}
                <div className="flex items-center">
                    <ActionButton
                        title={minimapVisible ? "Hide Minimap" : "Show Minimap"}
                        bgColor={utilityButtonColor}
                        textColor={utilityButtonTextColor}
                        hoverBgColor={utilityButtonHoverColor}
                        activeBgColor={utilityButtonActiveBgColor}
                        activeTextColor={utilityButtonActiveTextColor}
                        isActive={minimapVisible}
                        onClick={toggleMinimapVisibility}
                    >
                        <svg 
                            className="w-3 h-3" 
                            fill="none" 
                            viewBox="0 0 24 24" 
                            stroke="currentColor"
                        >
                            <path 
                            strokeLinecap="round" 
                            strokeLinejoin="round" 
                            strokeWidth={2} 
                            d={minimapVisible 
                                ? "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                : "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z M3 3l18 18"
                            } 
                            />
                        </svg>
                    </ActionButton>
                </div>
            </div>
        </div>
    );
});

export default MainToolBar;