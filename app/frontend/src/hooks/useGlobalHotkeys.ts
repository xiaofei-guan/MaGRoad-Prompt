import { useEffect } from 'react';
import hotkeys from 'hotkeys-js';
import { useUiStore, InteractionMode } from '../store/uiStore';
import { globalEventBus } from '../utils/eventBus';
import { useImageStore } from '../store/imageStore';
import { useRoadNetworkStore } from '../store/roadNetworkStore';
import { usePromptStore } from '../store/promptStore';

function useGlobalHotkeys() {
    // Select state and actions individually
    const toggleLayerVisibility = useUiStore((state) => state.toggleLayerVisibility);
    const setInteractionMode = useUiStore((state) => state.setInteractionMode);
    const toggleMinimapVisibility = useUiStore((state) => state.toggleMinimapVisibility);

    useEffect(() => {
        // Configure hotkeys to work globally on all elements
        hotkeys.filter = function(event) {
            return true; // Allow all hotkeys to work globally
        };
        
        // Enable capturing hotkeys across all elements
        hotkeys.setScope('all');

        // Add a fallback event listener for Ctrl+V to handle browser conflicts
        const handleKeyDown = (event: KeyboardEvent) => {
            // Handle Ctrl+V specifically for view/edit mode switching
            // IMPORTANT: Check for exact combination to avoid conflicts with paste operations
            if (event.ctrlKey && event.code === 'KeyV' && !event.shiftKey && !event.altKey && !event.metaKey) {
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                
                const currentMode = useUiStore.getState().interactionMode;
                console.log('Fallback Ctrl+V handler triggered, current mode:', currentMode);
                
                // Updated logic: allow switching from any mode, including label
                if (currentMode === 'view') {
                    setInteractionMode('edit');
                    console.log('Fallback: Switched from view to edit mode');
                } else if (currentMode === 'edit') {
                    setInteractionMode('view');
                    console.log('Fallback: Switched from edit to view mode');
                } else if (currentMode === 'label') {
                    // From label mode, go to view mode first
                    setInteractionMode('view');
                    console.log('Fallback: Switched from label to view mode');
                }
            }
        };
        
        // Add event listener with capture to catch events early
        document.addEventListener('keydown', handleKeyDown, { capture: true });

        // --- Layer Visibility --- 
        hotkeys('ctrl+shift+1, command+shift+1', (event) => {
            event.preventDefault(); // Prevent browser default
            toggleLayerVisibility('image');
            return false; // Prevent bubbling
        });
        hotkeys('ctrl+shift+2, command+shift+2', (event) => {
            event.preventDefault();
            toggleLayerVisibility('prompts');
            return false;
        });
        hotkeys('ctrl+shift+3, command+shift+3', (event) => {
            event.preventDefault();
            toggleLayerVisibility('network');
            return false;
        });
        hotkeys('ctrl+shift+4, command+shift+4', (event) => {
            event.preventDefault();
            toggleLayerVisibility('roadMask');
            return false;
        });
        hotkeys('ctrl+shift+5, command+shift+5', (event) => {
            event.preventDefault();
            toggleLayerVisibility('keypointMask');
            return false;
        });
        hotkeys('ctrl+shift+6, command+shift+6', (event) => {
            event.preventDefault();
            toggleLayerVisibility('patchGrid');
            return false;
        });

        // --- View/Edit Mode Cycling (Ctrl+V) - ONLY between view and edit ---
        hotkeys('ctrl+v, command+v', function(event) {
            // Check for exact key combination to avoid conflicts
            if (event.shiftKey || event.altKey) {
                return; // Ignore if other modifiers are pressed
            }
            
            event.preventDefault();
            event.stopPropagation();
            
            const currentMode = useUiStore.getState().interactionMode;
            
            console.log('Hotkeys Ctrl+V pressed, current mode:', currentMode);
            
            // Updated logic: allow switching from any mode, including label
            if (currentMode === 'view') {
                setInteractionMode('edit');
                console.log('Hotkeys: Switched from view to edit mode');
            } else if (currentMode === 'edit') {
                setInteractionMode('view');
                console.log('Hotkeys: Switched from edit to view mode');
            } else if (currentMode === 'label') {
                // From label mode, go to view mode first
                setInteractionMode('view');
                console.log('Hotkeys: Switched from label to view mode');
            }
            return false;
        });



        // --- Label Mode (Ctrl+B) - ONLY activate label mode ---
        hotkeys('ctrl+b, command+b', function(event) {
            event.preventDefault();
            event.stopPropagation();
            
            const currentMode = useUiStore.getState().interactionMode;
            
            console.log('Ctrl+B pressed, current mode:', currentMode);
            
            // Always switch to label mode (no toggle, just activate)
            if (currentMode !== 'label') {
                setInteractionMode('label');
                console.log('Activated label mode');
            }
            return false;
        });



        // --- Edit Tools Toggle (Ctrl+E) - Toggle between AddEdge and Modify ---
        hotkeys('ctrl+e, command+e', function(event) {
            event.preventDefault();
            event.stopPropagation();
            
            const currentMode = useUiStore.getState().interactionMode;
            const { setEditMode, editState } = useRoadNetworkStore.getState();
            
            console.log('Ctrl+E pressed, current edit mode:', editState.mode);
            
            // Only work in edit mode - remove roadNetwork dependency
            // Users should be able to prepare edit tools before having a road network
            if (currentMode === 'edit') {
                // Toggle between addEdge and modify modes (exclusive)
                if (editState.mode === 'addEdge') {
                    setEditMode('modify');
                    console.log('Switched from AddEdge to Modify mode');
                } else {
                    setEditMode('addEdge');
                    console.log('Switched from Modify to AddEdge mode');
                }
            }
            return false;
        });

        // --- Snap Tool (Ctrl+M) - Only in Edit Mode ---
        hotkeys('ctrl+m, command+m', (event) => {
            event.preventDefault();
            const currentMode = useUiStore.getState().interactionMode;
            const { setSnappingEnabled, snappingEnabled } = useRoadNetworkStore.getState();
            
            // Only work in edit mode - remove roadNetwork dependency for consistency
            if (currentMode === 'edit') {
                setSnappingEnabled(!snappingEnabled);
            }
            return false;
        });

        // --- Zoom --- 
        hotkeys('ctrl+=, command+=', (event) => { // Corresponds to Ctrl+Plus
            event.preventDefault(); // Prevent browser zoom
            globalEventBus.emit('zoom-in');
            return false;
        });
        hotkeys('ctrl+-, command+-', (event) => {
            event.preventDefault(); // Prevent browser zoom
            globalEventBus.emit('zoom-out');
            return false;
        });

        // --- Generate Road Network (Ctrl+Shift+Space) ---
        hotkeys('ctrl+shift+space, command+shift+space', (event) => {
            // Ensure this is the exact combination we want
            if (!event.shiftKey) {
                return; // Ignore if shift is not pressed
            }
            
            event.preventDefault();
            const { currentImage } = useImageStore.getState();
            const prompts = usePromptStore.getState().prompts;
            const { generateRoadNetwork, isLoading } = useRoadNetworkStore.getState();
            
            console.log('Ctrl+Shift+Space pressed for road network generation');
            
            // Updated logic to match ActionButtons.tsx - allow generation even without features
            if (currentImage && prompts.length > 0 && !isLoading) {
                generateRoadNetwork(currentImage.id, prompts);
                console.log('Generated road network via keyboard shortcut');
            } else {
                console.log('Cannot generate road network: missing requirements');
            }
            return false;
        });

        // --- Save Road Network (Ctrl+S) ---
        hotkeys('ctrl+s, command+s', (event) => {
            event.preventDefault(); // Prevent browser save dialog
            const { currentImage } = useImageStore.getState();
            const { roadNetwork, saveRoadNetwork, isLoading } = useRoadNetworkStore.getState();
            
            const canSave = currentImage && roadNetwork;
            if (canSave && !isLoading) {
                saveRoadNetwork(currentImage.id);
            }
            return false;
        });

        // --- Compute Features (Ctrl+F) ---
        hotkeys('ctrl+f, command+f', (event) => {
            event.preventDefault(); // Prevent browser search dialog
            const { currentImage, featureStatus, startFeatureComputation } = useImageStore.getState();
            
            if (currentImage && (featureStatus === 'none' || featureStatus === 'error')) {
                startFeatureComputation(currentImage.id);
            }
            return false;
        });

        // --- Toggle Minimap (Ctrl+Shift+M) ---
        hotkeys('ctrl+shift+m, command+shift+m', (event) => {
            event.preventDefault(); // Prevent browser default
            toggleMinimapVisibility();
            return false;
        });

        // --- Image Navigation (Left/Right Arrow Keys) ---
        hotkeys('left', (event) => {
            event.preventDefault();
            // Emit event instead of directly calling store method to ensure save confirmation
            globalEventBus.emit('navigate-to-previous-image');
            return false;
        });

        hotkeys('right', (event) => {
            event.preventDefault();
            // Emit event instead of directly calling store method to ensure save confirmation
            globalEventBus.emit('navigate-to-next-image');
            return false;
        });

        // Cleanup function to unbind all hotkeys
        return () => {
            document.removeEventListener('keydown', handleKeyDown, { capture: true });
            hotkeys.unbind('ctrl+shift+1, command+shift+1');
            hotkeys.unbind('ctrl+shift+2, command+shift+2');
            hotkeys.unbind('ctrl+shift+3, command+shift+3');
            hotkeys.unbind('ctrl+shift+4, command+shift+4');
            hotkeys.unbind('ctrl+shift+5, command+shift+5');
            hotkeys.unbind('ctrl+shift+6, command+shift+6');
            hotkeys.unbind('ctrl+v, command+v');
            hotkeys.unbind('ctrl+b, command+b');
            hotkeys.unbind('ctrl+e, command+e');
            hotkeys.unbind('ctrl+m, command+m');
            hotkeys.unbind('ctrl+=, command+=');
            hotkeys.unbind('ctrl+-, command+-');
            hotkeys.unbind('ctrl+shift+space, command+shift+space');
            hotkeys.unbind('ctrl+s, command+s');
            hotkeys.unbind('ctrl+f, command+f');
            hotkeys.unbind('ctrl+shift+m, command+shift+m');
            hotkeys.unbind('left');
            hotkeys.unbind('right');
        };

        // Depend only on the stable action references
        // The interactionMode is read fresh inside the ctrl+space handler
    }, [toggleLayerVisibility, setInteractionMode, toggleMinimapVisibility]);
}

export default useGlobalHotkeys; 