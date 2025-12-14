import { useEffect, useRef, useState, useCallback } from 'react';
import './App.css';
import { Header } from './components/layout/Header';
import { ToolBar } from './components/layout/ToolBar';
import { ControlPanel } from './components/layout/ControlPanel';
import CanvasArea from './components/layout/CanvasArea';
import { StatusBar } from './components/layout/StatusBar';
import ResizableSplitter from './components/common/ResizableSplitter';
import useGlobalHotkeys from './hooks/useGlobalHotkeys';
import { useImageStore } from './store/imageStore';
import { useRoadNetworkStore } from './store/roadNetworkStore';
import { usePromptStore } from './store/promptStore';
import SaveConfirmationDialog from './components/common/SaveConfirmationDialog';
import { ImageInfo } from './types';
import { ensureWindowFocus } from './utils/promptUtils';
import { globalEventBus } from './utils/eventBus';

function App() {
    useGlobalHotkeys();
    
    const currentImage = useImageStore((state) => state.currentImage);
    // const setCurrentImage = useImageStore.getState().setCurrentImage; // Temporarily comment out due to linter issue
    
    const {
        saveRoadNetwork,
        loadSavedRoadNetwork,
        clearCurrentRoadNetwork,
        isDirty
    } = useRoadNetworkStore();
    const { clearPrompts } = usePromptStore();

    // State for managing control panel width
    const [controlPanelWidth, setControlPanelWidth] = useState(250);
    const [isSaveDialogVisible, setIsSaveDialogVisible] = useState(false);
    const imageToSwitchToRef = useRef<ImageInfo | null>(null);
    const previousImageIdRef = useRef<string | null>(null);
    const isProgrammaticallySwitchingImageRef = useRef<boolean>(false);
    
    // AbortController for managing image loading cancellation
    const currentLoadingAbortController = useRef<AbortController | null>(null);

    // Add an effect to ensure window focus for hotkeys
    useEffect(() => {
        // Focus the window when the app loads
        ensureWindowFocus();
        
        // Set up a click listener to ensure focus stays in the app
        const handleDocumentClick = () => {
            ensureWindowFocus();
        };
        
        document.addEventListener('click', handleDocumentClick);
        
        // Also try to regain focus when window becomes visible again
        const handleVisibilityChange = () => {
            if (document.visibilityState === 'visible') {
                ensureWindowFocus();
            }
        };
        
        document.addEventListener('visibilitychange', handleVisibilityChange);
        
        return () => {
            document.removeEventListener('click', handleDocumentClick);
            document.removeEventListener('visibilitychange', handleVisibilityChange);
        };
    }, []);

    // Handle beforeunload event for unsaved changes
    useEffect(() => {
        const handleBeforeUnload = (event: BeforeUnloadEvent) => {
            if (isDirty) {
                // Show browser's native dialog
                event.preventDefault();
                return '';
            }
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    }, [isDirty]);

    // Cleanup AbortController on component unmount
    useEffect(() => {
        return () => {
            if (currentLoadingAbortController.current) {
                console.log('Cleaning up AbortController on component unmount');
                currentLoadingAbortController.current.abort();
                currentLoadingAbortController.current = null;
            }
        };
    }, []);



    const performImageSwitch = useCallback(async (newImage: ImageInfo | null) => {
        isProgrammaticallySwitchingImageRef.current = true;
        
        // Abort any previous loading operation
        if (currentLoadingAbortController.current) {
            console.log('Aborting previous image loading operation');
            currentLoadingAbortController.current.abort();
            currentLoadingAbortController.current = null;
        }
        
        // First clear current data
        clearPrompts();
        clearCurrentRoadNetwork();
        
        if (newImage) {
            // First set the current image
            useImageStore.getState().setCurrentImageById(newImage.id);
            
            // Create new AbortController for this loading operation
            const abortController = new AbortController();
            currentLoadingAbortController.current = abortController;
            
            try {
                // Then try to load any saved road network data with abort signal
                console.log(`Trying to load saved road network for image ${newImage.id}`);
                await loadSavedRoadNetwork(newImage.id, abortController.signal);
                
                // Check if this operation was aborted
                if (abortController.signal.aborted) {
                    console.log(`Loading for image ${newImage.id} was aborted`);
                    return;
                }
                
                // Clear the controller reference if successful
                if (currentLoadingAbortController.current === abortController) {
                    currentLoadingAbortController.current = null;
                }
                
                // Ensure window focus after image switch
                ensureWindowFocus();
                
            } catch (error: any) {
                if (error.name === 'AbortError') {
                    console.log(`Loading for image ${newImage.id} was aborted`);
                    return; // Don't proceed with any further operations
                }
                
                // For non-abort errors, clear the controller and let error propagate
                if (currentLoadingAbortController.current === abortController) {
                    currentLoadingAbortController.current = null;
                }
                console.error('Error loading saved road network:', error);
                // Error is already handled in roadNetworkStore
            }
        } else {
            useImageStore.getState().setCurrentImageById('');
        }
        
        previousImageIdRef.current = newImage ? newImage.id : null;
        setTimeout(() => {
            isProgrammaticallySwitchingImageRef.current = false;
        }, 0);
    }, [clearPrompts, loadSavedRoadNetwork, clearCurrentRoadNetwork]);

    const requestImageSwitch = useCallback((newImageInfo: ImageInfo) => {
        const currentImg = useImageStore.getState().currentImage;
        const networkIsDirty = useRoadNetworkStore.getState().isDirty;
        if (currentImg && currentImg.id === newImageInfo.id) return;
        if (networkIsDirty && currentImg) {
            previousImageIdRef.current = currentImg.id;
            imageToSwitchToRef.current = newImageInfo;
            setIsSaveDialogVisible(true);
        } else {
            performImageSwitch(newImageInfo);
        }
    }, [performImageSwitch]);

    useEffect(() => {
        if (isProgrammaticallySwitchingImageRef.current) return;
        const storeCurrentImage = currentImage;
        const lastProcessedImageId = previousImageIdRef.current;
        if (storeCurrentImage && (!lastProcessedImageId || storeCurrentImage.id !== lastProcessedImageId)) {
            performImageSwitch(storeCurrentImage);
        }
        else if (!storeCurrentImage && lastProcessedImageId) {
            performImageSwitch(null);
        }
    }, [currentImage, performImageSwitch]);

    const handleSaveDialogSave = async () => {
        setIsSaveDialogVisible(false);
        const oldImageId = previousImageIdRef.current;
        if (oldImageId && useRoadNetworkStore.getState().roadNetwork) {
            await saveRoadNetwork(oldImageId);
        }
        if (imageToSwitchToRef.current) {
            performImageSwitch(imageToSwitchToRef.current);
        }
        imageToSwitchToRef.current = null;
    };

    const handleSaveDialogDiscard = async () => {
        setIsSaveDialogVisible(false);
        if (imageToSwitchToRef.current) {
            performImageSwitch(imageToSwitchToRef.current);
        }
        imageToSwitchToRef.current = null;
    };

    const handleSaveDialogCancel = () => {
        setIsSaveDialogVisible(false);
        imageToSwitchToRef.current = null;
        previousImageIdRef.current = useImageStore.getState().currentImage?.id || null;
    };

    // Handle control panel width resize
    const handleControlPanelResize = useCallback((newWidth: number) => {
        console.log('App: Resizing control panel to width:', newWidth);
        setControlPanelWidth(newWidth);
    }, []);

    // Effect to handle navigation events from keyboard shortcuts
    useEffect(() => {
        // Add navigation event listeners for keyboard shortcuts
        const handlePreviousImage = () => {
            const { availableImages, currentImage } = useImageStore.getState();
            if (!availableImages || availableImages.length === 0 || !currentImage) return;
            
            const currentIndex = availableImages.findIndex(img => img.id === currentImage.id);
            if (currentIndex > 0) {
                const previousImage = availableImages[currentIndex - 1];
                requestImageSwitch(previousImage);
            }
        };

        const handleNextImage = () => {
            const { availableImages, currentImage } = useImageStore.getState();
            if (!availableImages || availableImages.length === 0 || !currentImage) return;
            
            const currentIndex = availableImages.findIndex(img => img.id === currentImage.id);
            if (currentIndex >= 0 && currentIndex < availableImages.length - 1) {
                const nextImage = availableImages[currentIndex + 1];
                requestImageSwitch(nextImage);
            }
        };

        globalEventBus.on('navigate-to-previous-image', handlePreviousImage);
        globalEventBus.on('navigate-to-next-image', handleNextImage);

        // Cleanup listeners on unmount
        return () => {
            globalEventBus.off('navigate-to-previous-image', handlePreviousImage);
            globalEventBus.off('navigate-to-next-image', handleNextImage);
        };
    }, [requestImageSwitch]);

    return (
        <div className="flex flex-col h-screen bg-gray-800 text-white">
            <Header />
            <div className="flex flex-grow overflow-hidden">
                <ToolBar />
                <ControlPanel 
                    requestImageSwitch={requestImageSwitch} 
                    width={controlPanelWidth}
                />
                <ResizableSplitter 
                    onResize={handleControlPanelResize}
                    minWidth={200}
                    maxWidth={500}
                />
                <CanvasArea />
            </div>
            <StatusBar controlPanelWidth={controlPanelWidth} />
            <SaveConfirmationDialog 
                isOpen={isSaveDialogVisible}
                onSave={handleSaveDialogSave}
                onDiscard={handleSaveDialogDiscard}
                onCancel={handleSaveDialogCancel}
                itemName="road network"
            />
        </div>
    );
}

export default App;
