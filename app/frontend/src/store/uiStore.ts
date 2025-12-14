import { create } from 'zustand';

export type InteractionMode = 'view' | 'label' | 'edit';

// Added Layer Visibility State - Now exported
export interface LayerVisibility {
  image: boolean;
  prompts: boolean;
  network: boolean;
  roadMask: boolean;
  keypointMask: boolean;
  patchGrid: boolean; // Add patch grid visibility
}

export interface PerformanceSettings {
    enableLOD: boolean;
    enableViewportCulling: boolean;
    maxFeatures: number;
    simplificationTolerance: number;
    showPerformanceMonitor: boolean;
}

interface UiState {
  interactionMode: InteractionMode;
  layerVisibility: LayerVisibility;
  // Add minimap visibility state
  minimapVisible: boolean;
  performanceSettings: PerformanceSettings;
  setInteractionMode: (mode: InteractionMode) => void;
  toggleLayerVisibility: (layer: keyof LayerVisibility) => void;
  setLayerVisibility: (layer: keyof LayerVisibility, visible: boolean) => void;
  // Add function to toggle minimap visibility
  toggleMinimapVisibility: () => void;
  // Add function to set minimap visibility directly
  setMinimapVisibility: (visible: boolean) => void;
  updatePerformanceSettings: (settings: Partial<PerformanceSettings>) => void;
}

export const useUiStore = create<UiState>((set) => ({
  interactionMode: 'view', // Default mode
  // Default visibility state
  layerVisibility: {
    image: true,
    prompts: true,
    network: true,
    roadMask: false,
    keypointMask: false,
    patchGrid: true, // Default to true, user can toggle on when needed
  },
  // Default minimap visibility - visible
  minimapVisible: true,
  
  setInteractionMode: (mode) => {
    set({ interactionMode: mode });
    
    // When entering edit mode, set default edit tools
    // Remove any roadNetwork dependency - users should be able to prepare edit mode anytime
    if (mode === 'edit') {
      // Import roadNetworkStore actions dynamically to avoid circular imports
      import('./roadNetworkStore').then(({ useRoadNetworkStore }) => {
        const { setEditMode, setSnappingEnabled, editState } = useRoadNetworkStore.getState();
        // Only set defaults if no edit mode is currently set (first time entering edit mode)
        if (editState.mode === null) {
          setEditMode('modify');
          setSnappingEnabled(true);
        }
      });
    }
  },

  // Action to toggle a specific layer's visibility
  toggleLayerVisibility: (layer) => set((state) => ({
    layerVisibility: {
      ...state.layerVisibility,
      [layer]: !state.layerVisibility[layer],
    }
  })),

  // Action to set a specific layer's visibility
  setLayerVisibility: (layer, visible) => set((state) => ({
    layerVisibility: {
      ...state.layerVisibility,
      [layer]: visible,
    }
  })),
  
  // Action to toggle minimap visibility
  toggleMinimapVisibility: () => set((state) => ({
    minimapVisible: !state.minimapVisible,
  })),
  
  // Action to set minimap visibility directly
  setMinimapVisibility: (visible) => set({ minimapVisible: visible }),

  performanceSettings: {
    enableLOD: true,
    enableViewportCulling: true,
    maxFeatures: 2000,
    simplificationTolerance: 0.01,
    showPerformanceMonitor: false,
  },

  updatePerformanceSettings: (settings) => set((state) => ({
    performanceSettings: {
      ...state.performanceSettings,
      ...settings,
    },
  })),
})); 