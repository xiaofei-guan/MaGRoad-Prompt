import { useUiStore } from '../../store/uiStore';

function LayerControls() {
    // Select state pieces individually to prevent unnecessary re-renders
    const layerVisibility = useUiStore((state) => state.layerVisibility);
    const toggleLayerVisibility = useUiStore((state) => state.toggleLayerVisibility);

    // Helper to create consistent checkbox+label rows with improved styling
    const renderCheckbox = (layerKey: keyof typeof layerVisibility, label: string) => (
        <div className="flex items-center space-x-2 py-0.5">
            <input
                type="checkbox"
                id={`layer-${layerKey}`}
                checked={layerVisibility[layerKey]}
                onChange={() => toggleLayerVisibility(layerKey)}
                className="h-3.5 w-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-1"
            />
            <label 
                htmlFor={`layer-${layerKey}`} 
                className="text-xs font-medium text-gray-700 cursor-pointer select-none"
                title={`Toggle ${label} visibility`}
            >
                {label}
            </label>
        </div>
    );

    return (
        <div className="mt-3 pt-3 border-t border-gray-200">
            <h3 className="text-sm font-semi text-gray-800 mb-2">Layers</h3>
            <div className="space-y-1.5">
                {renderCheckbox('image', 'Base Image')}
                {renderCheckbox('prompts', 'Prompt Points')}
                {renderCheckbox('network', 'Road Network')}
                {renderCheckbox('roadMask', 'Road Mask')}
                {renderCheckbox('keypointMask', 'Keypoint Mask')}
                {renderCheckbox('patchGrid', 'Patch Grid')}
            </div>
        </div>
    );
}

export default LayerControls; // Using default export as it's the main component in this file 