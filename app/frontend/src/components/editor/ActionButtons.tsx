import React from 'react';
import { useImageStore } from '../../store/imageStore';
import { usePromptStore } from '../../store/promptStore';
import { useRoadNetworkStore } from '../../store/roadNetworkStore';

export const ActionButtons: React.FC = () => {
  const { currentImage, featureStatus } = useImageStore();
  const prompts = usePromptStore((state) => state.prompts);
  const {
    roadNetwork,
    generateRoadNetwork,
    saveRoadNetwork,
    isLoading: isNetworkLoading,
    saveSuccess,
    isComputingFeatures,
  } = useRoadNetworkStore();

  const canGenerate = currentImage && featureStatus === 'ready' && prompts.length > 0;
  const canSave = currentImage && roadNetwork;

  const handleGenerate = () => {
    if (currentImage && prompts.length > 0 && !isNetworkLoading) {
      generateRoadNetwork(currentImage.id, prompts);
    }
  };

  const handleSave = () => {
    if (canSave && !isNetworkLoading && roadNetwork) {
        saveRoadNetwork(currentImage.id);
    }
  };

  // Button text based on loading states
  let generateButtonText = 'Auto-run';
  if (isNetworkLoading) {
    generateButtonText = isComputingFeatures ? 'Computing Features...' : 'Generating...';
  }

  return (
    <div className="flex space-x-2">
      <button
        onClick={handleGenerate}
        disabled={(!currentImage || prompts.length === 0 || isNetworkLoading)}
        className="flex-1 px-3 py-2 text-[13px] font-medium bg-cyan-500 text-white rounded-md hover:bg-cyan-500 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
        title="Generate road network from prompts"
      >
        {generateButtonText}
      </button>
      <button
        onClick={handleSave}
        disabled={!canSave || isNetworkLoading}
        className="flex-1 px-3 py-2 text-[13px] font-medium bg-teal-500 text-white rounded-md hover:bg-teal-500 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
        title="Save current annotation results"
      >
        {isNetworkLoading && roadNetwork !== null ? 'Saving...' : 'Save Results'}
      </button>
    </div>
  );
}; 