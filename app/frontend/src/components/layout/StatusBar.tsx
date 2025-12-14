import React, { useEffect } from 'react';
import { useImageStore } from '../../store/imageStore';
// import { usePromptStore } from '../../store/promptStore';
import { useRoadNetworkStore } from '../../store/roadNetworkStore';
import { useUiStore } from '../../store/uiStore';

interface StatusBarProps {
  controlPanelWidth?: number;
}

export const StatusBar: React.FC<StatusBarProps> = ({ controlPanelWidth }) => {
  const currentImage = useImageStore((state) => state.currentImage);
  const featureStatus = useImageStore((state) => state.featureStatus);
  // const prompts = usePromptStore((state) => state.prompts);
  const isComputingFeatures = useRoadNetworkStore((state) => state.isComputingFeatures);
  // const isNetworkLoading = useRoadNetworkStore((state) => state.isLoading);
  const roadNetwork = useRoadNetworkStore((state) => state.roadNetwork);
  const interactionMode = useUiStore((state) => state.interactionMode);
  const minimapVisible = useUiStore((state) => state.minimapVisible);

  // Add logging to track when the component re-renders and what the annotation status is
  useEffect(() => {
    if (currentImage) {
      console.log(`StatusBar: Current image changed to ${currentImage.id} (${currentImage.original_filename}), isAnnotated=${currentImage.isAnnotated}`);
    }
  }, [currentImage]);

  // Prompt counters not displayed in footer
  const pointsCount = roadNetwork?.nodes?.length || 0;
  const edgesCount = roadNetwork?.edges?.length || 0;

  // Device display removed

  // English labels
  let featureStatusText = "Not computed";
  if (featureStatus === 'ready') featureStatusText = "Computed";
  else if (featureStatus === 'computing' || isComputingFeatures) featureStatusText = "Computing...";

  // Get display name for current image - prefer original filename
  const displayFilename = currentImage ? 
    (currentImage.original_filename || currentImage.name || currentImage.filename || 'None') : 'None';

  // Get annotation status for current image
  const isAnnotated = currentImage?.isAnnotated || false;
  const annotationStatus = isAnnotated ? "Annotated" : "Not annotated";
  
  // Log the final status that will be displayed
  // console.log(`StatusBar render: isAnnotated=${isAnnotated}, annotationStatus="${annotationStatus}"`);

  return (
    <footer className="h-[25px] bg-gray-200 border-t border-gray-300 text-xs text-gray-700 flex items-center px-4">
      {/* Device removed */}
      <span>Image: {displayFilename}</span>
      <span className="mx-2">|</span>
      <span>Feature: {featureStatusText}</span>
      <span className="mx-2">|</span>
      <span>Annotation: <span className={isAnnotated ? "text-green-600 font-medium" : ""}>{annotationStatus}</span></span>
      <span className="mx-2">|</span>
      <span>Mode: <span className="font-medium capitalize">{interactionMode}</span></span>
      <span className="mx-2">|</span>
      <span>Minimap: <span className={`font-medium ${minimapVisible ? "text-green-600" : "text-red-600"}`}>{minimapVisible ? "Shown" : "Hidden"}</span></span>
      {controlPanelWidth && (
        <>
          <span className="mx-2">|</span>
          <span>Panel width: <span className="font-medium">{Math.round(controlPanelWidth)}px</span></span>
        </>
      )}
      
      <span className="ml-auto"></span>
      <span>Point: {pointsCount} | Edge: {edgesCount}</span>
    </footer>
  );
}; 