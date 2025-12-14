import React from 'react';
import { useImageStore } from '../../store/imageStore';

export const FeatureStatusDisplay: React.FC = () => {
  const { currentImage, featureStatus, isFeatureLoading, startFeatureComputation, recomputeImageFeatures } = useImageStore();

  const handleComputeClick = () => {
    if (!currentImage || isFeatureLoading) return;
    startFeatureComputation(currentImage.id);
  };

  const handleRecomputeClick = () => {
    if (!currentImage || isFeatureLoading) return;
    recomputeImageFeatures(currentImage.id);
  };

  let statusText = "Please select an image first";
  let statusBgColor = "bg-gray-200";
  let statusTextColor = "text-gray-600";
  // Button visibility logic
  let showComputeButton = false;
  let showRecomputeButton = false;

  if (currentImage) {
    switch (featureStatus) {
      case 'none':
        statusText = "Features not computed";
        statusBgColor = "bg-yellow-100";
        statusTextColor = "text-yellow-800";
        showComputeButton = true;
        break;
      case 'computing':
        statusText = "Computing features...";
        statusBgColor = "bg-blue-100";
        statusTextColor = "text-blue-800";
        break;
      case 'ready':
        statusText = "Features computed";
        statusBgColor = "bg-indigo-100"; // Using indigo based on design
        statusTextColor = "text-indigo-700";
        showRecomputeButton = true; // allow recompute when features already exist
        break;
      case 'error':
        statusText = "Feature computation failed";
        statusBgColor = "bg-red-100";
        statusTextColor = "text-red-800";
        showComputeButton = true; // Allow re-try when failed (no features yet)
        break;
    }
  }

  return (
    <div className="bg-gray-50 rounded-md border border-gray-200">
      <div className="p-3 space-y-2">
        <div className={`px-2.5 py-1.5 rounded-md text-xs font-medium ${statusBgColor} ${statusTextColor}`}>
          {statusText}
        </div>
        {showComputeButton && (
          <button
            onClick={handleComputeClick}
            disabled={isFeatureLoading || !currentImage}
            className="w-full px-2.5 py-1.5 text-xs bg-blue-500 text-white rounded-md hover:bg-blue-600 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            title="Start feature computation for current image"
          >
            {isFeatureLoading ? 'Computing...' : 'Compute Features'}
          </button>
        )}
        {showRecomputeButton && (
          <button
            onClick={handleRecomputeClick}
            disabled={isFeatureLoading || !currentImage}
            className="w-full px-2.5 py-1.5 text-xs bg-indigo-500 text-white rounded-md hover:bg-indigo-600 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            title="Recompute features for current image"
          >
            {isFeatureLoading ? 'Computing...' : 'Recompute Features'}
          </button>
        )}
      </div>
    </div>
  );
};