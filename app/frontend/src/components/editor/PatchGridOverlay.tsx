import React, { useMemo } from 'react';
import { Line } from 'react-konva';

interface PatchGridOverlayProps {
  imageWidth: number;
  imageHeight: number;
  patchSize?: number;
  overlap?: number;
  visible?: boolean;
  offsetX?: number;
  offsetY?: number;
  scaleX?: number;
  scaleY?: number;
}

// Default values for patch configuration (matching backend settings)
const DEFAULT_PATCH_SIZE = 1024;
const DEFAULT_OVERLAP = 128;
const LARGE_IMAGE_THRESHOLD = 1024;

// Component to render patch grid overlay with dashed lines
const PatchGridOverlay: React.FC<PatchGridOverlayProps> = ({
  imageWidth,
  imageHeight,
  patchSize = DEFAULT_PATCH_SIZE,
  overlap = DEFAULT_OVERLAP,
  visible = true,
  offsetX = 0,
  offsetY = 0,
  scaleX = 1,
  scaleY = 1
}) => {
  
  // Calculate grid lines only when image dimensions or patch settings change
  const gridLines = useMemo(() => {
    // Only show grid for large images
    if (!visible || (imageWidth <= LARGE_IMAGE_THRESHOLD && imageHeight <= LARGE_IMAGE_THRESHOLD)) {
      return { verticalLines: [], horizontalLines: [] };
    }

    const verticalLines: number[][] = [];
    const horizontalLines: number[][] = [];
    
    // Calculate effective step size (patch size minus overlap)
    const effectiveStep = patchSize - overlap;
    
    // Generate vertical lines (for width divisions) 
    // Show lines at patch boundaries based on backend logic
    if (imageWidth > LARGE_IMAGE_THRESHOLD) {
      // Calculate grid based on backend logic
      const stride = patchSize - overlap;
      const gridCols = Math.max(1, Math.ceil((imageWidth - overlap) / stride));
      
      // Draw lines at patch boundaries (where patches start, not end)
      for (let col = 1; col < gridCols; col++) {
        const x = col * stride; // This matches backend: col * stride
        if (x < imageWidth) {
          const scaledX = x * scaleX + offsetX;
          const scaledHeight = imageHeight * scaleY;
          verticalLines.push([scaledX, offsetY, scaledX, offsetY + scaledHeight]);
        }
      }
    }
    
    // Generate horizontal lines (for height divisions)
    // Show lines at patch boundaries based on backend logic  
    if (imageHeight > LARGE_IMAGE_THRESHOLD) {
      // Calculate grid based on backend logic
      const stride = patchSize - overlap;
      const gridRows = Math.max(1, Math.ceil((imageHeight - overlap) / stride));
      
      // Draw lines at patch boundaries (where patches start, not end)
      for (let row = 1; row < gridRows; row++) {
        const y = row * stride; // This matches backend: row * stride
        if (y < imageHeight) {
          const scaledY = y * scaleY + offsetY;
          const scaledWidth = imageWidth * scaleX;
          horizontalLines.push([offsetX, scaledY, offsetX + scaledWidth, scaledY]);
        }
      }
    }
    
    return { verticalLines, horizontalLines };
  }, [imageWidth, imageHeight, patchSize, overlap, visible, offsetX, offsetY, scaleX, scaleY]);

  // Don't render anything if not visible or if it's a small image
  if (!visible || (imageWidth <= LARGE_IMAGE_THRESHOLD && imageHeight <= LARGE_IMAGE_THRESHOLD)) {
    return null;
  }

  // Determine if this is a minimap based on scale (minimap typically has scale < 1)
  const isMinimapContext = scaleX < 1 || scaleY < 1;
  
  // Adjust line properties for minimap context
  const lineProps = isMinimapContext 
    ? {
        strokeWidth: 0.7, // Thinner lines for minimap
        opacity: 0.9,     // Higher opacity for better visibility in minimap
        dash: [2, 2]      // Shorter dash pattern for minimap
      }
    : {
        strokeWidth: 5,   // User-adjusted thicker lines for main canvas
        opacity: 1.0,     // User-adjusted higher opacity for main canvas  
        dash: [10, 10]    // Longer dash pattern for main canvas
      };

  return (
    <>
      {/* Render vertical lines */}
      {gridLines.verticalLines.map((points, index) => (
        <Line
          key={`vertical-${index}`}
          points={points}
          stroke="#d9f713" // #ff6b35
          strokeWidth={lineProps.strokeWidth}
          dash={lineProps.dash}
          opacity={lineProps.opacity}
          listening={false} // Don't capture events
        />
      ))}
      
      {/* Render horizontal lines */}
      {gridLines.horizontalLines.map((points, index) => (
        <Line
          key={`horizontal-${index}`}
          points={points}
          stroke="#d9f713" // #ff6b35
          strokeWidth={lineProps.strokeWidth}
          dash={lineProps.dash}
          opacity={lineProps.opacity}
          listening={false} // Don't capture events
        />
      ))}
    </>
  );
};

export default PatchGridOverlay; 