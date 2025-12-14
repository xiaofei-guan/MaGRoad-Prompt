import React, { useState, useRef, useCallback, useEffect } from 'react';

// Interface for ResizableSplitter component props
interface ResizableSplitterProps {
    onResize: (newWidth: number) => void;
    minWidth?: number;
    maxWidth?: number;
    className?: string;
}

// Resizable splitter component for adjusting panel width
const ResizableSplitter: React.FC<ResizableSplitterProps> = ({
    onResize,
    minWidth = 200,
    maxWidth = 500,
    className = ''
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const splitterRef = useRef<HTMLDivElement>(null);
    
    // Use refs to store drag state to avoid closure issues
    const dragStateRef = useRef({
        isDragging: false,
        startX: 0,
        startWidth: 0
    });

    // Handle mouse move event during dragging
    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!dragStateRef.current.isDragging) {
            console.log('Mouse move but not dragging');
            return;
        }
        
        const deltaX = e.clientX - dragStateRef.current.startX;
        const newWidth = Math.max(minWidth, Math.min(maxWidth, dragStateRef.current.startWidth + deltaX));
        
        console.log('Resizing:', { deltaX, newWidth, currentX: e.clientX });
        onResize(newWidth);
    }, [minWidth, maxWidth, onResize]);

    // Handle mouse up event to stop dragging
    const handleMouseUp = useCallback(() => {
        dragStateRef.current.isDragging = false;
        setIsDragging(false);
        
        // Remove mouse events from document
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }, [handleMouseMove]);

    // Handle mouse down event to start dragging
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        
        // Get the current width of the control panel
        const controlPanel = document.querySelector('[data-control-panel]') as HTMLElement;
        if (!controlPanel) {
            console.warn('Control panel not found for resizing');
            return;
        }
        
        // Update both state and ref
        const startX = e.clientX;
        const startWidth = controlPanel.offsetWidth;
        
        console.log('Starting drag:', { startX, startWidth });
        
        dragStateRef.current = {
            isDragging: true,
            startX,
            startWidth
        };
        
        setIsDragging(true);
        
        // Add mouse events to document for better tracking
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }, [handleMouseMove, handleMouseUp]);

    // Cleanup effect for removing event listeners
    useEffect(() => {
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [handleMouseMove, handleMouseUp]);

    return (
        <div
            ref={splitterRef}
            className={`
                w-2 bg-gray-400 hover:bg-blue-500 cursor-col-resize relative
                transition-colors duration-200 flex items-center justify-center
                ${isDragging ? 'bg-blue-600' : ''}
                ${className}
            `}
            onMouseDown={handleMouseDown}
            title="Drag to resize control panel"
            style={{ minWidth: '8px', zIndex: 10 }} // on the top layer
        >
            {/* Visual indicator always visible */}
            <div className="w-0.5 h-8 bg-white rounded-full opacity-70"></div>
            
            {/* Active dragging indicator */}
            {isDragging && (
                <div className="absolute inset-0 bg-blue-600 opacity-50 pointer-events-none animate-pulse"></div>
            )}
        </div>
    );
};

export default ResizableSplitter; 