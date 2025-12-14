import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Stage as StageType } from 'konva/lib/Stage';
import Konva from 'konva';
import { useRoadNetworkStore } from '../../store/roadNetworkStore';

// Props definition for StatusOverlay
interface StatusOverlayProps {
    stageRef: React.RefObject<StageType | null>;
    imageDimensions: { width: number; height: number } | null;
}

// Component to display coordinates and legend information
const StatusOverlay = React.memo(({ 
    stageRef, 
    imageDimensions 
}: StatusOverlayProps) => {
    const [coords, setCoords] = useState({ x: 0, y: 0 });
    const rafIdRef = useRef<number | null>(null);
    const roadNetwork = useRoadNetworkStore((state) => state.roadNetwork);

    const updateCoords = useCallback(() => {
        const stage = stageRef.current;
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        const imageNode = stage.find('.main-image')[0] as Konva.Image | undefined;

        if (pointerPos && imageNode) {
            const transform = imageNode.getAbsoluteTransform().copy().invert();
            const point = transform.point(pointerPos);
            setCoords({ x: Math.round(point.x), y: Math.round(point.y) });
        } else if (pointerPos) {
            const stagePos = stage.position();
            const stageScale = stage.scaleX();
            setCoords({
                x: Math.round((pointerPos.x - stagePos.x) / stageScale),
                y: Math.round((pointerPos.y - stagePos.y) / stageScale),
            });
        } else {
            setCoords({ x: 0, y: 0 });
        }
    }, [stageRef]);

    const updateCoordsThrottled = useCallback(() => {
        if (rafIdRef.current !== null) return;
        rafIdRef.current = window.requestAnimationFrame(() => {
            rafIdRef.current = null;
            updateCoords();
        });
    }, [updateCoords]);

    useEffect(() => {
        const stage = stageRef.current;
        if (!stage) return;

        stage.on('mousemove dragmove', updateCoordsThrottled);
        stage.on('wheel', updateCoordsThrottled);

        return () => {
            if (stage) {
                stage.off('mousemove dragmove', updateCoordsThrottled);
                stage.off('wheel', updateCoordsThrottled);
            }
            if (rafIdRef.current !== null) {
                window.cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
            }
        };
    }, [stageRef, updateCoordsThrottled]);

    return (
        <div className="absolute top-2 left-2 right-2 bottom-2 pointer-events-none">
            {/* Bottom-left coordinates & dimensions */}
            <div className="absolute bottom-0 left-0 bg-gray-400 bg-opacity-50 text-black text-xs p-1 px-2 rounded">
                {imageDimensions && <span>Size: {imageDimensions.width} x {imageDimensions.height} | </span>}
                <span>Coords: ({coords.x}, {coords.y})</span>
            </div>
            
            {/* Bottom-right legend -> point / edge */}
            <div className="absolute bottom-0 right-0 bg-gray-400 bg-opacity-50 text-black text-xs p-1 px-2 rounded flex items-center space-x-3">
                <span className="font-medium">Legend:</span>
                <div className="flex items-center space-x-1">
                    <span className="w-3 h-3 bg-[#cacc28] rounded-full inline-block border border-gray-900"></span>
                    <span>point</span>
                </div>
                <div className="flex items-center space-x-1">
                    <span className="w-3 h-3 bg-[#3b82f6] inline-block border border-gray-900"></span>
                    <span>edge</span>
                </div>
            </div>
        </div>
    );
});

export default StatusOverlay; 