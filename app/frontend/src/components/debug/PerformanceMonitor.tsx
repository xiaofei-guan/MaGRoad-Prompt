import React from 'react';

interface PerformanceMonitorProps {
    renderingStats: {
        originalFeatures: number;
        filteredFeatures: number;
        originalPoints: number;
        filteredPoints: number;
        detailLevel: string;
        reductionRatio: number;
    } | null;
    stageScale: number;
    isInteracting: boolean;
    visible?: boolean;
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
    renderingStats,
    stageScale,
    isInteracting,
    visible = false
}) => {
    if (!visible || !renderingStats) {
        return null;
    }

    const getPerformanceColor = (reductionRatio: number) => {
        if (reductionRatio > 0.7) return 'text-green-400';
        if (reductionRatio > 0.4) return 'text-yellow-400';
        return 'text-red-400';
    };

    const getDetailLevelColor = (level: string) => {
        switch (level) {
            case 'MINIMAL':
            case 'LOW':
                return 'text-green-400';
            case 'MEDIUM':
                return 'text-yellow-400';
            case 'HIGH':
            case 'FULL':
                return 'text-orange-400';
            default:
                return 'text-gray-400';
        }
    };

    return (
        <div className="fixed top-20 right-4 bg-black bg-opacity-80 text-white p-3 rounded-lg text-xs font-mono z-50 min-w-48">
            <div className="font-bold mb-2 text-blue-400">Performance Monitor</div>
            
            <div className="space-y-1">
                <div className="flex justify-between">
                    <span>Scale:</span>
                    <span className="text-cyan-400">{stageScale.toFixed(2)}x</span>
                </div>
                
                <div className="flex justify-between">
                    <span>LOD:</span>
                    <span className={getDetailLevelColor(renderingStats.detailLevel)}>
                        {renderingStats.detailLevel}
                    </span>
                </div>
                
                <div className="flex justify-between">
                    <span>Features:</span>
                    <span className="text-blue-400">
                        {renderingStats.filteredFeatures}/{renderingStats.originalFeatures}
                    </span>
                </div>
                
                <div className="flex justify-between">
                    <span>Points:</span>
                    <span className="text-purple-400">
                        {renderingStats.filteredPoints}/{renderingStats.originalPoints}
                    </span>
                </div>
                
                <div className="flex justify-between">
                    <span>Reduction:</span>
                    <span className={getPerformanceColor(renderingStats.reductionRatio)}>
                        {(renderingStats.reductionRatio * 100).toFixed(1)}%
                    </span>
                </div>
                
                <div className="flex justify-between">
                    <span>Interacting:</span>
                    <span className={isInteracting ? 'text-red-400' : 'text-green-400'}>
                        {isInteracting ? 'YES' : 'NO'}
                    </span>
                </div>
            </div>
            
            {renderingStats.reductionRatio > 0.5 && (
                <div className="mt-2 text-green-400 text-xs">
                    ⚡ Optimized rendering active
                </div>
            )}
        </div>
    );
};

export default PerformanceMonitor; 