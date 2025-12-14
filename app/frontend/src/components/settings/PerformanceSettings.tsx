import React from 'react';
import { useUiStore, PerformanceSettings as PerformanceSettingsType } from '../../store/uiStore';

interface PerformanceSettingsProps {
    visible: boolean;
    onClose: () => void;
}

const PerformanceSettings: React.FC<PerformanceSettingsProps> = ({ visible, onClose }) => {
    const performanceSettings = useUiStore((state) => state.performanceSettings);
    const updatePerformanceSettings = useUiStore((state) => state.updatePerformanceSettings);

    if (!visible) return null;

    const handleSettingChange = (key: keyof PerformanceSettingsType, value: any) => {
        updatePerformanceSettings({ [key]: value });
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold text-gray-800">Performance Settings</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700"
                    >
                        ✕
                    </button>
                </div>

                <div className="space-y-4">
                    <div className="space-y-2">
                        <label className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-700">Enable LOD Rendering</span>
                            <input
                                type="checkbox"
                                checked={performanceSettings.enableLOD}
                                onChange={(e) => handleSettingChange('enableLOD', e.target.checked)}
                                className="ml-2"
                            />
                        </label>
                        <p className="text-xs text-gray-500">
                            Automatically adjust detail level based on zoom
                        </p>
                    </div>

                    <div className="space-y-2">
                        <label className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-700">Enable Viewport Culling</span>
                            <input
                                type="checkbox"
                                checked={performanceSettings.enableViewportCulling}
                                onChange={(e) => handleSettingChange('enableViewportCulling', e.target.checked)}
                                className="ml-2"
                            />
                        </label>
                        <p className="text-xs text-gray-500">
                            Only render elements visible in current view
                        </p>
                    </div>

                    <div className="space-y-2">
                        <label className="block text-sm font-medium text-gray-700">
                            Max Features: {performanceSettings.maxFeatures}
                        </label>
                        <input
                            type="range"
                            min="100"
                            max="2000"
                            step="100"
                            value={performanceSettings.maxFeatures}
                            onChange={(e) => handleSettingChange('maxFeatures', parseInt(e.target.value))}
                            className="w-full"
                        />
                        <p className="text-xs text-gray-500">
                            Maximum number of features to render simultaneously
                        </p>
                    </div>

                    <div className="space-y-2">
                        <label className="block text-sm font-medium text-gray-700">
                            Simplification Tolerance: {performanceSettings.simplificationTolerance.toFixed(3)}
                        </label>
                        <input
                            type="range"
                            min="0"
                            max="0.1"
                            step="0.001"
                            value={performanceSettings.simplificationTolerance}
                            onChange={(e) => handleSettingChange('simplificationTolerance', parseFloat(e.target.value))}
                            className="w-full"
                        />
                        <p className="text-xs text-gray-500">
                            Line simplification threshold (higher = more simplified)
                        </p>
                    </div>

                    {process.env.NODE_ENV === 'development' && (
                        <div className="space-y-2">
                            <label className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Show Performance Monitor</span>
                                <input
                                    type="checkbox"
                                    checked={performanceSettings.showPerformanceMonitor}
                                    onChange={(e) => handleSettingChange('showPerformanceMonitor', e.target.checked)}
                                    className="ml-2"
                                />
                            </label>
                            <p className="text-xs text-gray-500">
                                Display real-time performance statistics
                            </p>
                        </div>
                    )}
                </div>

                <div className="mt-6 flex justify-end space-x-3">
                    <button
                        onClick={() => {
                            updatePerformanceSettings({
                                enableLOD: true,
                                enableViewportCulling: true,
                                maxFeatures: 2000,
                                simplificationTolerance: 0.01,
                                showPerformanceMonitor: true,
                            });
                        }}
                        className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                    >
                        Reset to Defaults
                    </button>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
};

export default PerformanceSettings; 