import React from 'react';
import { ImageSelector } from '../editor/ImageSelector';
import { FeatureStatusDisplay } from '../editor/FeatureStatusDisplay';
import { PromptController } from '../editor/PromptController';
import { ActionButtons } from '../editor/ActionButtons';
import LayerControls from '../editor/LayerControls';
import { ImageInfo } from '../../types';
import styles from './ControlPanel.module.css';

interface ControlPanelProps {
    requestImageSwitch: (image: ImageInfo) => void;
    width?: number; // Optional width prop for resizable functionality
}

export const ControlPanel: React.FC<ControlPanelProps> = ({ requestImageSwitch, width = 300 }) => {
    return (
        <aside 
            className={`bg-white h-full border-r border-gray-200 overflow-y-auto flex flex-col ${styles.controlPanel}`}
            style={{ width: `${width}px` }}
            data-control-panel // Data attribute for splitter component to find this element
        >
            <div className="p-3 space-y-4 flex-grow">
                <ImageSelector requestImageSwitch={requestImageSwitch} />
                <FeatureStatusDisplay />
                <PromptController />
            </div>
            <div className="p-3 border-t border-gray-200 mt-auto">
                <ActionButtons />
                <LayerControls />
            </div>
        </aside>
    );
}; 