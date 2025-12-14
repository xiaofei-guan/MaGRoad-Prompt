import React from 'react';
import { usePromptStore } from '../../store/promptStore';
import styles from '../layout/ControlPanel.module.css';

export const PromptController: React.FC = () => {
  const {
    prompts,
    removePrompt,
    clearPrompts,
  } = usePromptStore();

  // all prompts sorted by adding order (latest first)
  const allPrompts = [...prompts].reverse();
  
  const positivePromptsCount = prompts.filter(p => p.type === 'positive').length;
  const negativePromptsCount = prompts.filter(p => p.type === 'negative').length;

  // Performance optimization: limit rendering for large lists
  const VISIBLE_THRESHOLD = 50;
  const shouldUseVirtualization = allPrompts.length > VISIBLE_THRESHOLD;
  const visiblePrompts = shouldUseVirtualization ? allPrompts.slice(0, VISIBLE_THRESHOLD) : allPrompts;

  const PromptItem = React.memo<{ prompt: import('../../types').Prompt }>(({ prompt }) => (
    <div className="flex items-center justify-between px-2 py-1.5 bg-white border border-gray-200 rounded-md hover:bg-gray-50 transition-colors duration-150">
      <div className="flex items-center">
        <span 
          className={`w-2.5 h-2.5 rounded-full mr-2 ${prompt.type === 'positive' ? 'bg-green-500': 'bg-red-500'}`}
          title={prompt.type === 'positive' ? 'Positive prompt' : 'Negative prompt'}
        ></span>
        <span className="text-xs text-gray-700 font-mono">
          X: {prompt.x.toFixed(0)}, Y: {prompt.y.toFixed(0)}
        </span>
      </div>
      <button
        onClick={() => removePrompt(prompt.id)}
        className="text-red-400 hover:text-red-600 hover:bg-red-50 rounded p-0.5 transition-colors duration-150"
        aria-label={`Remove point ${prompt.id}`}
        title="Remove this prompt"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  ));

  return (
    <div className="bg-gray-50 rounded-md border border-gray-200 flex flex-col h-[220px]">
      {/* Fixed Header - More compact */}
      <h3 className="text-sm font-semi text-gray-800 px-2 py-2 border-b border-gray-200 bg-gray-100 rounded-t-md flex-shrink-0">
        Prompt Points
      </h3>
      
      {/* Scrollable Prompt List - Expanded height to compensate for removed buttons */}
      <div className="px-2 flex-grow overflow-hidden">
        <div 
          className={`overflow-y-auto space-y-1 ${styles.promptListScroll}`}
          style={{ height: '130px' }} // Increased height since we removed the button section
        >
          {prompts.length === 0 ? (
            <div className="text-xs text-gray-500 text-center py-6 italic">
              Left-click to add positive points, right-click for negative...
            </div>
          ) : (
            <>
              {visiblePrompts.map(p => <PromptItem key={p.id} prompt={p} />)}
              {shouldUseVirtualization && (
                <div className="text-xs text-gray-500 text-center py-2 bg-gray-100 rounded border-t border-gray-200">
                  Displaying first {VISIBLE_THRESHOLD} points, total {allPrompts.length} points
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Fixed Footer - More compact padding */}
      <div className="p-2 border-t border-gray-200 flex-shrink-0">
        <div className="flex items-center justify-between">
          <button
            onClick={clearPrompts}
            disabled={prompts.length === 0}
            className="px-2.5 py-1.5 text-xs bg-red-500 text-white rounded-md hover:bg-red-600 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            title="Clear all prompt points"
          >
            Clear all points
          </button>
          <div className="text-xs text-gray-600 space-y-0.5">
            <div className="flex items-center">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-1"></span>
              Pos: {positivePromptsCount}
            </div>
            <div className="flex items-center">
              <span className="w-2 h-2 bg-red-500 rounded-full mr-1"></span>
              Neg: {negativePromptsCount}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};