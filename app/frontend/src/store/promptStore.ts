import { create } from 'zustand';
import { Prompt, Point, PromptType } from '../types';
import { useRoadNetworkStore } from './roadNetworkStore'; // For setting network dirty

export interface PromptState {
  prompts: Prompt[];
  nextId: number;
  addPrompt: (point: Point, type?: PromptType) => void; // Modified to accept optional type parameter
  removePrompt: (id: string) => void;
  clearPrompts: () => void;
  setLoadedPrompts: (loadedPrompts: Prompt[]) => void;
}

export const usePromptStore = create<PromptState>((set, get) => ({
  prompts: [],
  nextId: 1,

  addPrompt: (point, type = 'positive') => { // Default to 'positive' if no type specified
    const newIdNumber = get().nextId;
    const newPrompt: Prompt = {
      id: `prompt-${newIdNumber}`,
      x: point.x,
      y: point.y,
      type: type,
      label: type === 'positive' ? 1 : 0,
    };
    console.log('Adding prompt:', newPrompt);
    set((state) => ({
      prompts: [...state.prompts, newPrompt],
      nextId: newIdNumber + 1,
    }));
    if (useRoadNetworkStore.getState().roadNetwork) {
      useRoadNetworkStore.getState().setNetworkAsDirty();
    }
  },

  removePrompt: (idToRemove) => {
    set((state) => ({
      prompts: state.prompts.filter((prompt) => prompt.id !== idToRemove),
    }));
    if (useRoadNetworkStore.getState().roadNetwork) {
      useRoadNetworkStore.getState().setNetworkAsDirty();
    }
  },

  clearPrompts: () => {
    set({ prompts: [], nextId: 1 });
  },

  setLoadedPrompts: (loadedPrompts: Prompt[]) => {
    let nextIdToAssign = 1;
    
    // Ensure all prompts have valid IDs
    const promptsWithValidIds = loadedPrompts.map(prompt => {
      // Check if the prompt has a valid ID
      if (!prompt.id || prompt.id === 'undefined') {
        // Assign a new ID if missing or invalid
        return {
          ...prompt,
          id: `prompt-${nextIdToAssign++}`
        };
      }
      
      // If ID exists and is valid, extract its number for nextId calculation
      if (typeof prompt.id === 'string') {
        const parts = prompt.id.split('-');
          if (parts.length === 2) {
            const idNum = parseInt(parts[1]);
            if (!isNaN(idNum)) {
            nextIdToAssign = Math.max(nextIdToAssign, idNum + 1);
            }
          }
        }
      
      return prompt;
    });
    
    console.log(`Loading ${promptsWithValidIds.length} prompts with validated IDs. Next ID will be ${nextIdToAssign}`);
    
    set({
      prompts: promptsWithValidIds,
      nextId: nextIdToAssign,
    });
  },
})); 