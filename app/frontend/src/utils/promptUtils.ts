import { Prompt } from '../types';

/**
 * Ensures all prompts have valid IDs by generating them if needed
 * Also ensures the type field matches the label value
 * 
 * @param prompts Array of prompts that might be missing IDs or type
 * @returns Array of prompts with guaranteed valid IDs and types
 */
export function ensurePromptIds(prompts: Prompt[]): Prompt[] {
  let nextId = 1;
  
  return prompts.map(prompt => {
    let updatedPrompt = { ...prompt };
    
    // Ensure ID is valid
    if (!prompt.id || prompt.id === 'undefined') {
      updatedPrompt.id = `prompt-${nextId++}`;
    } else if (typeof prompt.id === 'string') {
      const parts = prompt.id.split('-');
      if (parts.length === 2) {
        const idNum = parseInt(parts[1]);
        if (!isNaN(idNum) && idNum >= nextId) {
          nextId = idNum + 1;
        }
      }
    }
    
    // Ensure type matches label
    // If label is 1, type should be 'positive'
    // If label is 0, type should be 'negative'
    if (prompt.label === 1 && prompt.type !== 'positive') {
      updatedPrompt.type = 'positive';
    } else if (prompt.label === 0 && prompt.type !== 'negative') {
      updatedPrompt.type = 'negative';
    } else if (prompt.type === undefined) {
      // Default if no type or label is specified
      updatedPrompt.type = prompt.label === 1 ? 'positive' : 'negative';
    }
    
    return updatedPrompt;
  });
}

/**
 * Ensures the app window has focus for keyboard shortcuts to work
 * This is useful when switching between images or after loading the application
 */
export function ensureWindowFocus(): void {
  // If window doesn't have focus, try to bring focus back
  if (!document.hasFocus()) {
    window.focus();
  }
} 