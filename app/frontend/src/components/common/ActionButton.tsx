import React from 'react';

// Define types for ActionButton props
interface ActionButtonProps {
    bgColor: string; // e.g., "bg-blue-600"
    textColor: string; // e.g., "text-white"
    hoverBgColor: string; // e.g., "hover:bg-blue-500"
    onClick?: React.MouseEventHandler<HTMLButtonElement>;
    title?: string;
    children: React.ReactNode;
    isActive?: boolean; // Optional prop to indicate active state
    activeBgColor?: string; // Optional: Background color when active
    activeTextColor?: string; // Optional: Text color when active
    disabled?: boolean; // Optional: Whether the button is disabled
}

// Reusable Action Button component
function ActionButton({ 
    bgColor, 
    textColor,
    hoverBgColor, 
    onClick, 
    title, 
    children, 
    isActive = false, 
    activeBgColor = 'bg-blue-600', // Default active BG
    activeTextColor = 'text-white', // Default active Text
    disabled = false
}: ActionButtonProps) {
    // Base styles common to all buttons (padding, text, flex, etc.)
    const commonStyles = "font-bold py-1 px-2 rounded text-xs mx-0.5 flex items-center justify-center transition-colors duration-150 ease-in-out";

    // Disabled styles
    const disabledStyles = disabled ? "opacity-50 cursor-not-allowed" : "";

    // Determine background color based on active state and disabled state
    const currentBgColor = disabled ? 'bg-gray-400' : (isActive ? activeBgColor : bgColor);
    // Determine text color based on active state and disabled state
    const currentTextColor = disabled ? 'text-gray-600' : (isActive ? activeTextColor : textColor);
    // Hover effect should not apply when disabled
    const hoverEffect = disabled ? '' : (isActive ? `hover:${activeBgColor}` : hoverBgColor);

    return (
        <button
            title={title}
            className={`${commonStyles} ${currentBgColor} ${currentTextColor} ${hoverEffect} ${disabledStyles}`}
            onClick={disabled ? undefined : onClick}
            disabled={disabled}
        >
            {children}
        </button>
    );
}

export default ActionButton; 