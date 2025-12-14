import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="h-[60px] bg-blue-400 text-white flex items-center justify-center px-6 shadow-md z-10 relative">
      <h1 className="text-xl font-bold">Semi-Automatic Road Network Annotation Tool</h1>
      <div className="absolute right-6 flex items-center space-x-4">
        {/* Placeholder for future user info/actions */}
        <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center text-gray-600 font-semibold">
          G
        </div>
      </div>
    </header>
  );
};