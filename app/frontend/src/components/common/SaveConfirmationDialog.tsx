import React from 'react';

interface SaveConfirmationDialogProps {
    isOpen: boolean;
    onSave: () => void;
    onDiscard: () => void;
    onCancel: () => void; // Optional: if you want a cancel that does neither save nor discard, just closes dialog
    itemName?: string; // Optional: name of the item being saved, e.g., "road network"
}

const SaveConfirmationDialog: React.FC<SaveConfirmationDialogProps> = ({
    isOpen,
    onSave,
    onDiscard,
    onCancel,
    itemName = 'current changes'
}) => {
    if (!isOpen) {
        return null;
    }

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-xl max-w-sm w-full">
                <h2 className="text-xl font-semibold mb-4">Unsaved Changes</h2>
                <p className="mb-6 text-gray-700">
                    You have unsaved {itemName}. Would you like to save them before proceeding?
                </p>
                <div className="flex justify-end space-x-3">
                    <button
                        onClick={onCancel} // Or onDiscard if Cancel means discard
                        className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={onDiscard}
                        className="px-4 py-2 text-sm font-medium text-white bg-red-400 rounded-md hover:bg-red-500 focus:outline-none focus:ring-2 focus:ring-red-400"
                    >
                        Discard
                    </button>
                    <button
                        onClick={onSave}
                        className="px-4 py-2 text-sm font-medium text-white bg-green-400 rounded-md hover:bg-green-500 focus:outline-none focus:ring-2 focus:ring-green-400"
                    >
                        Save
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SaveConfirmationDialog; 