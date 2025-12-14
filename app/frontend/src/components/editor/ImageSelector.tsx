import React, { useEffect, useRef, ChangeEvent, useState, useCallback } from 'react';
import { useImageStore } from '../../store/imageStore';
import { ImageInfo } from '../../types';
import ActionButton from '../common/ActionButton'; // Import ActionButton for navigation

const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'];

interface ImageSelectorProps {
    requestImageSwitch: (image: ImageInfo) => void;
}

export const ImageSelector: React.FC<ImageSelectorProps> = ({ requestImageSwitch }) => {
    const availableImages = useImageStore((state) => state.availableImages);
    const currentImage = useImageStore((state) => state.currentImage);
    const fetchAvailableImages = useImageStore((state) => state.fetchAvailableImages);
    const uploadAndRefreshImages = useImageStore((state) => state.uploadAndRefreshImages);
    const isFeatureLoading = useImageStore((state) => state.isFeatureLoading);
    const isUploadingStore = useImageStore((state) => state.isUploading);
    const uploadProgress = useImageStore((state) => state.uploadProgress);

    const [isProcessingLocal, setIsProcessingLocal] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const dirInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        fetchAvailableImages();
    }, [fetchAvailableImages]);

    const handleImageItemClick = useCallback((image: ImageInfo) => {
        if (currentImage?.id !== image.id) {
            requestImageSwitch(image);
        }
    }, [currentImage?.id, requestImageSwitch]);

    // Navigation functions for previous/next image
    const handlePreviousImage = useCallback(() => {
        if (!availableImages || availableImages.length === 0 || !currentImage) return;
        
        const currentIndex = availableImages.findIndex(img => img.id === currentImage.id);
        if (currentIndex > 0) {
            const previousImage = availableImages[currentIndex - 1];
            // Only call requestImageSwitch to ensure consistent save confirmation behavior
            requestImageSwitch(previousImage);
        }
    }, [availableImages, currentImage, requestImageSwitch]);

    const handleNextImage = useCallback(() => {
        if (!availableImages || availableImages.length === 0 || !currentImage) return;
        
        const currentIndex = availableImages.findIndex(img => img.id === currentImage.id);
        if (currentIndex < availableImages.length - 1) {
            const nextImage = availableImages[currentIndex + 1];
            // Only call requestImageSwitch to ensure consistent save confirmation behavior
            requestImageSwitch(nextImage);
        }
    }, [availableImages, currentImage, requestImageSwitch]);

    const triggerFileInput = () => fileInputRef.current?.click();
    const triggerDirInput = () => dirInputRef.current?.click();

    const processFilesAndUpload = async (files: FileList | null) => {
        if (!files || files.length === 0) return;

        setIsProcessingLocal(true);
        setError(null);

        const imageFiles = Array.from(files).filter(file => {
            if (!file.type && file.size === 0 && file.name.indexOf('.') === -1) {
                return false;
            }
            const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
            return ALLOWED_EXTENSIONS.includes(extension);
        });

        if (imageFiles.length === 0) {
            alert('No supported image formats found in the selected folder (jpg, png, tif, webp, bmp).');
            if (fileInputRef.current) fileInputRef.current.value = '';
            if (dirInputRef.current) dirInputRef.current.value = '';
            setIsProcessingLocal(false);
            return;
        }

        try {
            await uploadAndRefreshImages(imageFiles);
        } catch (err) {
            console.error("Error during uploadAndRefreshImages:", err);
            setError(err instanceof Error ? err.message : 'Upload and refresh failed');
        } finally {
            setIsProcessingLocal(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
            if (dirInputRef.current) dirInputRef.current.value = '';
        }
    };

    const handleFileSelected = (event: ChangeEvent<HTMLInputElement>) => {
        processFilesAndUpload(event.target.files);
    };

    const handleDirectorySelected = (event: ChangeEvent<HTMLInputElement>) => {
        processFilesAndUpload(event.target.files);
    };

    const isLoadingOverall = isFeatureLoading || isUploadingStore || isProcessingLocal;

    // Function to get badge text for an image
    const getImageBadges = (img: ImageInfo) => {
        const badges = [];
        
        if (img.features_computed) {
            badges.push('✓ Features computed');
        }
        
        if (img.isAnnotated) {
            badges.push('✓ Annotated');
        }
        
        return badges.length > 0 ? badges.join(', ') : '';
    };

    // Function to get filename with reasonable length
    const getDisplayName = (img: ImageInfo) => {
        const name = img.original_filename || img.name || img.id;
        return name.length > 15 ? `${name.substring(0, 12)}...` : name;
    };

    // Calculate statistics
    const totalImages = availableImages?.length || 0;
    const annotatedImages = availableImages?.filter(img => img.isAnnotated).length || 0;
    const currentIndex = currentImage && availableImages ? 
        availableImages.findIndex(img => img.id === currentImage.id) : -1;

    // Check if navigation buttons should be disabled
    const canGoPrevious = currentIndex > 0;
    const canGoNext = currentIndex >= 0 && currentIndex < totalImages - 1;

    return (
        <div className="bg-gray-50 rounded-md border border-gray-200">
            <div className="p-3 space-y-0">
                {/* Hidden Inputs */}
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelected}
                    className="hidden"
                    accept={ALLOWED_EXTENSIONS.join(',')}
                />
                <input
                    type="file"
                    ref={dirInputRef}
                    onChange={handleDirectorySelected}
                    className="hidden"
                    {...{webkitdirectory: "true", mozdirectory: "true", directory: "true"} as any}
                />

                {/* Action Buttons - Compact design */}
                <div className="flex space-x-1.5 mb-3">
                    <button
                        onClick={triggerFileInput}
                        className="flex-1 px-2 py-1.5 text-[10px] bg-lime-600 text-white rounded-md hover:bg-lime-700 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                        disabled={isLoadingOverall}
                        title="Upload single image file"
                    >
                        Open an image
                    </button>
                    <button
                        onClick={triggerDirInput}
                        className="flex-1 px-2 py-1.5 text-[10px] bg-green-600 text-white rounded-md hover:bg-green-700 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                        disabled={isLoadingOverall}
                        title="Upload folder with images"
                    >
                        Open a folder
                    </button>
                </div>
                
                {/* Status Messages - Compact */}
                {isUploadingStore && (
                    <div className="text-xs text-blue-800 py-2 bg-blue-50 rounded border border-blue-200 mb-3">
                        <div className="px-2 flex items-center justify-between">
                            <span className="font-medium">Uploading images...</span>
                            {uploadProgress && (
                                <span>
                                    {uploadProgress.uploadedFiles}/{uploadProgress.totalFiles}
                                </span>
                            )}
                        </div>
                        <div className="px-2 mt-1">
                            <div className="w-full bg-blue-100 rounded-full h-2 overflow-hidden">
                                <div
                                    className="h-full bg-blue-500 rounded-full transition-all duration-200"
                                    style={{ width: `${uploadProgress?.percent ?? 0}%` }}
                                />
                            </div>
                            {uploadProgress && (
                                <div className="mt-1 flex items-center justify-between">
                                    <span className="text-[10px] text-blue-700 truncate max-w-[180px]" title={uploadProgress.currentFile || ''}>
                                        {uploadProgress.currentFile || ''}
                                    </span>
                                    <span className="text-[10px] text-blue-700">{uploadProgress.percent}%</span>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {isProcessingLocal && !isUploadingStore && (
                    <div className="text-xs text-yellow-600 text-center py-1 bg-yellow-50 rounded border border-yellow-200 mb-3">
                        Processing files...
                    </div>
                )}

                {error && (
                    <div className="text-xs text-red-600 text-center py-1 bg-red-50 rounded border border-red-200 mb-3">
                        Error: {error}
                    </div>
                )}

                {/* Image List - Scrollable */}
                <div className="space-y-4">
                    {isLoadingOverall ? (
                        <div className="text-center py-4 text-xs text-gray-500">
                            Processing...
                        </div>
                    ) : !availableImages || availableImages.length === 0 ? (
                        <div className="text-center py-4 text-xs text-gray-500">
                            Please upload the image first.
                        </div>
                    ) : (
                        <div className="max-h-64 overflow-y-auto space-y-1 border border-gray-200 rounded-md bg-white p-1">
                            {availableImages.map((img: ImageInfo, index: number) => {
                                const isSelected = currentImage?.id === img.id;
                                const badges = getImageBadges(img);
                                const displayName = getDisplayName(img);
                                
                                return (
                                    <div
                                        key={img.id}
                                        onClick={() => handleImageItemClick(img)}
                                        className={`
                                            relative flex items-center justify-between p-2 rounded-md cursor-pointer transition-all duration-150 
                                            ${isSelected 
                                                ? 'bg-blue-100 border-2 border-blue-500 shadow-sm' 
                                                : 'hover:bg-gray-50 border border-transparent'
                                            }
                                            ${img.isAnnotated ? 'bg-green-50' : ''}
                                            ${isLoadingOverall ? 'pointer-events-none opacity-50' : ''}
                                        `}
                                        title={`${img.original_filename || img.name || img.id}${badges ? ` (${badges})` : ''}`}
                                    >
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center space-x-2">
                                                {/* Sequence number */}
                                                <span className={`text-xs font-bold flex-shrink-0 w-6 text-center ${isSelected ? 'text-blue-600' : 'text-gray-500'}`}>
                                                    {index + 1}.
                                                </span>
                                                
                                                {/* Image name */}
                                                <span className={`text-xs font-medium truncate ${isSelected ? 'text-blue-800' : 'text-gray-700'}`}>
                                                    {displayName}
                                                </span>
                                                
                                                {/* Status indicators */}
                                                <div className="flex items-center space-x-1 flex-shrink-0">
                                                    {img.features_computed && (
                                                        <span className="inline-flex items-center px-1 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800" title="特征已计算">
                                                            ✓
                                                        </span>
                                                    )}
                                                    {img.isAnnotated && (
                                                        <span className="inline-flex items-center px-1 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800" title="已标注">
                                                            ✓
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                        
                                        {/* Selected indicator */}
                                        {isSelected && (
                                            <div className="absolute right-1 top-1 w-2 h-2 bg-blue-500 rounded-full"></div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Progress and Navigation Section */}
                {totalImages > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-300">
                        {/* Navigation Row */}
                        <div className="flex items-center justify-center space-x-3">
                            <ActionButton
                                bgColor="bg-[#9b89b3]"
                                textColor="text-white"
                                hoverBgColor="hover:bg-[#845ec2]"
                                onClick={handlePreviousImage}
                                disabled={!canGoPrevious || isLoadingOverall}
                                title="Previous image (Left Arrow)"
                            >
                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                                </svg>
                            </ActionButton>
                            
                            <span className="text-xs text-gray-600">
                                Annotated: {annotatedImages}/{totalImages}
                                <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                    <div 
                                        className="h-full bg-gradient-to-r from-blue-400 to-green-400 transition-all duration-300 ease-out"
                                        style={{ 
                                            width: `${totalImages > 0 ? (annotatedImages / totalImages) * 100 : 0}%` 
                                        }}
                                    />
                                </div>
                            </span>
                            
                            <ActionButton
                                bgColor="bg-[#9b89b3]"
                                textColor="text-white"
                                hoverBgColor="hover:bg-[#845ec2]"
                                onClick={handleNextImage}
                                disabled={!canGoNext || isLoadingOverall}
                                title="Next image (Right Arrow)"
                            >
                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                                </svg>
                            </ActionButton>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}; 